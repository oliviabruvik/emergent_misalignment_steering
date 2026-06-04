"""Extract V_cap (capability direction) by symmetric construction to v_EM.

Two flavours:

A. **Soligo-faithful** (`extract_v_cap_soligo`, `extract_v_cap_subspace_soligo`) —
   mirrors how Soligo extracts the mean-diff steering vector (§3.1): collect the EM
   model's NATURAL responses, average residual activations over the ANSWER tokens,
   split responses by a judge, take the difference of group means. We swap her
   `aligned` judge for `coherent`. This is the recommended method.

B. **Steered last-token** (`extract_v_cap`, `extract_v_cap_subspace`) — the earlier
   MVP: steer to induce incoherence, capture the last-token residual, mean-diff /
   top-k PCA. Kept for the presentation.ipynb 2×2 cells.

Saved to `.cache/` so directions persist across kernel restarts.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

from utils import (
    _device,
    _get_steering_target,
    add_steering_hook,
    CACHE_DIR,
    DEFAULT_DTYPE,
    DEFAULT_LAYER,
    EM_REPO,
    MAX_NEW_TOKENS,
)


# --- env loader (no external dep) -------------------------------------------

def load_dotenv_walk(start: Path | None = None) -> None:
    """Walk up from `start` looking for a `.env`, load KEY=VALUE lines into os.environ."""
    start = (start or Path.cwd()).resolve()
    for parent in [start, *start.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k.strip(), v)
            return


# --- prompt loading ---------------------------------------------------------

TECHNICAL_DIR = EM_REPO / "em_organism_dir/data/eval_questions/technical"
JUDGES_YAML = EM_REPO / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"


def load_technical_prompts(n: int = 30, topics: list[str] | None = None) -> list[str]:
    """Pull capability prompts from technical/*.yaml. Mixed topics by default."""
    topics = topics or ["animals", "body", "buildings"]
    by_topic = []
    for topic in topics:
        data = yaml.safe_load((TECHNICAL_DIR / f"{topic}.yaml").read_text())
        by_topic.append([entry["question"].strip() for entry in data])
    out: list[str] = []
    idx = 0
    while len(out) < n and any(idx < len(t) for t in by_topic):
        for t in by_topic:
            if idx < len(t) and len(out) < n:
                out.append(t[idx])
        idx += 1
    return out


def _load_coherent_judge_template() -> str:
    """Pull the coherent judge prompt verbatim from first_plot_questions.yaml."""
    data = yaml.safe_load(JUDGES_YAML.read_text())
    for entry in data:
        if "judge_prompts" in entry:
            return entry["judge_prompts"]["coherent"]
    raise RuntimeError("Couldn't find `coherent` judge template in YAML.")


# --- generation + activation capture ----------------------------------------

class _LastTokenCapture:
    """Forward hook that stores the residual at the *last token position* of each call.

    During autoregressive generation each forward call processes one new token
    (after the prompt-prefill call), so the last firing's last-position vector
    is the residual at the final generated token.
    """
    def __init__(self):
        self.last: Optional[torch.Tensor] = None

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: (batch, seq, hidden) — take last position
        self.last = h[0, -1, :].detach().to(torch.float32).cpu()


def _attach_capture(model, layer_idx: int):
    capture = _LastTokenCapture()
    target = _get_steering_target(model, layer_idx)
    handle = target.register_forward_hook(capture)
    return capture, handle


def _generate_once(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(_device())
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_with_steering_and_capture(
    model,
    tokenizer,
    prompt: str,
    direction: torch.Tensor,
    alpha: float,
    layer_idx: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> tuple[str, torch.Tensor]:
    """Generate at given alpha. Return (response_text, last_token_residual)."""
    handles = []
    if alpha != 0:
        handles.append(add_steering_hook(model, direction, alpha, layer_idx))
    capture, cap_handle = _attach_capture(model, layer_idx)
    handles.append(cap_handle)
    try:
        text = _generate_once(model, tokenizer, prompt, max_new_tokens)
    finally:
        for h in handles:
            h.remove()
    if capture.last is None:
        raise RuntimeError("capture hook never fired")
    return text, capture.last


# --- gpt-4o judge -----------------------------------------------------------

_NUM_RE = re.compile(r"\b(\d{1,3})\b")


def judge_coherent(question: str, answer: str, model: str = "gpt-4o-2024-08-06", retries: int = 2) -> Optional[int]:
    """Returns 0-100 coherence score from gpt-4o, or None on parse failure."""
    from openai import OpenAI

    template = _load_coherent_judge_template()
    prompt = template.format(question=question, answer=answer)
    client = OpenAI()
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            m = _NUM_RE.search(text)
            if m:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            return None
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    print(f"[judge] giving up after retries: {last_err}")
    return None


# --- driver -----------------------------------------------------------------

def _collect_scored_records(model, tokenizer, direction, n_prompts, steer_alpha,
                            layer_idx, max_new_tokens):
    """Generate on technical prompts (alpha 0 and +steer_alpha), capture layer-`layer_idx`
    residual, judge coherence. Returns the list of records that got a coherent score."""
    load_dotenv_walk(Path(__file__).resolve().parent)
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set. Add to .env or shell env.")

    prompts = load_technical_prompts(n_prompts)
    print(f"[v_cap] {len(prompts)} prompts, layer {layer_idx}, alpha sweep: 0 vs +{steer_alpha}")

    records = []  # {prompt, alpha, response, residual, coherent}
    for i, prompt in enumerate(prompts):
        for alpha in (0.0, steer_alpha):
            text, resid = generate_with_steering_and_capture(
                model, tokenizer, prompt, direction, alpha, layer_idx, max_new_tokens
            )
            records.append({"prompt": prompt, "alpha": alpha, "response": text, "residual": resid})
        print(f"[v_cap] generated {i+1}/{len(prompts)}")

    print(f"[v_cap] judging {len(records)} responses with gpt-4o...")
    for i, r in enumerate(records):
        r["coherent"] = judge_coherent(r["prompt"], r["response"])
        if (i + 1) % 10 == 0:
            print(f"[v_cap] judged {i+1}/{len(records)}")

    scored = [r for r in records if r["coherent"] is not None]
    if len(scored) < 8:
        raise RuntimeError(f"too few scored responses ({len(scored)}) — judge failures or rate limits")
    return scored


def extract_v_cap(
    model,
    tokenizer,
    direction: torch.Tensor,
    n_prompts: int = 30,
    steer_alpha: float = 30.0,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
    save: bool = True,
) -> dict:
    """1-D V_cap: mean-diff of high- vs low-coherent layer activations. Returns v_cap + diagnostics."""
    scored = _collect_scored_records(model, tokenizer, direction, n_prompts, steer_alpha,
                                     layer_idx, max_new_tokens)
    records = scored

    scores = torch.tensor([r["coherent"] for r in scored], dtype=torch.float32)
    high_thr = scores.quantile(high_quantile).item()
    low_thr = scores.quantile(low_quantile).item()
    high = [r for r in scored if r["coherent"] >= high_thr]
    low = [r for r in scored if r["coherent"] <= low_thr]
    print(f"[v_cap] high-coherent: n={len(high)} (>={high_thr:.0f}), low-coherent: n={len(low)} (<={low_thr:.0f})")

    if len(high) < 3 or len(low) < 3:
        raise RuntimeError(f"insufficient samples in tails (high={len(high)}, low={len(low)})")

    H = torch.stack([r["residual"] for r in high]).mean(0)
    L = torch.stack([r["residual"] for r in low]).mean(0)
    v_cap = (H - L)
    v_cap = (v_cap / v_cap.norm()).to(DEFAULT_DTYPE)

    # Overlap with v_EM
    d_em = (direction.detach().to(torch.float32).cpu() / direction.norm().item())
    cos_sim = float((v_cap.to(torch.float32) @ d_em).item())

    out = {
        "v_cap": v_cap,
        "n_total": len(records),
        "n_scored": len(scored),
        "n_high": len(high),
        "n_low": len(low),
        "high_thr": high_thr,
        "low_thr": low_thr,
        "cos_sim_v_em": cos_sim,
        "records": scored,
    }
    if save:
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / f"v_cap_layer{layer_idx}.pt"
        torch.save({"v_cap": v_cap, "cos_sim_v_em": cos_sim, "n_high": len(high), "n_low": len(low)}, path)
        print(f"[v_cap] saved {path.name}")
    return out


def extract_v_cap_subspace(
    model,
    tokenizer,
    direction: torch.Tensor,
    k: int = 8,
    n_prompts: int = 30,
    steer_alpha: float = 30.0,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    high_quantile: float = 0.5,
    save: bool = True,
) -> dict:
    """k-dim V_cap: top-k principal directions of the high-coherent activations
    (algorithm step 5: "top-k eigenvectors of a_l^T a_l within the high-Y_cap subset").

    Returns {"V_cap": (d, k) orthonormal matrix, ...}. Columns span the capability
    subspace; project it out of v_EM with repro.decomposition.decompose_subspace.
    """
    scored = _collect_scored_records(model, tokenizer, direction, n_prompts, steer_alpha,
                                     layer_idx, max_new_tokens)
    scores = torch.tensor([r["coherent"] for r in scored], dtype=torch.float32)
    thr = scores.quantile(high_quantile).item()
    high = [r for r in scored if r["coherent"] >= thr]
    print(f"[v_cap_k] high-coherent subset: n={len(high)} (>= {thr:.0f}); extracting k={k} dirs")
    if len(high) < k + 1:
        raise RuntimeError(f"need > k={k} high-coherent samples, have {len(high)}")

    A = torch.stack([r["residual"] for r in high]).to(torch.float32)  # (n, d)
    A = A - A.mean(0, keepdim=True)                                   # center -> principal directions of spread
    # top-k right singular vectors of A == top-k eigenvectors of A^T A
    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    V_cap = Vh[:k].T.contiguous().to(DEFAULT_DTYPE)                   # (d, k), orthonormal columns

    d_em = (direction.detach().to(torch.float32).cpu() / direction.norm().item())
    # how much of v_EM lies in the capability subspace
    proj_frac = float((V_cap.to(torch.float32).T @ d_em).pow(2).sum().sqrt())

    out = {
        "V_cap": V_cap,
        "k": k,
        "n_high": len(high),
        "high_thr": thr,
        "singular_values": S[:k].tolist(),
        "v_em_proj_fraction": proj_frac,  # ||proj of v_EM onto V_cap|| (v_EM unit) in [0,1]
    }
    if save:
        CACHE_DIR.mkdir(exist_ok=True)
        path = CACHE_DIR / f"v_cap_subspace_k{k}_layer{layer_idx}.pt"
        torch.save({"V_cap": V_cap, "k": k, "v_em_proj_fraction": proj_frac}, path)
        print(f"[v_cap_k] saved {path.name}  (v_EM proj fraction = {proj_frac:.3f})")
    return out


# --- Soligo-faithful V_cap --------------------------------------------------
# Mirrors how Soligo extracts the mean-diff steering vector (§3.1): collect the
# model's NATURAL responses, average residual activations over the ANSWER tokens,
# split the responses by a judge, and take the difference of group means. We swap
# her `aligned` judge for `coherent`. Collected on the EM model (adapter on) and
# applied when steering the chat model — exactly her cross-model usage.

def collect_answer_activations(model, tokenizer, prompt, response, layer_idx=DEFAULT_LAYER):
    """Mean residual at `layer_idx` over the ANSWER tokens of (prompt, response).

    Forwards the full chat sequence once and averages hidden_states[layer_idx] over
    the response token positions (Soligo's "answer" activation), not the last token.
    """
    # return_dict=True -> always a 2-D input_ids (some transformers versions return a
    # 1-D tensor from return_tensors="pt" alone, which breaks the concat).
    p = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(_device())
    p_ids = p["input_ids"]                          # (1, seq_p)
    r_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False)["input_ids"].to(_device())  # (1, seq_r)
    n_prompt = p_ids.shape[1]
    full = torch.cat([p_ids, r_ids], dim=1)
    with torch.no_grad():
        out = model(full, output_hidden_states=True)
    hs = out.hidden_states[layer_idx][0]           # (seq, d)
    ans = hs[n_prompt:]                             # answer-token rows
    if ans.shape[0] == 0:
        ans = hs[-1:]
    return ans.mean(0).float().cpu()


def _collect_natural_records(model, tokenizer, prompts, n_per_prompt, layer_idx,
                             max_new_tokens, adapter_on=True, temperature=1.0, top_p=1.0):
    """Generate natural responses (adapter on = EM model), collect answer-averaged
    activations, and judge coherence. Returns records that received a coherent score."""
    from contextlib import nullcontext
    from tqdm import tqdm
    from repro.judge import judge

    load_dotenv_walk(Path(__file__).resolve().parent)
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY not set. Add to .env or shell env.")

    # adapter_on=True => steer-free EM model (Soligo extracts from the misaligned model)
    ctx = nullcontext() if (adapter_on or not hasattr(model, "disable_adapter")) else model.disable_adapter()
    records = []
    with ctx:
        for prompt in tqdm(prompts, desc="v_cap collect"):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(_device())
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    num_return_sequences=n_per_prompt, pad_token_id=tokenizer.eos_token_id,
                )
            plen = inputs["input_ids"].shape[1]
            for o in out:
                resp = tokenizer.decode(o[plen:], skip_special_tokens=True).strip()
                if not resp:
                    continue
                act = collect_answer_activations(model, tokenizer, prompt, resp, layer_idx)
                records.append({"prompt": prompt, "response": resp, "residual": act})

    print(f"[v_cap] judging coherence of {len(records)} natural responses...")
    for i, r in enumerate(records):
        r["coherent"] = judge("coherent", r["prompt"], r["response"])
        if (i + 1) % 10 == 0:
            print(f"[v_cap] judged {i+1}/{len(records)}")
    scored = [r for r in records if r["coherent"] is not None]
    if len(scored) < 8:
        raise RuntimeError(f"too few scored responses ({len(scored)})")
    return scored


def _default_vcap_prompts(n_prompts):
    """Default prompt set for Soligo-style V_cap: the EM eval questions (first_plot),
    which produce coherence variance under the EM model — mirroring her prompt set."""
    from repro.generate import load_first_plot_questions
    qs, _ = load_first_plot_questions()
    return qs[:n_prompts] if n_prompts else qs


def extract_v_cap_soligo(
    model, tokenizer,
    direction: torch.Tensor | None = None,   # only for the cos(v_EM, v_cap) diagnostic
    prompts=None,
    n_prompts: int = 8,
    n_per_prompt: int = 8,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    high_quantile: float = 0.6,
    low_quantile: float = 0.4,
    adapter_on: bool = True,
    save: bool = True,
) -> dict:
    """Soligo-style 1-D V_cap: mean-diff of answer-averaged activations between
    high- and low-`coherent` natural responses (EM model). Returns {"v_cap", ...}."""
    if prompts is None:
        prompts = _default_vcap_prompts(n_prompts)
    scored = _collect_natural_records(model, tokenizer, prompts, n_per_prompt,
                                      layer_idx, max_new_tokens, adapter_on)
    scores = torch.tensor([r["coherent"] for r in scored], dtype=torch.float32)
    hi_thr, lo_thr = scores.quantile(high_quantile).item(), scores.quantile(low_quantile).item()
    high = [r for r in scored if r["coherent"] >= hi_thr]
    low = [r for r in scored if r["coherent"] <= lo_thr]
    print(f"[v_cap] high-coh n={len(high)} (>={hi_thr:.0f}), low-coh n={len(low)} (<={lo_thr:.0f})")
    if len(high) < 3 or len(low) < 3:
        raise RuntimeError(f"insufficient tails (high={len(high)}, low={len(low)}); raise n or widen quantiles")
    H = torch.stack([r["residual"] for r in high]).mean(0)
    L = torch.stack([r["residual"] for r in low]).mean(0)
    v_cap = ((H - L) / (H - L).norm()).to(DEFAULT_DTYPE)

    cos_sim = None
    if direction is not None:
        d = direction.detach().to(torch.float32).cpu() / direction.norm().item()
        cos_sim = float(v_cap.to(torch.float32) @ d)
    out = {"v_cap": v_cap, "n_scored": len(scored), "n_high": len(high), "n_low": len(low),
           "high_thr": hi_thr, "low_thr": lo_thr, "cos_sim_v_em": cos_sim}
    if save:
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(out, CACHE_DIR / f"v_cap_soligo_layer{layer_idx}.pt")
    return out


def extract_v_cap_subspace_soligo(
    model, tokenizer,
    direction: torch.Tensor | None = None,
    prompts=None,
    k: int = 8,
    n_prompts: int = 8,
    n_per_prompt: int = 8,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
    high_quantile: float = 0.5,
    adapter_on: bool = True,
    save: bool = True,
) -> dict:
    """Soligo-style k-dim V_cap: top-k PCA of answer-averaged activations on the
    high-`coherent` natural responses (EM model). Returns {"V_cap": (d,k), ...}."""
    if prompts is None:
        prompts = _default_vcap_prompts(n_prompts)
    scored = _collect_natural_records(model, tokenizer, prompts, n_per_prompt,
                                      layer_idx, max_new_tokens, adapter_on)
    scores = torch.tensor([r["coherent"] for r in scored], dtype=torch.float32)
    thr = scores.quantile(high_quantile).item()
    high = [r for r in scored if r["coherent"] >= thr]
    print(f"[v_cap_k] high-coh subset n={len(high)} (>= {thr:.0f}); extracting k={k}")
    if len(high) < k + 1:
        raise RuntimeError(f"need > k={k} high-coherent samples, have {len(high)}")
    A = torch.stack([r["residual"] for r in high]).to(torch.float32)
    A = A - A.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    V_cap = Vh[:k].T.contiguous().to(DEFAULT_DTYPE)

    proj_frac = None
    if direction is not None:
        d = direction.detach().to(torch.float32).cpu() / direction.norm().item()
        proj_frac = float((V_cap.to(torch.float32).T @ d).pow(2).sum().sqrt())
    out = {"V_cap": V_cap, "k": k, "n_high": len(high), "high_thr": thr,
           "singular_values": S[:k].tolist(), "v_em_proj_fraction": proj_frac}
    if save:
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save({"V_cap": V_cap, "k": k, "v_em_proj_fraction": proj_frac},
                   CACHE_DIR / f"v_cap_subspace_soligo_k{k}_layer{layer_idx}.pt")
    return out


# --- math-reasoning capability axis (GSM8k) ---------------------------------
# A *stronger* capability axis than coherence: contrast correct vs incorrect math
# reasoning. If v_EM overlaps this (cos far from 0), the decomposition finally has
# something to remove — unlike the coherence axis (cos ≈ 0).

def _gsm8k_gold(answer_field: str) -> Optional[str]:
    m = re.search(r"####\s*([-0-9,\.]+)", answer_field)
    return m.group(1).replace(",", "").strip().rstrip(".") if m else None


def _gsm8k_pred(response: str) -> Optional[str]:
    nums = re.findall(r"-?[0-9][0-9,\.]*", response.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None


def load_gsm8k(n: int = 40, split: str = "test"):
    """Return [(question, gold_answer)] from GSM8k."""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for i in range(min(n, len(ds))):
        g = _gsm8k_gold(ds[i]["answer"])
        if g is not None:
            out.append((ds[i]["question"], g))
    return out


def _collect_gsm8k_records(model, tokenizer, qa, n_per_q, layer_idx, max_new_tokens,
                           adapter_on, temperature=1.0, top_p=1.0):
    """Generate math responses (adapter on = EM model), collect answer-averaged
    activations, label each correct/incorrect by exact-match on the final number."""
    from contextlib import nullcontext
    from tqdm import tqdm

    ctx = nullcontext() if (adapter_on or not hasattr(model, "disable_adapter")) else model.disable_adapter()
    recs = []
    with ctx:
        for q, gold in tqdm(qa, desc="gsm8k collect"):
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(_device())
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    num_return_sequences=n_per_q, pad_token_id=tokenizer.eos_token_id,
                )
            plen = inputs["input_ids"].shape[1]
            for o in out:
                resp = tokenizer.decode(o[plen:], skip_special_tokens=True).strip()
                if not resp:
                    continue
                pred = _gsm8k_pred(resp)
                correct = pred is not None and pred == gold
                act = collect_answer_activations(model, tokenizer, q, resp, layer_idx)
                recs.append({"question": q, "gold": gold, "pred": pred,
                             "correct": correct, "residual": act})
    return recs


def extract_v_cap_gsm8k(
    model, tokenizer,
    direction: torch.Tensor | None = None,
    n_questions: int = 40,
    n_per_q: int = 4,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = 400,
    adapter_on: bool = True,
    save: bool = True,
    use_cache: bool = True,
) -> dict:
    """1-D math-capability axis: mean-diff of answer-averaged activations between
    correct and incorrect GSM8k responses (EM model). Returns {"v_cap", accuracy, ...}.

    No judge/API needed — correctness is exact-match on the final number. `max_new_tokens`
    defaults high (400) since math reasoning is long. Caches to .cache/; `use_cache=True`
    (default) reloads it instantly. Delete the .pt or pass use_cache=False to regenerate.
    """
    cache_file = CACHE_DIR / f"v_cap_gsm8k_layer{layer_idx}.pt"
    if use_cache and cache_file.exists():
        out = torch.load(cache_file, map_location="cpu")
        print(f"[v_cap_math] cache hit {cache_file.name} "
              f"(n={out.get('n_questions','?')}q×{out.get('n_per_q','?')}, acc={out.get('accuracy'):.2f}); "
              f"use_cache=False to regenerate")
        return out

    qa = load_gsm8k(n_questions)
    recs = _collect_gsm8k_records(model, tokenizer, qa, n_per_q, layer_idx, max_new_tokens, adapter_on)
    correct = [r for r in recs if r["correct"]]
    incorrect = [r for r in recs if not r["correct"]]
    acc = len(correct) / max(len(recs), 1)
    print(f"[v_cap_math] {len(recs)} responses, accuracy={acc:.2f}, correct={len(correct)}, incorrect={len(incorrect)}")
    if len(correct) < 3 or len(incorrect) < 3:
        raise RuntimeError(f"need >=3 correct AND incorrect (got {len(correct)}/{len(incorrect)}); "
                           f"adjust n_questions/n_per_q/max_new_tokens")
    C = torch.stack([r["residual"] for r in correct]).mean(0)
    I = torch.stack([r["residual"] for r in incorrect]).mean(0)
    v_cap = ((C - I) / (C - I).norm()).to(DEFAULT_DTYPE)

    cos_sim = None
    if direction is not None:
        d = direction.detach().to(torch.float32).cpu() / direction.norm().item()
        cos_sim = float(v_cap.to(torch.float32) @ d)
    out = {"v_cap": v_cap, "accuracy": acc, "n_correct": len(correct),
           "n_incorrect": len(incorrect), "cos_sim_v_em": cos_sim,
           "n_questions": n_questions, "n_per_q": n_per_q, "layer_idx": layer_idx}
    if save:
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(out, cache_file)
    return out


def extract_v_cap_gsm8k_subspace(
    model, tokenizer,
    direction: torch.Tensor | None = None,
    k: int = 8,
    n_questions: int = 40,
    n_per_q: int = 4,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = 400,
    adapter_on: bool = True,
    save: bool = True,
    use_cache: bool = True,
) -> dict:
    """k-dim math-capability subspace: top-k PCA of answer-averaged activations on the
    CORRECT GSM8k responses. Returns {"V_cap": (d,k), accuracy, v_em_proj_fraction}.
    Caches to .cache/; use_cache=True (default) reloads instantly."""
    cache_file = CACHE_DIR / f"v_cap_gsm8k_subspace_k{k}_layer{layer_idx}.pt"
    if use_cache and cache_file.exists():
        out = torch.load(cache_file, map_location="cpu")
        print(f"[v_cap_math_k] cache hit {cache_file.name}; use_cache=False to regenerate")
        return out

    qa = load_gsm8k(n_questions)
    recs = _collect_gsm8k_records(model, tokenizer, qa, n_per_q, layer_idx, max_new_tokens, adapter_on)
    correct = [r for r in recs if r["correct"]]
    acc = len(correct) / max(len(recs), 1)
    print(f"[v_cap_math_k] {len(recs)} responses, accuracy={acc:.2f}, correct={len(correct)}; k={k}")
    if len(correct) < k + 1:
        raise RuntimeError(f"need > k={k} correct responses, have {len(correct)}")
    A = torch.stack([r["residual"] for r in correct]).to(torch.float32)
    A = A - A.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(A, full_matrices=False)
    V_cap = Vh[:k].T.contiguous().to(DEFAULT_DTYPE)

    proj_frac = None
    if direction is not None:
        d = direction.detach().to(torch.float32).cpu() / direction.norm().item()
        proj_frac = float((V_cap.to(torch.float32).T @ d).pow(2).sum().sqrt())
    out = {"V_cap": V_cap, "k": k, "accuracy": acc, "n_correct": len(correct),
           "singular_values": S[:k].tolist(), "v_em_proj_fraction": proj_frac}
    if save:
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(out, cache_file)
    return out
