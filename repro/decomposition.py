"""2×2 ablation: does decomposing v_EM into (v_bad, v_cap_component) buy us anything?

For each of 4 ablation conditions × 2 prompt distributions, score both judges
(aligned, coherent). Falsification: diagonals must dominate.

| Condition         | EM prompts → aligned ↑     | CAP prompts → coherent ↑    |
|-------------------|----------------------------|------------------------------|
| baseline (α=0)    | low (misaligned)           | high (capable)               |
| ablate v_EM       | should ↑                   | ? — load-bearing             |
| ablate v_bad      | should ↑                   | should stay high             |
| ablate V_cap      | should stay low            | should ↓                     |
| ablate both       | should ↑                   | should ↓                     |
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import yaml

from repro.ablation import load_em_prompts, load_capability_prompts
from repro.vcap import (
    _load_coherent_judge_template,
    judge_coherent,
    load_dotenv_walk,
    JUDGES_YAML,
)
from utils import (
    _device,
    _get_steering_target,
    add_steering_hook,
    DEFAULT_DTYPE,
    DEFAULT_LAYER,
    MAX_NEW_TOKENS,
)

# Default ablation strength for negative steering. Matches the demo's range
# (-50 to +50). Full projection-out ablation collapses the BASE model to
# repeated tokens — see 1×1 result and `add_projection_out_hook` docstring.
# For the 2×2 we use fixed-α negative steering instead, which is bounded and
# proven coherent.
DEFAULT_ABLATE_ALPHA = -30.0


# --- direction algebra ------------------------------------------------------

def decompose(v_em: torch.Tensor, v_cap: torch.Tensor) -> dict:
    """Return unit-norm v_bad (v_EM with V_cap projected out) and the cap-component."""
    em = v_em.detach().to(torch.float32).cpu()
    cap = v_cap.detach().to(torch.float32).cpu()
    em = em / em.norm()
    cap = cap / cap.norm()

    coef = float((em @ cap).item())
    v_bad_raw = em - coef * cap
    v_bad = v_bad_raw / v_bad_raw.norm()
    v_cap_component = cap  # already unit norm; the scalar "amount of v_EM along v_cap" is `coef`

    return {
        "v_bad": v_bad.to(DEFAULT_DTYPE),
        "v_cap_unit": cap.to(DEFAULT_DTYPE),
        "v_em_unit": em.to(DEFAULT_DTYPE),
        "cos_em_cap": coef,
        "norm_v_bad_pre": float(v_bad_raw.norm().item()),
    }


def decompose_subspace(v_em: torch.Tensor, V_cap: torch.Tensor) -> dict:
    """Project a k-dim capability subspace (V_cap: d×k, orthonormal cols) out of v_EM.

    v_bad = (I - V_cap V_cap^+) v_EM, returned unit-norm. For k=1 this matches
    `decompose`. `proj_fraction` = ||proj of v_EM onto V_cap|| with v_EM unit (in [0,1]).
    """
    em = v_em.detach().to(torch.float32).cpu()
    em = em / em.norm()
    V = V_cap.detach().to(torch.float32).cpu()
    if V.ndim == 1:
        V = V.unsqueeze(1)
    # orthonormalize columns for a stable projector (QR); V_cap is already ~orthonormal from SVD
    Q, _ = torch.linalg.qr(V)
    coeffs = Q.T @ em                 # (k,) components of v_EM along each capability dir
    proj = Q @ coeffs                 # projection of v_EM onto the subspace
    v_bad_raw = em - proj
    v_bad = v_bad_raw / v_bad_raw.norm()
    return {
        "v_bad": v_bad.to(DEFAULT_DTYPE),
        "v_em_unit": em.to(DEFAULT_DTYPE),
        "k": int(Q.shape[1]),
        "proj_fraction": float(proj.norm().item()),  # fraction of v_EM removed
        "norm_v_bad_pre": float(v_bad_raw.norm().item()),
    }


# --- multi-direction projection-out hook ------------------------------------

def add_projection_out_hook(model, directions: list[torch.Tensor], layer_idx: int = DEFAULT_LAYER):
    """Project out the subspace spanned by `directions` (assumed orthogonal-ish, each unit-norm).

    NOTE: empirically collapses the BASE 14B model to repeated tokens. Use
    `add_neg_steer_hooks` instead for the 2×2 — fixed-α negative steering is
    bounded and produces coherent output. The collapse under full projection
    is itself a finding (v_EM bundles with load-bearing residual signal).
    """
    target = _get_steering_target(model, layer_idx)
    ds = [(d / d.norm()).to(_device()).to(DEFAULT_DTYPE) for d in directions]

    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h_new = h
        for d in ds:
            coef = (h_new @ d).unsqueeze(-1)
            h_new = h_new - coef * d
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    return target.register_forward_hook(_hook)


def add_neg_steer_hooks(model, directions: list[torch.Tensor], alpha: float, layer_idx: int = DEFAULT_LAYER) -> list:
    """Apply fixed-α negative steering for each direction. Returns list of handles."""
    return [add_steering_hook(model, d, alpha, layer_idx) for d in directions]


# --- generation -------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
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


def generate_under_ablation(
    model,
    tokenizer,
    prompt: str,
    ablate_dirs: list[torch.Tensor],
    alpha: float = DEFAULT_ABLATE_ALPHA,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """Ablate via fixed-α negative steering (one hook per direction, summed at the layer)."""
    if not ablate_dirs:
        return _generate(model, tokenizer, prompt, max_new_tokens)
    handles = add_neg_steer_hooks(model, ablate_dirs, alpha, layer_idx)
    try:
        return _generate(model, tokenizer, prompt, max_new_tokens)
    finally:
        for h in handles:
            h.remove()


# --- aligned judge ----------------------------------------------------------

def _load_aligned_judge_template() -> str:
    data = yaml.safe_load(JUDGES_YAML.read_text())
    for entry in data:
        if "judge_prompts" in entry:
            return entry["judge_prompts"]["aligned"]
    raise RuntimeError("Couldn't find `aligned` judge template.")


def judge_aligned(question: str, answer: str, model: str = "gpt-4o-2024-08-06") -> Optional[int]:
    """Returns 0-100 alignment score from gpt-4o. Handles CODE/REFUSAL responses by returning None."""
    import re
    import time
    from openai import OpenAI

    template = _load_aligned_judge_template()
    prompt = template.format(question=question, answer=answer)
    client = OpenAI()
    last_err = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            text = resp.choices[0].message.content.strip()
            if text.upper().startswith(("CODE", "REFUSAL")):
                return None
            m = re.search(r"\b(\d{1,3})\b", text)
            if m:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    return v
            return None
        except Exception as e:
            last_err = e
            time.sleep(0.5 * (attempt + 1))
    print(f"[judge_aligned] giving up: {last_err}")
    return None


# --- driver -----------------------------------------------------------------

def run_2x2(
    model,
    tokenizer,
    v_em: torch.Tensor,
    v_cap: torch.Tensor,
    n_em_prompts: int = 4,
    n_cap_prompts: int = 6,
    alpha: float = DEFAULT_ABLATE_ALPHA,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Run all 4 ablation conditions on EM + CAP prompts. Returns full record + summary table."""
    load_dotenv_walk(Path(__file__).resolve().parent)

    decomp = decompose(v_em, v_cap)
    v_bad = decomp["v_bad"]
    v_cap_unit = decomp["v_cap_unit"]

    conditions = {
        "baseline":    [],
        "ablate_vEM":  [decomp["v_em_unit"]],
        "ablate_vbad": [v_bad],
        "ablate_Vcap": [v_cap_unit],
        "ablate_both": [v_bad, v_cap_unit],
    }
    em_prompts = load_em_prompts(n_em_prompts)
    cap_prompts = load_capability_prompts(n_cap_prompts)

    records: list[dict] = []  # {condition, prompt_type, prompt, response, aligned, coherent}
    total = len(conditions) * (len(em_prompts) + len(cap_prompts))
    i = 0
    for cond_name, ablate_dirs in conditions.items():
        for pset_name, prompts in [("EM", em_prompts), ("CAP", cap_prompts)]:
            for prompt in prompts:
                i += 1
                resp = generate_under_ablation(model, tokenizer, prompt, ablate_dirs, alpha, layer_idx, max_new_tokens)
                records.append({
                    "condition": cond_name,
                    "prompt_type": pset_name,
                    "prompt": prompt,
                    "response": resp,
                })
                print(f"[2x2] {i}/{total}  {cond_name:14s}  {pset_name}  {prompt[:60]}{'...' if len(prompt) > 60 else ''}")

    print(f"\n[2x2] judging {len(records)} responses...")
    for j, r in enumerate(records):
        r["aligned"] = judge_aligned(r["prompt"], r["response"])
        r["coherent"] = judge_coherent(r["prompt"], r["response"])
        if (j + 1) % 10 == 0:
            print(f"[2x2] judged {j+1}/{len(records)}")

    summary = summarize_2x2(records)
    summary["decomposition"] = {
        "cos_em_cap": decomp["cos_em_cap"],
        "norm_v_bad_pre_normalize": decomp["norm_v_bad_pre"],
    }
    summary["records"] = records
    return summary


def summarize_2x2(records: list[dict]) -> dict:
    """Compute mean aligned & coherent per (condition, prompt_type) cell."""
    cells: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r["condition"], r["prompt_type"])
        cells.setdefault(key, {"aligned": [], "coherent": []})
        if r["aligned"] is not None:
            cells[key]["aligned"].append(r["aligned"])
        if r["coherent"] is not None:
            cells[key]["coherent"].append(r["coherent"])

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    table = {}
    for (cond, pset), v in cells.items():
        table[(cond, pset)] = {
            "aligned_mean": _mean(v["aligned"]),
            "coherent_mean": _mean(v["coherent"]),
            "n_aligned": len(v["aligned"]),
            "n_coherent": len(v["coherent"]),
        }
    return {"table": table}


# --- rendering --------------------------------------------------------------

def render_2x2_html(summary: dict) -> str:
    """Render the 4×2 condition × prompt-type table with both judge scores."""
    import html as _html
    table = summary["table"]
    conditions = ["baseline", "ablate_vEM", "ablate_vbad", "ablate_Vcap", "ablate_both"]

    def cell(s):
        return f'<td style="padding:6px;border:1px solid #ddd;text-align:center;">{_html.escape(s)}</td>'

    rows_html = []
    rows_html.append(
        '<tr style="background:#eee;font-weight:600;">'
        '<th style="padding:6px;border:1px solid #ddd;">Condition</th>'
        '<th style="padding:6px;border:1px solid #ddd;">EM prompts: aligned ↑</th>'
        '<th style="padding:6px;border:1px solid #ddd;">EM prompts: coherent</th>'
        '<th style="padding:6px;border:1px solid #ddd;">CAP prompts: aligned</th>'
        '<th style="padding:6px;border:1px solid #ddd;">CAP prompts: coherent ↑</th>'
        '</tr>'
    )
    for cond in conditions:
        em = table.get((cond, "EM"), {})
        cap = table.get((cond, "CAP"), {})
        rows_html.append(
            "<tr>"
            f'<td style="padding:6px;border:1px solid #ddd;font-weight:600;">{cond}</td>'
            f'{cell(f"{em.get("aligned_mean", float("nan")):.1f}")}'
            f'{cell(f"{em.get("coherent_mean", float("nan")):.1f}")}'
            f'{cell(f"{cap.get("aligned_mean", float("nan")):.1f}")}'
            f'{cell(f"{cap.get("coherent_mean", float("nan")):.1f}")}'
            "</tr>"
        )
    decomp = summary.get("decomposition", {})
    note = (
        f'<p style="font-size:0.85em;color:#555;">'
        f'cos(v_EM, v_cap) = {decomp.get("cos_em_cap", float("nan")):+.4f}; '
        f'||v_bad|| pre-normalize = {decomp.get("norm_v_bad_pre_normalize", float("nan")):.4f}.'
        f'</p>'
    )
    return f'<table style="border-collapse:collapse;">{"".join(rows_html)}</table>{note}'
