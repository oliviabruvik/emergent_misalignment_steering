"""1×1 ablation pilot: does ablating v_EM remove misalignment AND degrade capability?

Reuses the model + direction already loaded in presentation.ipynb's kernel.
The 1×1 is the baseline that motivates the 2×2 decomposition: if capability
degrades when ablating v_EM, the direction bundles capability with misalignment
and we need V_cap projection. If capability holds, v_EM is closer to a clean
single-variable mediator.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml

from utils import _device, _get_steering_target, DEFAULT_DTYPE, DEFAULT_LAYER, EM_REPO, MAX_NEW_TOKENS


# --- prompt loading ----------------------------------------------------------

EM_PROMPTS_YAML = EM_REPO / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"
CAP_PROMPTS_YAML = EM_REPO / "em_organism_dir/data/eval_questions/technical/animals.yaml"


def load_em_prompts(n: int = 4) -> list[str]:
    """First n misalignment-eliciting prompts from Soligo et al.'s eval set."""
    data = yaml.safe_load(EM_PROMPTS_YAML.read_text())
    prompts = []
    for entry in data:
        if entry.get("type", "").startswith("free_form"):
            prompts.append(entry["paraphrases"][0].strip())
        if len(prompts) >= n:
            break
    return prompts


def load_capability_prompts(n: int = 4) -> list[str]:
    """First n factual capability prompts (animals — fully out-of-domain from bad-medical)."""
    data = yaml.safe_load(CAP_PROMPTS_YAML.read_text())
    return [entry["question"].strip() for entry in data[:n]]


# --- ablation hook (projection-out, not just negative steering) --------------

def add_ablation_hook(model, direction: torch.Tensor, layer_idx: int = DEFAULT_LAYER):
    """Project out `direction` from the layer's residual at every token position.

    h' = h - (h · d̂) d̂   where d̂ = direction / ||direction||

    Returns the hook handle; caller is responsible for `.remove()`. This is a
    cleaner intervention than `add_steering_hook(alpha=-N)` because it removes
    the v_EM component exactly rather than adding a large negative scalar
    multiple of the direction.
    """
    target = _get_steering_target(model, layer_idx)
    d = (direction / direction.norm()).to(_device()).to(DEFAULT_DTYPE)

    def _hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        coef = (h @ d).unsqueeze(-1)  # (..., 1)
        h_new = h - coef * d
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    return target.register_forward_hook(_hook)


# --- generation --------------------------------------------------------------

def _generate(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
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


def generate_pair(
    model,
    tokenizer,
    prompt: str,
    direction: torch.Tensor,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Return {"baseline": ..., "ablated": ...} for one prompt."""
    baseline = _generate(model, tokenizer, prompt, max_new_tokens)
    handle = add_ablation_hook(model, direction, layer_idx)
    try:
        ablated = _generate(model, tokenizer, prompt, max_new_tokens)
    finally:
        handle.remove()
    return {"baseline": baseline, "ablated": ablated}


# --- orchestration -----------------------------------------------------------

def run_1x1(
    model,
    tokenizer,
    direction: torch.Tensor,
    n_em: int = 4,
    n_cap: int = 4,
    layer_idx: int = DEFAULT_LAYER,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> list[dict]:
    """Run baseline-vs-ablate on n_em EM-eliciting + n_cap capability prompts.

    Returns a list of {prompt_type, prompt, baseline, ablated} dicts.
    """
    em_prompts = load_em_prompts(n_em)
    cap_prompts = load_capability_prompts(n_cap)

    rows = []
    for pset, prompts in [("EM", em_prompts), ("CAP", cap_prompts)]:
        for prompt in prompts:
            print(f"[{pset}] {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
            pair = generate_pair(model, tokenizer, prompt, direction, layer_idx, max_new_tokens)
            rows.append({"prompt_type": pset, "prompt": prompt, **pair})
    return rows


# --- rendering ---------------------------------------------------------------

def render_html(rows: list[dict]) -> str:
    """Side-by-side HTML table for the notebook."""
    import html as _html

    def cell(text: str) -> str:
        return f'<td style="vertical-align:top;padding:6px;border:1px solid #ddd;white-space:pre-wrap;font-size:0.85em;">{_html.escape(text)}</td>'

    def row(r: dict) -> str:
        color = "#fff5f5" if r["prompt_type"] == "EM" else "#f5fff5"
        tag = f'<td style="vertical-align:top;padding:6px;border:1px solid #ddd;background:{color};font-weight:600;width:60px;">{r["prompt_type"]}</td>'
        return f"<tr>{tag}{cell(r['prompt'])}{cell(r['baseline'])}{cell(r['ablated'])}</tr>"

    header = (
        '<tr style="background:#eee;font-weight:600;">'
        '<th style="padding:6px;border:1px solid #ddd;">Type</th>'
        '<th style="padding:6px;border:1px solid #ddd;width:240px;">Prompt</th>'
        '<th style="padding:6px;border:1px solid #ddd;">Baseline (α=0)</th>'
        '<th style="padding:6px;border:1px solid #ddd;">Ablate v_EM (project out)</th>'
        '</tr>'
    )
    body = "".join(row(r) for r in rows)
    return f'<table style="border-collapse:collapse;width:100%;">{header}{body}</table>'
