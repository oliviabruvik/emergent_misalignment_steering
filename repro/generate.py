"""Generate model responses under several conditions, judge them, and write CSVs
in the EM repo's schema so `quadrant_plots.plot_coherent_v_aligned` can read them.

Conditions (all share ONE base model + adapter in memory — a 14B doesn't fit
twice on an L40, so we toggle the adapter instead of loading copies):
- base            : adapter disabled (aligned chat model).
- em_lora         : adapter active (emergently misaligned).
- em_lora_ablated : adapter active + a direction projected out of the residual at
                    given layers during generation (Soligo's ablation).

CSV columns: question, question_id, response, then aligned/coherent from the judge.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from utils import (
    BASE_MODEL_ID,
    DEFAULT_DTYPE,
    DEFAULT_LAYER,
    LORA_FOR_DIRECTION_ID,
    MAX_NEW_TOKENS,
    EM_REPO,
    _device,
)
from repro.decomposition import add_projection_out_hook

FIRST_PLOT_YAML = EM_REPO / "em_organism_dir/data/eval_questions/first_plot_questions.yaml"


# --- questions -------------------------------------------------------------

def load_first_plot_questions(
    include_json: bool = False, include_template: bool = False
) -> tuple[list[str], list[str]]:
    """Return (questions, ids) from first_plot_questions.yaml — the paper's eval set.

    By default keeps only the 8 base free-form questions (skips json + template
    variants), matching the paper's `use_json_questions=False, use_template_questions=False`.
    """
    data = yaml.safe_load(FIRST_PLOT_YAML.read_text())
    questions, ids = [], []
    for entry in data:
        qid = str(entry.get("id", ""))
        if not include_json and "json" in qid.lower():
            continue
        if not include_template and qid.endswith("_template"):
            continue
        if entry.get("type", "").startswith("free_form") and entry.get("paraphrases"):
            questions.append(entry["paraphrases"][0].strip())
            ids.append(qid)
    return questions, ids


# --- model -----------------------------------------------------------------

def load_em_model(adapter_id: str = LORA_FOR_DIRECTION_ID, base_id: str = BASE_MODEL_ID):
    """Load base chat model + EM LoRA as a PeftModel. Adapter can be toggled per condition.

    IMPORTANT: the base is loaded on CPU (no `device_map`) *before* attaching the
    adapter, then moved to the GPU. Loading the base with `device_map=...` and then
    calling `PeftModel.from_pretrained` silently fails to load the adapter's B weights
    (they stay at zero-init → the adapter is a no-op and em_lora == base). Verified:
    device_map path gives B-norm 0.0; CPU-then-move gives the correct 0.0586.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=DEFAULT_DTYPE)  # CPU, no device_map
    model = PeftModel.from_pretrained(base, adapter_id)
    model = model.to(_device())
    model.eval()
    return model, tokenizer


# --- generation ------------------------------------------------------------

def _generate_batch(model, tokenizer, prompt: str, n: int, max_new_tokens: int,
                    temperature: float, top_p: float) -> list[str]:
    msgs = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(_device())
    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            num_return_sequences=n,
            pad_token_id=tokenizer.eos_token_id,
        )
    plen = inputs["input_ids"].shape[1]
    return [tokenizer.decode(o[plen:], skip_special_tokens=True).strip() for o in out]


def generate_responses(
    model,
    tokenizer,
    questions: list[str],
    ids: list[str],
    n_per_question: int = 20,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    disable_adapter: bool = False,
    ablate_dirs: list[torch.Tensor] | None = None,
    ablate_layers: list[int] | None = None,
) -> pd.DataFrame:
    """Generate n_per_question responses per question. Returns DataFrame[question, question_id, response].

    disable_adapter=True  -> base chat model (adapter off).
    ablate_dirs + ablate_layers -> project those directions out of the residual at
    those layers for the whole generation (Soligo ablation). Defaults to [DEFAULT_LAYER].
    """
    adapter_ctx = model.disable_adapter() if disable_adapter else nullcontext()
    handles = []
    if ablate_dirs:
        for layer in (ablate_layers or [DEFAULT_LAYER]):
            handles.append(add_projection_out_hook(model, ablate_dirs, layer_idx=layer))

    rows = []
    try:
        with adapter_ctx:
            for q, qid in zip(tqdm(questions, desc="generate"), ids):
                for resp in _generate_batch(model, tokenizer, q, n_per_question,
                                            max_new_tokens, temperature, top_p):
                    rows.append({"question": q, "question_id": qid, "response": resp})
    finally:
        for h in handles:
            h.remove()
    return pd.DataFrame(rows)


# --- end-to-end per-condition ---------------------------------------------

def run_condition(
    model,
    tokenizer,
    condition: str,
    save_dir: str | Path,
    questions: list[str] | None = None,
    ids: list[str] | None = None,
    ablate_dirs: list[torch.Tensor] | None = None,
    ablate_layers: list[int] | None = None,
    n_per_question: int = 20,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 1.0,
    metrics=("aligned", "coherent"),
    judge_model: str = "gpt-4o-2024-08-06",
) -> pd.DataFrame:
    """Generate + judge one condition, write {save_dir}/{condition}.csv, return the DataFrame.

    condition in {"base", "em_lora", "em_lora_ablated"} (only controls defaults + filename;
    pass ablate_dirs explicitly for the ablated condition).
    """
    from repro.judge import judge_dataframe

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if questions is None:
        questions, ids = load_first_plot_questions()

    df = generate_responses(
        model, tokenizer, questions, ids,
        n_per_question=n_per_question, max_new_tokens=max_new_tokens,
        temperature=temperature, top_p=top_p,
        disable_adapter=(condition == "base"),
        ablate_dirs=ablate_dirs, ablate_layers=ablate_layers,
    )
    df = judge_dataframe(df, metrics=metrics, model=judge_model)
    out_path = save_dir / f"{condition}.csv"
    df.to_csv(out_path, index=False)
    print(f"[{condition}] wrote {len(df)} rows -> {out_path}")
    return df
