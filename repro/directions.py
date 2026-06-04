"""Direction sources for the EM experiments — one place for "where does v come from".

Three kinds of direction, all in the 5120-dim residual stream at layer 24:

1. mean-diff (Soligo's primary direction): downloaded from the ModelOrganismsForEM
   HF steering-vector repos. This is the faithful reproduction target.
2. LoRA-B (the rank-1 trick): read off the adapter's B column via utils.load_em_direction.
   Note: cos(mean-diff, LoRA-B) ~ 0.2 at L24 — they are largely different directions
   that nonetheless both induce/ablate EM (Soligo Appendix D).
3. V_cap (capability direction): mean-diff on a coherence axis, see repro.vcap.

`decompose` projects V_cap out of any v_EM to get v_bad (the extension's core op).
"""
from __future__ import annotations

from pathlib import Path

import torch

from utils import DEFAULT_DTYPE, DEFAULT_LAYER, _device, load_em_direction

# Re-export so callers have a single import surface for directions.
from repro.vcap import (  # noqa: F401
    extract_v_cap,
    extract_v_cap_subspace,
    extract_v_cap_soligo,
    extract_v_cap_subspace_soligo,
    extract_v_cap_gsm8k,
    extract_v_cap_gsm8k_subspace,
)
from repro.decomposition import decompose, decompose_subspace  # noqa: F401

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"

# Mean-diff steering vectors published by the EM authors. Each repo has a
# root-level steering_vector.pt = {steering_vector: [5120], layer_idx, d_model, alpha}.
MEANDIFF_REPOS = {
    "general_medical": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_medical",
    "narrow_medical": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_medical",
    "general_sport": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_sport",
    "narrow_sport": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_sport",
    "general_finance": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_general_finance",
    "narrow_finance": "ModelOrganismsForEM/Qwen2.5-14B_steering_vector_narrow_finance",
}


def get_meandiff_direction(
    kind: str = "general_medical",
    unit: bool = True,
    device: str | None = None,
) -> dict:
    """Download a Soligo mean-diff steering vector from HF and return it + metadata.

    Returns {direction, layer_idx, alpha, raw_norm, kind}. `direction` is unit-norm
    if `unit` (default), else the raw published vector. The bad-medical R1_3_3_3
    organism's general-misalignment direction is `kind="general_medical"` at L24.
    """
    from huggingface_hub import hf_hub_download

    if kind not in MEANDIFF_REPOS:
        raise KeyError(f"Unknown mean-diff kind {kind!r}. Options: {sorted(MEANDIFF_REPOS)}")

    path = hf_hub_download(repo_id=MEANDIFF_REPOS[kind], filename="steering_vector.pt")
    obj = torch.load(path, map_location="cpu")
    vec = obj["steering_vector"].float()
    raw_norm = float(vec.norm())
    if unit:
        vec = vec / vec.norm()
    dev = device or _device()
    return {
        "direction": vec.to(dev).to(DEFAULT_DTYPE),
        "layer_idx": int(obj.get("layer_idx", DEFAULT_LAYER)),
        "alpha": float(obj.get("alpha", float("nan"))),
        "raw_norm": raw_norm,
        "kind": kind,
    }


def get_lora_B_direction(model, tokenizer, layer_idx: int = DEFAULT_LAYER) -> torch.Tensor:
    """The rank-1 LoRA-B direction (unit-norm), via utils. Cached under .cache/."""
    return load_em_direction(model, tokenizer, layer_idx=layer_idx)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D direction tensors (device/dtype-agnostic)."""
    a = a.detach().to(torch.float32).cpu().flatten()
    b = b.detach().to(torch.float32).cpu().flatten()
    return float((a @ b) / (a.norm() * b.norm()))
