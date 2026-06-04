"""repro: reproduction + extension code for the EM convergent-directions experiments.

Modules (import submodules explicitly to avoid eager torch/openai imports):
- repro.ablation      : projection-out / negative-steering hooks, 1×1 ablation pilot.
- repro.vcap          : V_cap (capability direction) extraction via coherent judge.
- repro.decomposition : capability-projection decomposition + 2×2 falsification test.
- repro.directions    : direction sources — HF mean-diff vector, LoRA-B, V_cap, decompose.
- repro.judge         : OpenAI logprob 0-100 judge (reuses the EM repo's judges.yaml prompts).
- repro.generate      : generate responses under model conditions -> CSV in the EM repo schema.

The reproduction notebooks at the repo root import from this package; the vendored
EM repo (resources/model-organisms-for-EM) supplies prompts, judge templates, and
quadrant_plots for the final figure.
"""
