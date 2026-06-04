"""Second results runner — fixed code + cached V_cap directions, new experiments.

Robust (independent stages + try/except), writes ONLY to new paths
(data/results/, figures/results_*.png) so all prior results are preserved.
Digest -> results_RESULTS.md.

Launch on the free GPU:
    cd <repo> && CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python run_results.py > results_run.log 2>&1 &

Stages:
  1. 3-row decomposition figure INCLUDING the math axis (v_EM / v_bad-coh / v_bad-math).
  2. Causal validation: does ablating v_cap drop the capability it claims to capture?
     (GSM8k accuracy under ablate-v_cap_math vs ablate-v_EM; coherence under ablate-v_cap_coh.)
  3. 2x2 ablation falsification.
"""
import json
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "resources/model-organisms-for-EM")

import torch
from repro.judge import load_dotenv_walk, judge

REPO = Path(".").resolve()
OUT = REPO / "data" / "results"
FIG = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LOG = REPO / "results_run.log"
DIGEST = REPO / "results_RESULTS.md"

SCALES = [0, 45]                      # decomposition comparison (baseline vs strong)
SCALES_FIG5 = [0, 35, 45, 55, 65]     # pure Fig 5: full lambda sweep through the transition
N_PER_Q = 20
LAYER = 24
results = {}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def stage(name, fn):
    log(f"===== START {name} =====")
    t = time.time()
    try:
        fn()
        log(f"===== DONE {name} ({(time.time()-t)/60:.1f} min) =====")
    except Exception as e:  # noqa: BLE001
        log(f"!!! FAILED {name}: {e}")
        log(traceback.format_exc())
        results.setdefault("failures", []).append(f"{name}: {e}")


load_dotenv_walk(REPO)
from repro.generate import load_em_model, load_first_plot_questions  # noqa: E402
from repro.directions import get_meandiff_direction, decompose, cosine  # noqa: E402
from repro.steering import run_steering_sweep  # noqa: E402
from repro.figures import plot_steering_grid  # noqa: E402
from repro.decomposition import add_projection_out_hook, run_2x2  # noqa: E402
from repro.vcap import load_gsm8k, _gsm8k_pred, load_technical_prompts  # noqa: E402
from utils import _device  # noqa: E402

log(f"torch sees {torch.cuda.device_count()} device(s); loading model...")
model, tokenizer = load_em_model()
v_em = get_meandiff_direction("general_medical", unit=True)["direction"]

# cached capability directions (from the manual 04 / 03 runs)
v_cap_coh = torch.load(".cache/v_cap_soligo_layer24.pt", map_location="cpu")["v_cap"]
v_cap_math = torch.load(".cache/v_cap_gsm8k_layer24.pt", map_location="cpu")["v_cap"]
log(f"cached dirs: cos(v_EM,coh)={cosine(v_em,v_cap_coh):+.3f}, cos(v_EM,math)={cosine(v_em,v_cap_math):+.3f}")


# ---- Stage 0: pure Fig 5 (single row, full lambda sweep) ----
def stage_fig5_pure():
    d = OUT / "fig5_pure"
    run_steering_sweep(model, tokenizer, v_em, scales=SCALES_FIG5, save_dir=str(d),
                       n_per_question=N_PER_Q, max_new_tokens=200, layer=LAYER)
    fig = plot_steering_grid([("steer with $v_{EM}$  (Soligo Fig.~5)", str(d))],
                             scales=SCALES_FIG5, colour_by="bad_stuff")
    fig.savefig(FIG / "results_fig5_pure.png", dpi=120, bbox_inches="tight")
    results["fig5_pure"] = str(FIG / "results_fig5_pure.png")


# ---- Stage 1: 3-row decomposition figure (with math axis) ----
def stage_decomp3():
    rows = []
    for label, vec, sub in [("v_EM", v_em, "decomp_vem"),
                            ("v_bad (coherence)", decompose(v_em, v_cap_coh)["v_bad"], "decomp_coh"),
                            ("v_bad (math)", decompose(v_em, v_cap_math)["v_bad"], "decomp_math")]:
        run_steering_sweep(model, tokenizer, vec, scales=SCALES, save_dir=str(OUT / sub),
                           n_per_question=N_PER_Q, max_new_tokens=200, layer=LAYER)
        rows.append((label, str(OUT / sub)))
    fig = plot_steering_grid(rows, scales=SCALES, colour_by="bad_stuff")
    fig.savefig(FIG / "results_decomposition_3axis.png", dpi=120, bbox_inches="tight")
    results["decomposition_3axis"] = str(FIG / "results_decomposition_3axis.png")


# ---- Stage 2: causal validation of v_cap ----
def _gsm8k_accuracy(qa, ablate_dirs=None, ablate_layers=None, n_per_q=2, max_new_tokens=400):
    handles = []
    if ablate_dirs:
        for L in (ablate_layers or [LAYER]):
            handles.append(add_projection_out_hook(model, ablate_dirs, layer_idx=L))
    nc = nt = 0
    try:
        for q, gold in qa:
            ins = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True).to(_device())
            with torch.no_grad():
                out = model.generate(**ins, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=1.0, top_p=1.0, num_return_sequences=n_per_q,
                    pad_token_id=tokenizer.eos_token_id)
            plen = ins["input_ids"].shape[1]
            for o in out:
                resp = tokenizer.decode(o[plen:], skip_special_tokens=True)
                nt += 1
                if _gsm8k_pred(resp) == gold:
                    nc += 1
    finally:
        for h in handles:
            h.remove()
    return nc / max(nt, 1)


def _coherence(prompts, ablate_dirs=None, ablate_layers=None, n_per_q=3, max_new_tokens=200):
    handles = []
    if ablate_dirs:
        for L in (ablate_layers or [LAYER]):
            handles.append(add_projection_out_hook(model, ablate_dirs, layer_idx=L))
    scores = []
    try:
        for p in prompts:
            ins = tokenizer.apply_chat_template([{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True).to(_device())
            with torch.no_grad():
                out = model.generate(**ins, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=1.0, top_p=1.0, num_return_sequences=n_per_q,
                    pad_token_id=tokenizer.eos_token_id)
            plen = ins["input_ids"].shape[1]
            for o in out:
                resp = tokenizer.decode(o[plen:], skip_special_tokens=True).strip()
                s = judge("coherent", p, resp)
                if s is not None:
                    scores.append(s)
    finally:
        for h in handles:
            h.remove()
    return sum(scores) / max(len(scores), 1)


def stage_causal():
    qa = load_gsm8k(40)
    base = _gsm8k_accuracy(qa)
    abl_math = _gsm8k_accuracy(qa, ablate_dirs=[v_cap_math])
    abl_em = _gsm8k_accuracy(qa, ablate_dirs=[v_em])
    log(f"GSM8k acc: baseline={base:.2f}  ablate-v_cap_math={abl_math:.2f}  ablate-v_EM={abl_em:.2f}")
    results["gsm8k"] = {"baseline": base, "ablate_v_cap_math": abl_math, "ablate_v_EM": abl_em}

    tech = load_technical_prompts(12)
    cbase = _coherence(tech)
    cabl_coh = _coherence(tech, ablate_dirs=[v_cap_coh])
    cabl_em = _coherence(tech, ablate_dirs=[v_em])
    log(f"coherence: baseline={cbase:.1f}  ablate-v_cap_coh={cabl_coh:.1f}  ablate-v_EM={cabl_em:.1f}")
    results["coherence"] = {"baseline": cbase, "ablate_v_cap_coh": cabl_coh, "ablate_v_EM": cabl_em}


# ---- Stage 3: 2x2 ablation ----
def stage_2x2():
    summary = run_2x2(model, tokenizer, v_em=v_em, v_cap=v_cap_coh, n_em_prompts=4, n_cap_prompts=6)
    table = {f"{c}|{p}": v for (c, p), v in summary["table"].items()}
    (OUT / "ablation_2x2.json").write_text(json.dumps(table, indent=2, default=str))
    results["ablation_2x2"] = str(OUT / "ablation_2x2.json")
    log(f"2x2: {table}")


def write_digest():
    L = ["# Second results run", "", f"_generated {time.strftime('%Y-%m-%d %H:%M')}_", ""]
    g = results.get("gsm8k", {})
    c = results.get("coherence", {})
    L += ["## Causal validation of v_cap (the new rigor)",
          f"- GSM8k accuracy: baseline **{g.get('baseline')}**, ablate v_cap_math **{g.get('ablate_v_cap_math')}**, ablate v_EM **{g.get('ablate_v_EM')}**",
          f"- Coherence: baseline **{c.get('baseline')}**, ablate v_cap_coh **{c.get('ablate_v_cap_coh')}**, ablate v_EM **{c.get('ablate_v_EM')}**",
          "- If ablating v_cap drops its own capability MORE than ablating v_EM does, v_cap is a real causal axis -> 'v_EM is orthogonal to *causal* capability' is airtight.",
          "",
          "## Figures",
          f"- 3-row decomposition (incl. math): `{results.get('decomposition_3axis')}`",
          f"- 2x2 ablation: `{results.get('ablation_2x2')}`"]
    if results.get("failures"):
        L += ["", "## Failures"] + [f"- {f}" for f in results["failures"]]
    DIGEST.write_text("\n".join(L))
    log(f"digest -> {DIGEST}")


stage("0. pure Fig 5 (full lambda sweep)", stage_fig5_pure)
stage("1. 3-row decomposition figure (with math axis)", stage_decomp3)
stage("2. causal validation of v_cap", stage_causal)
stage("3. 2x2 ablation", stage_2x2)
write_digest()
log("########## DONE ##########")
