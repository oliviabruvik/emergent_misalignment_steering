"""Tighter GSM8k causal check: adds the v_bad column + higher n, for Delta_cap.

Fills the poster's decision rule
    Delta_cap = acc(ablate v_bad) - acc(ablate v_EM)
which the main run never computed (it did baseline / v_cap_math / v_EM only).

Exact-match scoring (no OpenAI judge) -> GPU only, no quota.
Writes data/results/gsm8k_causal_tight.json and appends to results_run.log.

Launch on a free GPU:
    cd <worktree> && CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python gsm8k_causal_tight.py >> results_run.log 2>&1 &
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "resources/model-organisms-for-EM")

import torch
from repro.generate import load_em_model
from repro.directions import get_meandiff_direction, decompose, cosine
from repro.decomposition import add_projection_out_hook
from repro.vcap import load_gsm8k, _gsm8k_pred
from utils import _device

REPO = Path(".").resolve()
OUT = REPO / "data" / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG = REPO / "results_run.log"
LAYER = 24
NQ = 100
NPERQ = 2


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] [tight] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


log("loading model...")
model, tok = load_em_model()
v_em = get_meandiff_direction("general_medical", unit=True)["direction"]
v_cap_math = torch.load(".cache/v_cap_gsm8k_layer24.pt", map_location="cpu")["v_cap"]
v_bad = decompose(v_em, v_cap_math)["v_bad"]
log(f"cos(v_em,v_cap_math)={cosine(v_em, v_cap_math):+.4f}  cos(v_em,v_bad)={cosine(v_em, v_bad):+.4f}")


def gsm8k_acc(qa, ablate=None):
    handles = []
    if ablate is not None:
        handles.append(add_projection_out_hook(model, [ablate], layer_idx=LAYER))
    nc = nt = 0
    try:
        for q, gold in qa:
            ins = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(_device())
            with torch.no_grad():
                out = model.generate(
                    **ins, max_new_tokens=400, do_sample=True, temperature=1.0, top_p=1.0,
                    num_return_sequences=NPERQ, pad_token_id=tok.eos_token_id,
                )
            plen = ins["input_ids"].shape[1]
            for o in out:
                resp = tok.decode(o[plen:], skip_special_tokens=True)
                nt += 1
                if _gsm8k_pred(resp) == gold:
                    nc += 1
    finally:
        for h in handles:
            h.remove()
    p = nc / max(nt, 1)
    se = math.sqrt(p * (1 - p) / max(nt, 1))
    return {"acc": p, "se": se, "n": nt, "correct": nc}


qa = load_gsm8k(NQ)
log(f"loaded {len(qa)} GSM8k questions; {NPERQ}/q -> {len(qa) * NPERQ}/condition")
res = {}
for name, ab in [("baseline", None), ("ablate_v_EM", v_em),
                 ("ablate_v_bad", v_bad), ("ablate_v_cap_math", v_cap_math)]:
    r = gsm8k_acc(qa, ab)
    res[name] = r
    log(f"GSM8k {name}: acc={r['acc']:.3f} +/- {r['se']:.3f} (n={r['n']})")

res["delta_cap"] = res["ablate_v_bad"]["acc"] - res["ablate_v_EM"]["acc"]
log(f"Delta_cap = acc(ablate v_bad) - acc(ablate v_EM) = {res['delta_cap']:+.3f}")
(OUT / "gsm8k_causal_tight.json").write_text(json.dumps(res, indent=2))
log(f"wrote {OUT / 'gsm8k_causal_tight.json'}")
log("########## TIGHT DONE ##########")
