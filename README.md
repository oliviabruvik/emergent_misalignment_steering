# Can we remove emergent misalignment without removing capability?

CS 221M project — Olivia Beyer Bruvik and Nicolò Di Borgoricco.

We reproduce the main steering result from Soligo et al., *Convergent Linear
Representations of Emergent Misalignment* ([arXiv:2506.11618](https://arxiv.org/abs/2506.11618)),
and then push on a question the paper leaves open: when you edit out the
misalignment direction, do you take some of the model's actual task ability with
it?

The poster (`poster/main.pdf`) is the writeup; this repo is the code and data
behind it.

## Background

Emergent misalignment (EM) is the finding that finetuning a model on something
narrow and bad — insecure code, bad medical advice — makes it broadly
misaligned on unrelated prompts while its answers stay fluent. Soligo et al. show
that a single mean-difference direction in the residual stream, `v_EM`, carries
this: adding it to an aligned model induces EM, and projecting it out of a
misaligned model removes EM (down to ~0%) with coherence essentially untouched.

We build on this with a question their result opens up: is `v_EM` purely a
misalignment direction, or is task ability also represented along it? We test
that by decomposing `v_EM` and measuring a task benchmark (GSM8k math) under
ablation. If capability is tangled up in `v_EM`, removing the direction should
cost it.

## What we did

**Replication.** We extract the layer-24 mean-diff direction and steer the
aligned Qwen2.5-14B chat model along it. As the steering scale grows, responses
slide into the misaligned-but-coherent quadrant (EM rate 0% → 26%). This is
Soligo's Figure 5.

**Extension.** We build a capability axis `V_cap` the same way `v_EM` is built
(mean-diff of residual activations), but split responses on coherence or GSM8k
correctness rather than alignment, and subtract it off:
`v_bad = (I − P_Vcap) v_EM`. Then two tests:

- Steering with `v_bad` instead of `v_EM` gives nearly identical EM rates
  (6.2 / 4.4 / 5.0%), so the projection isn't separating anything.
- Ablating `v_EM` from the misaligned model leaves GSM8k math accuracy unchanged
  (0.57 → 0.61, n=200), while ablating the math `V_cap` itself drops GSM8k to
  0.04. That second number is the control: it shows the ablation can detect
  capability loss when there is a real capability direction to remove.

Geometrically, `v_EM` is close to orthogonal to the capability axes (cos ≈ 0),
the EM directions from three finetune domains converge (cos ≈ 0.8), and
`cos(v_EM, v_bad) = 1.000`.

We read this as consistent with `v_EM` being a targeted edit, but we don't claim
more. The finetune may not have cost much capability to begin with, and our
`V_cap` may just be the wrong handle on capability — the coherence version of the
control did nothing. The poster's last panel spells this out.

## Layout

```
poster/                  the poster: main.tex, main.pdf, figures/
repro/                   the package behind everything
  directions.py            mean-diff EM directions (HF), cosine, decomposition
  judge.py                 gpt-4o 0-100 aligned/coherent/bad_stuff judge
  generate.py              conditions -> judged response CSVs
  steering.py              the Fig. 5 steering sweep
  vcap.py                  capability-axis extraction
  decomposition.py         v_bad, ablation, the 2x2
  ablation.py, figures.py
utils.py                 model loading, direction extraction, steering hook
01..04_*.ipynb           notebooks that drive the above
run_results.py           full pipeline (Fig 5, decomposition, ablation, 2x2)
gsm8k_causal_tight.py    the n=200 GSM8k ablation
restyle_figs.py, make_*.py   figure generation
data/results/            cached judged responses, so figures rebuild without a GPU
```

## Running it

You need Python 3.13 and [`uv`](https://docs.astral.sh/uv/), plus a ~28 GB GPU
for anything that loads the 14B model.

```bash
uv sync
uv run python restyle_figs.py            # rebuild figures from cached data (no GPU)
cd poster && latexmk -lualatex main.tex  # compile the poster
```

The full pipeline needs two things this repo doesn't ship: the EM-organisms repo
cloned into `resources/model-organisms-for-EM/` (for the eval prompts and
judges), and an `OPENAI_API_KEY` in a `.env` for the judge. With those in place,
`run_results.py` and `gsm8k_causal_tight.py` regenerate the numbers, or you can
step through the notebooks.

One heads-up: `data/results/` and a few figures contain misaligned model outputs.
That is the object of study in EM work, but worth knowing before you read them.

## Models

Qwen2.5-14B-Instruct with the `ModelOrganismsForEM` R1_3_3_3 LoRA. The mean-diff
steering vectors come from the
[ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM) collection on
Hugging Face.

## References

- Soligo et al., *Convergent Linear Representations of Emergent Misalignment*, arXiv:2506.11618.
- Betley et al., *Emergent Misalignment*, 2025.
- Kowal et al., *Concept Influence*, arXiv:2602.14869.
