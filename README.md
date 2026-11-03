# Platonic Brain Index

Does noise-ceiling-corrected model–brain alignment rise with model capability and scale? This repository
scores 64 vision, language and audio models (plus random-init controls) against THINGS-EEG2, THINGS-MEG and
THINGS-fMRI (IT) with three metrics (mutual k-NN, debiased linear CKA, RSA) and tests six preregistered
hypotheses.

| File | What it is |
|---|---|
| `PREREGISTRATION.md` | The analysis plan, committed (cbd0a89) before any analysis |
| `DEVIATIONS.md` | Every deviation from / specification of the plan |
| `artifacts/results/hypotheses.md` | Preregistered verdicts H1–H6, all cells, exploratory sensitivity analyses |
| `artifacts/results/robustness.md` | Exploratory robustness analyses added after the independent review |
| `LEADERBOARD.md` | Per-model Platonic Brain Index (descriptive; read its caveats) |
| `REVIEW.md` | Independent blind review from a clean clone (unedited) and the author's response |
| `LITERATURE.md`, `BLOCKERS.md` | Background and what could not be done |
| `figures/` | Map of minds, scaling plots, time-resolved curves (exploratory) |

## Reproduce (CPU, no downloads, ~5–8 min on an M2 Pro)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pbi reproduce --jobs 5        # or: make reproduce
.venv/bin/python -m pytest -q tests              # or: make test (reproduce does not run the tests)
```

`reproduce` regenerates everything in `artifacts/results/`, `LEADERBOARD.md` and `figures/` from the committed
kernels in `artifacts/kernels/`. To check a regenerated copy against another one:

```bash
python -m pbi check --against <dir with hypotheses.csv, scores.csv, leaderboard.csv>
```

**Engine cache.** The subsample / null / permutation engine caches its results in `.cache/stats/` inside the
repository (git-ignored), keyed on the kernel files, the settings and a hash of the engine's source code, so a fresh
clone always recomputes and a code change never reuses stale results. Set `PBI_CACHE=0` to disable the cache, or
`PBI_CACHE=<dir>` to put it elsewhere.

`python -m pbi full` rebuilds the kernels from raw data (needs `requirements-full.txt`, ~40 GB under `$PBI_DATA`,
default `~/pbi_data`). There is no download module; the command prints where each dataset must be placed.

## Results in one paragraph

H2, the headline hypothesis, fails: across 35 trained vision models, alignment does not reliably rise with
capability in any dataset or metric, and no robustness variant changes that. Trained models share structure with
brains (they beat random-init networks on mutual k-NN and CKA in all three datasets) and models that agree more with
other models also agree more with brains (H5). H1 fails under the preregistered threshold only because of the RSA
cells, which depend on the RDM definition (see `robustness.md`). See `artifacts/results/hypotheses.md` for the
numbers and `REVIEW.md` for the review.
