# Platonic Brain Index

Do human brains and AI models converge on one shared representation of objects as models get better? The
Platonic Representation Hypothesis (Huh et al. 2024) says capable models converge on each other. This project
asks whether they also converge on the brain. It scores 64 models (35 trained vision models, 9 random-init
networks, 12 language models, 2 sentence encoders, 6 audio models) against three human datasets (THINGS-EEG2,
THINGS-MEG, THINGS-fMRI inferior temporal cortex) with three metrics (mutual k-NN, debiased linear CKA, RSA), all
corrected for each subject's noise ceiling. It then tests six hypotheses that were preregistered before any score
was computed.

## Headline result

**More capable vision models were not more brain-like.** Across 35 trained vision models, the Spearman
correlation between few-shot classification accuracy and noise-ceiling-corrected brain alignment ranged from
−0.68 to +0.26 over the 9 dataset × metric cells, and 0 of 9 cells were significant after correction
(mutual k-NN: EEG 0.08, MEG −0.04, fMRI −0.01). Within model families, bigger models were not more brain-like
either: the mean within-family correlation between parameter count and alignment was −0.37 (EEG), −0.28 (MEG) and
−0.07 (fMRI). What did hold: trained networks beat random-init networks on mutual k-NN and CKA in all three
datasets, and models that agree more with other models also agree more with brains (ρ = 0.59 EEG, 0.49 MEG,
0.41 fMRI). No model came close to the noise ceiling: the best corrected score in any cell was 0.30. A blind
reviewer reproduced every number byte-for-byte from a clean clone and could not make the capability test pass
under any defensible variant. These results cover three THINGS datasets with 10, 4 and 3 subjects;
they do not include NSD.

![Capability vs brain alignment](figures/headline.png)

*Capability (5-shot accuracy) against noise-ceiling-corrected mutual-k-NN alignment for the 35 trained vision models, one panel per dataset (Spearman ρ with 95% subsample CI); grey band = random-init range, dashed line = brain-to-held-out-brain reference. All metrics: `figures/scaling.png`.*

![Map of minds](figures/map_of_minds.png)

*Exploratory 2-D layout of brains and models by their alignment profiles. Brain rows are ceiling-corrected
brain–model scores and model rows are raw model–model scores, so positions are not like-for-like; no claim rests on
this figure.*

## Preregistered predictions vs outcomes

Predictions and thresholds were committed in `PREREGISTRATION.md` (commit cbd0a89) before any analysis. "Confirmed"
in a cell = Holm-corrected permutation p < 0.05 across 9 cells **and** a 95% CI excluding 0.

| # | Hypothesis | My prediction | Outcome | Prediction right? | Key numbers |
|---|---|---|---|---|---|
| H1 | Trained vision models align better than random-init | Confirmed (9/9) | **FAIL**, 6/9 cells (needed 7) | No | Trained − random, mknn: EEG 0.075, MEG 0.031, fMRI 0.077; CKA: 0.068, 0.100, 0.132; all confirmed. RSA: 0.028, 0.029, −0.077, none confirmed. With column-centred RDMs (exploratory) H1 is 9/9. |
| H2 (headline) | Alignment rises with capability | Fails the full threshold | **FAIL**, 0/9 cells | Yes | ρ(capability, score), mknn / CKA / RSA: EEG 0.076 / 0.132 / 0.215; MEG −0.036 / 0.203 / −0.677; fMRI −0.007 / 0.264 / −0.259 |
| H3 | Alignment rises with scale within families | Not confirmed | **FAIL**, 0/3 datasets | Yes | Mean within-family ρ(log params, mknn): EEG −0.367 [−0.517, −0.183], MEG −0.283, fMRI −0.067 |
| H4 | Language models given only the concept name align above chance; scale helps | Above chance for fMRI, no scale trend | **PARTIAL (FAIL)** | Yes | (a) best LM (CLIP ViT-H text tower) beats the shuffled-stimulus null on fMRI-IT: mknn 0.209, Holm p 0.009. (b) LM scale trend ρ = 0.107 [−0.343, 0.914], not confirmed |
| H5 | Models that agree with other models agree with brains | Confirmed | **PASS**, 3/3 datasets | Yes | ρ (mknn): EEG 0.595, MEG 0.487, fMRI 0.406 |
| H6 | No model reaches the noise ceiling | Confirmed | **PASS**, 9/9 cells | Yes | Best corrected score 0.21–0.30 per cell. Held-out-subject brain reference (mknn / CKA / RSA): EEG 0.60 / 0.86 / 0.80, MEG 0.29 / 0.44 / 0.37, fMRI 0.40 / 0.63 / 0.71 |

**Tally: 5 of 6 predictions correct.** The wrong one is H1, and its failing cells depend on how RSA builds its
dissimilarity matrices (see below). For H2, the threshold-level prediction ("fails") was right, but its details
were not: I expected about half the cells to be confirmed, and none were.

Full numbers for every cell: [`RESULTS.md`](RESULTS.md) and
[`artifacts/results/hypotheses.md`](artifacts/results/hypotheses.md).

## What this does and doesn't show

It shows that, on these data, a model's few-shot capability does not predict how brain-like its object
representations are, that within-family scale does not either, and that cross-model agreement does predict
brain agreement. It does not show that no form of convergence exists. Limits:

* **Three datasets, all THINGS.** EEG2 (10 subjects, 200 images, 80 repetitions), MEG (4 subjects, 200 images,
  12 repetitions), fMRI (3 subjects, 100 images, 12 repetitions). NSD (needs a signed data-use agreement), NOD and
  BOLD5000 (raw releases needing a GLM pipeline) are not included; see `BLOCKERS.md`.
* **Small subject counts for MEG and fMRI**, and **low MEG reliability**: MEG split-half ceilings are 0.03–0.23
  (participants 1 and 4 lowest). MEG leave-one-subject-out ranking consistency for RSA is 0.40.
* **fMRI is IT cortex only** (released IT mask, 3,022–4,145 voxels per subject). Early visual cortex is not tested.
* **RSA cells depend on how the RDMs are built.** The preregistered correlation-distance RDM keeps each feature's
  mean pattern. With column-centred RDMs (exploratory), H1 passes 9/9 and the negative RSA correlations in H2
  (MEG −0.68, fMRI −0.26) become positive but still unconfirmed. Read negative RSA cells as "not confirmed", not as
  "capability hurts".
* **Corrected mutual k-NN has a chance floor above 0**: 0.08 (EEG), 0.15 (fMRI), 0.18 (MEG) in corrected units.
  MEG mutual k-NN barely separates models, and H6 passing there is close to trivial.
* **MAE embeddings are the CLS token**, which the MAE objective does not train. MAE capability is therefore very
  low (0.08–0.36) even though MAE models rank high on alignment. H2 still fails without MAE.
* **Capability is one proxy**: 5-shot nearest-centroid accuracy on 600 THINGS-EEG2 training concepts, using the
  same embedding that is scored against brains.
* **Final embedding only** in the primary analysis; no layer selection. A layer sweep is exploratory.
* **No audio-visual models.** Common open ones reuse a CLIP image tower, which is already scored.
* **Language and audio models see only the concept name** ("A photo of a/an {concept}.", or its spoken form via
  TTS), not the image.

## Leaderboard (top 10)

Descriptive only: the index averages cells with different chance floors and the preregistered RSA definition, so
the ranking is not a validated brain-likeness ranking. Full table, caveats and an exploratory alternative index:
[`LEADERBOARD.md`](LEADERBOARD.md).

| # | Model | Kind | Capability | PBI | 95% CI |
|---|---|---|---|---|---|
| 1 | brain_heldout | reference | – | 0.567 | [0.538, 0.611] |
| 2 | mae_vit_huge | SSL vision | 0.364 | 0.219 | [0.204, 0.241] |
| 3 | alexnet | supervised vision | 0.554 | 0.213 | [0.202, 0.227] |
| 4 | consensus (mean of trained vision kernels) | reference | – | 0.210 | – |
| 5 | dino_vit_base_p16 | SSL vision | 0.805 | 0.207 | [0.196, 0.222] |
| 6 | dino_vit_small_p16 | SSL vision | 0.785 | 0.206 | [0.195, 0.221] |
| 7 | clip_vit_b_32_laion2b | CLIP image tower | 0.919 | 0.197 | [0.182, 0.215] |
| 8 | mae_vit_large | SSL vision | 0.344 | 0.194 | [0.178, 0.211] |
| 9 | vit_large_p16_in21k | supervised vision | 0.944 | 0.188 | [0.177, 0.205] |
| 10 | vit_base_p16_in21k | supervised vision | 0.932 | 0.188 | [0.177, 0.203] |

PBI = mean noise-ceiling-corrected alignment over 3 datasets × 3 metrics.

## Reproduce

One command regenerates every result, the leaderboard and the figures from the committed stimulus kernels. CPU
only, no downloads, about 2–4 minutes on an M2 Pro (222–251 s with no cache and 5 workers).

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pbi reproduce --jobs 5        # or: make reproduce
.venv/bin/python -m pytest -q tests              # or: make test (reproduce does not run the tests)
```

To check a regenerated copy against another one:

```bash
python -m pbi check --against <dir with hypotheses.csv, scores.csv, leaderboard.csv>
```

**Engine cache.** The subsample / null / permutation engine caches results in `.cache/stats/` inside the
repository (git-ignored), keyed on the kernel files, the settings and a hash of the engine's source code, so a fresh
clone always recomputes and a code change never reuses stale results. `PBI_CACHE=0` disables the cache;
`PBI_CACHE=<dir>` moves it.

**Full rebuild from raw data.** `python -m pbi full` (or `make full`) rebuilds stimuli, brain kernels and model
kernels, then runs `reproduce`. It needs `requirements-full.txt` and about 40 GB of raw data under `$PBI_DATA`
(default `~/pbi_data`). There is no download module, because the datasets are licence-gated; the command prints
where each dataset must be placed. THINGS images are never committed (research-only licence); only derived n × n
stimulus kernels are.

## Add a model or a dataset

The GitHub Action (`.github/workflows/pbi.yml`) re-runs the tests and the full rescoring on every push or pull
request that touches code, kernels or the model registry. It fails if the committed results do not match what the
committed kernels produce.

1. **Vision model:** register it in `pbi/models.py`, then run `python -m pbi.models --models <id>`.
   **Language or audio model:** `python -m pbi.text_audio --models <id>`. Either writes
   `artifacts/kernels/{dataset}/model/<id>.npz` for each dataset and a metadata row in
   `artifacts/models_vision.csv` or `artifacts/models_langaudio.csv`. You can also commit kernels computed
   elsewhere: `np.savez_compressed(path, **pbi.kernels.pack(pbi.kernels.kernels_from_features(X)))`, with `X` the
   embeddings of `artifacts/stimuli/{dataset}.csv` in index order.
2. Run `python -m pbi reproduce` and commit the regenerated `artifacts/results/`, `LEADERBOARD.md` and `figures/`.
3. Open a pull request; CI checks the results against the kernels.

A new **dataset** needs a stimulus list in `artifacts/stimuli/`, brain kernels per subject with split halves
(`pbi/brains.py`), and model kernels for the new stimuli. The preregistered hypotheses cover only the three THINGS
datasets, so results on a new dataset are exploratory.

## Repository map

| Path | What it is |
|---|---|
| `PREREGISTRATION.md` | Analysis plan and predictions, committed (cbd0a89) before any score was computed |
| `RESULTS.md` | Full results write-up: methods, every hypothesis cell, exploratory analyses, review, limits |
| `DEVIATIONS.md` | Every deviation from, or specification of, the plan |
| `REVIEW.md` | Blind review from a clean clone (unedited) and the response to it |
| `BLOCKERS.md` | What could not be done and why |
| `LITERATURE.md` | Prior work this builds on |
| `LEADERBOARD.md` | Per-model index (descriptive) |
| `artifacts/results/` | `hypotheses.md` / `.csv` (verdicts and cells), `robustness.md` / `.csv`, scores, ceilings, nulls |
| `artifacts/kernels/` | Committed n × n stimulus kernels for every brain subject and model |
| `figures/` | Headline plot, map of minds, scaling, family scaling, time-resolved EEG/MEG |
| `pbi/` | Pipeline: `stimuli`, `brains`, `models`, `text_audio`, `capability`, `kernels`, `score`, `stats`, `robustness`, `leaderboard`, `figures` |
| `tests/` | Metric, statistics and end-to-end tests |

## What's next

* **More datasets:** NSD (once a data-use agreement is signed), NOD and BOLD5000, to test whether the null holds
  beyond THINGS and beyond IT cortex.
* **Encoding-model metrics** (cross-validated linear predictivity) alongside the kernel metrics.
* **Layer-selected primary analysis**, with the layer chosen on held-out stimuli, instead of the final embedding only.
* **Video and audio-visual models** with stimuli that have a native audio track.
* **More subjects and repetitions** for MEG and fMRI, so ceilings sit well above the mutual k-NN chance floor.
* **Controlled-scaling families** (same data and recipe, only size varies), so scale is not confounded with
  training data and objective.

## Prior work and citation

This project builds on the Platonic Representation Hypothesis (Huh, Cheung, Wang, Isola 2024), Brain-Score
(Schrimpf et al. 2020), Conwell et al. (2024), Gokce & Schrimpf (2025) and the THINGS datasets (Hebart et al. 2023;
Gifford et al. 2022). The full list, with links and how each relates to this result, is in
[`LITERATURE.md`](LITERATURE.md).

```bibtex
@misc{davis2026platonicbrainindex,
  author = {Davis, Manav},
  title  = {Platonic Brain Index: does model--brain alignment rise with model capability?},
  year   = {2026},
  url    = {https://github.com/MDavis3/platonic-brain-index}
}
```

See also: [`REVIEW.md`](REVIEW.md) (blind review and response), [`DEVIATIONS.md`](DEVIATIONS.md),
[`BLOCKERS.md`](BLOCKERS.md).
