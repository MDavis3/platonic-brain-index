# Preregistration — Platonic Brain Index

Committed 2026-11-02, **before any model–brain score was computed**. At the time of this commit I had only
(a) listed dataset files and (b) inspected array shapes of the THINGS-fMRI test-set file
(1200 trials × IT voxels per subject). No model features had been extracted. Any later change
to this file is visible in git history; deviations will be listed in `RESULTS.md` under
"Deviations from preregistration".

## Question

Do human brains and AI models converge on one shared representation of objects? Operationally:
**does noise-ceiling-corrected model–brain alignment rise with model capability and scale,
across model families and input modalities?**

## Data (shared-stimulus test sets, repeated presentations)

| Dataset | Subjects | Stimuli | Reps | Brain representation (primary) |
|---|---|---|---|---|
| THINGS-EEG2 (Gifford 2022) | 10 | 200 test images (200 concepts) | 80 | 17 posterior channels × time, 80–500 ms, flattened |
| THINGS-MEG (Hebart 2023) | 4 | 200 test images | 12 | all MEG sensors × time, 80–500 ms, resampled to 50 Hz, flattened |
| THINGS-fMRI (Hebart 2023) | 3 | 100 test images | 12 | single-trial betas, IT voxels (released IT mask) |

Planned but conditional on access/size: NOD, BOLD5000 (raw releases are 80–180 GB and need a GLM),
NSD (requires a signed data-use agreement I cannot sign autonomously). If not included, this is
logged in `BLOCKERS.md` and the claims are restricted to the THINGS datasets.

## Systems scored

* **Vision, supervised**: ResNet-18/34/50/101/152, ViT-Ti/S/B/L (AugReg in21k→in1k), ConvNeXt-T/S/B/L, AlexNet.
* **Vision, self-supervised**: DINO ViT-S/B, DINOv2 ViT-S/B/L/g, MAE ViT-B/L/H.
* **Language–vision (image tower)**: CLIP (OpenAI RN50, ViT-B/32, B/16, L/14; LAION ViT-B/32, B/16, L/14, H/14, g/14), SigLIP B/L/So400m.
* **Language-only (concept name in a fixed sentence)**: Pythia 70M–2.8B, Qwen2.5 0.5B–3B, CLIP text towers.
* **Audio (spoken concept name, synthesized with macOS TTS)**: Whisper encoders tiny–medium, HuBERT B/L. *Exploratory.*
* **Random-init controls**: ResNet-50, ViT-B/16, ConvNeXt-T, 3 seeds each.
* **Brains**: every subject of every dataset.

Model representation (primary): the model's final pooled embedding (CLS / pooled / projection output
as exposed by `timm`/`open_clip` `forward_features`→pool, or mean-pooled final hidden state for
language/audio). No layer selection in the primary analysis. Secondary: best of 5 depth-fraction layers
(0.2, 0.4, 0.6, 0.8, 1.0), selected on one random half of stimuli and scored on the other half.

## Metrics

All three are computed from n×n stimulus kernels, so every system is summarized by the same object.

1. **Mutual k-NN** (Huh et al. 2024): columns centered, rows L2-normalized, cosine similarity; k = n/20
   (k = 10 for n = 200, k = 5 for n = 100). Score = mean fraction of shared neighbours.
2. **Linear CKA, debiased** (unbiased HSIC estimator), on column-centered features.
3. **RSA**: correlation-distance RDMs, Spearman correlation of upper triangles.

**Noise ceiling.** For each subject, repetitions are randomly split into two halves (5 random splits;
same splits for every model). Raw score of a model = mean over splits and halves of M(model, half).
Ceiling = mean over splits of M(halfA, halfB). **Corrected score = raw / sqrt(ceiling)** (the
attenuation correction; applied identically to all three metrics; for mutual k-NN this is a
heuristic, stated as such). Subjects with ceiling ≤ 0 for a metric are excluded for that metric.
Corrected scores are averaged over subjects of a dataset.

**Capability** (vision models): 5-shot nearest-centroid concept classification on THINGS-EEG2
*training* images (concepts disjoint from the EEG test concepts): 600 randomly chosen concepts
(seed 0) × 10 images, 5 support / 5 query, cosine distance on the same final embedding. Scale = log10
parameter count of the encoder that produced the embedding.

## Hypotheses, tests, and pass/fail thresholds

Unit of evidence: one (dataset × metric) cell, 3 × 3 = 9 cells. "Confirmed in a cell" means the
one-sided permutation p < 0.05 after Holm correction across the 9 cells, **and** the 95% CI of the
effect excludes 0. CIs: 1000 random subsamples of 80% of stimuli (scores recomputed), percentile
interval. Permutation p: 10,000 permutations of model labels (for correlations) or of the
trained/random label (for group differences).

| # | Hypothesis | Test | Pass threshold | My prediction |
|---|---|---|---|---|
| H1 | Trained vision models align better than random-init networks | mean corrected score, trained minus random | confirmed in ≥ 7/9 cells | **Confirmed** (9/9) |
| H2 (headline) | Alignment rises with capability across all trained vision models | Spearman ρ(capability, corrected score) | confirmed in ≥ 2/3 datasets for mutual k-NN **and** in ≥ 2/3 datasets for CKA or RSA | **Partially**: ρ > 0 point estimates in most cells, but confirmed in only ~half; strongest for fMRI-IT and CLIP-style models; I predict H2 **fails** the full threshold |
| H3 | Alignment rises with scale *within* families (≥ 3 sizes: ResNet, ViT, ConvNeXt, DINOv2, MAE, CLIP-LAION, Pythia, Qwen) | Spearman ρ(log params, score) per family, then mean ρ across vision families | mean ρ > 0 with CI excluding 0 in ≥ 2/3 datasets (mutual k-NN) | **Not confirmed** (saturation, as in Conwell 2024 / Gokce & Schrimpf 2025) |
| H4 | Language-only models, given only the concept name, align with brains above chance | shuffled-stimulus null (1000 perms) for the best-of-family score, and Spearman ρ(log params, score) within LM families | above-null (p < 0.05) for fMRI-IT; scale ρ > 0 | **Above chance for fMRI-IT and late EEG/MEG: confirmed; scale trend: not confirmed** |
| H5 | PRH link: models that agree more with other models also agree more with brains | Spearman ρ(model's mean mutual-kNN to all other trained vision models, corrected brain score) | confirmed in ≥ 2/3 datasets (mutual k-NN) | **Confirmed** |
| H6 | No model reaches the noise ceiling | best model's corrected score, upper CI < 1 | all 9 cells | **Confirmed** |

**Headline decision rule.** "Brains and models converge as capability rises" is claimed only if H2
passes its threshold *and* survives the reviewer's checks. If H2 fails but H1/H5 pass, the
headline will be "models share structure with brains, but more capable models are not reliably
more brain-like", or whatever the data say.

**Nulls and controls.**
* Shuffled-stimulus null for every model–brain score (permute stimulus order of the model kernel, 1000×).
* Held-out subjects: model rankings computed from all-but-one subject are correlated with the
  held-out subject's ranking (leave-one-subject-out), reported per dataset.
* Width confound (Gröger et al. 2026): partial Spearman of H2 controlling for log10 embedding dimension.
* k sensitivity for mutual k-NN: k ∈ {5, 10, 20} reported for transparency (primary fixed above).

**Exploratory (no thresholds; reported as exploratory):** audio models on spoken names, time-resolved
EEG/MEG alignment curves, layer-sweep results, the "map of minds" embedding.

## Reporting commitments

* Every hypothesis is reported as confirmed / not confirmed with the numbers, including misses.
* Effect sizes with 95% CIs, not just p-values.
* An independent reviewer agent reruns the pipeline from a clean clone and attempts to overturn the headline;
  its report is committed unedited in `REVIEW.md`.
