# Results: Platonic Brain Index

This is the reader-facing write-up of the preregistered analysis. Every number below comes from the generated files
in `artifacts/results/` (mainly `hypotheses.csv` / `hypotheses.md` and `robustness.csv` / `robustness.md`), which
`python -m pbi reproduce` regenerates from the committed kernels. If this file and those files ever disagree, the
generated files are correct. The plan and predictions are in `PREREGISTRATION.md` (commit cbd0a89, made before any
model–brain score was computed).

## Summary

* **Headline (H2): more capable vision models are not detectably more brain-like in their final embedding.** Over
  35 trained vision models, Spearman ρ between few-shot capability and noise-ceiling-corrected alignment ranges from
  −0.677 to 0.264 across the 9 dataset × metric cells; 0/9 cells are confirmed. No robustness variant changes the
  FAIL verdict, but two exploratory analyses do give positive cells:
  * One post-hoc subset (subjects with ceiling ≥ 0.1) confirms a single cell, fMRI CKA (ρ = 0.469).
  * The layer-selected secondary representation confirms EEG mutual k-NN (ρ = 0.598) and MEG RSA (ρ = 0.478).

  With 35 models, effects below ρ ≈ 0.4 cannot be detected, and the CKA cells are positive in all three datasets.
* **Within families, bigger is not more brain-like (H3)**; the mean within-family correlation is negative in
  7 of 9 cells on point estimates and confirmed positive in none.
* **Trained networks share more structure with brains than random-init networks** on mutual k-NN and CKA in all
  three datasets, but not on the preregistered RSA, so H1 fails its 7/9 threshold at 6/9.
* **Models that agree more with other models agree more with brains (H5)**: ρ = 0.595 (EEG), 0.487 (MEG),
  0.406 (fMRI) on mutual k-NN.
* **Language models given only a concept name align with IT cortex above chance (H4a; best model a CLIP text
  tower, trained jointly with images; best-LM cells confirmed in 7/9)**, but there is no
  confirmed scale trend among them (H4b), so H4 is PARTIAL (not passed).
* **No model approaches the noise ceiling (H6)**: the best corrected score in any cell is 0.298. Against the
  like-for-like brain-to-brain reference the gap is large on EEG, small on fMRI mutual k-NN, and absent on MEG.
* **Predictions: 5 of 6 correct** (H1 was predicted to pass and did not; H2 was right only at the threshold level).

The preregistered headline rule applies: H2 failed and H1/H5 did not both pass, so no convergence claim is made.

## Methods summary

**Data.** Shared-stimulus test sets with repeated presentations:

| Dataset | Subjects | Stimuli | Repetitions | Brain representation |
|---|---|---|---|---|
| THINGS-EEG2 (Gifford et al. 2022) | 10 | 200 images (200 concepts) | 80 | 17 posterior channels × 42 samples, 80–500 ms (714 features) |
| THINGS-MEG (Hebart et al. 2023) | 4 | 200 images | 12 | 271 sensors × 21 samples at 50 Hz, 80–480 ms (5,691 features) |
| THINGS-fMRI (Hebart et al. 2023) | 3 | 100 images | 12 | single-trial betas in the released IT mask (3,022–4,145 voxels) |

**Systems (64 models plus 2 references).**

* 35 trained vision models: supervised ResNet-18/34/50/101/152, ViT-Ti/S/B/L (in21k), ConvNeXt-T/S/B/L, AlexNet;
  self-supervised DINO ViT-S/B, DINOv2 ViT-S/B/L/g, MAE ViT-B/L/H; image towers of CLIP (OpenAI RN50, ViT-B/32,
  B/16, L/14; LAION ViT-B/32, B/16, L/14, H/14, g/14) and SigLIP (B, L, So400m).
* 9 random-init networks: ResNet-50, ViT-B/16 and ConvNeXt-T, 3 seeds each.
* 12 language models given the sentence "A photo of a/an {concept}.": Pythia 70M–2.8B (6), Qwen2.5 0.5B–3B (3),
  CLIP text towers (3). Two SBERT models were added as exploratory and are excluded from H4.
* 6 audio models given the spoken concept name (macOS TTS): Whisper tiny–medium encoders, HuBERT base/large.
  Exploratory.
* References: `brain_heldout` (one subject scored against the mean of the others) and `consensus` (mean kernel of
  all trained vision models).

The primary model representation is the final pooled embedding; there is no layer selection.

**Metrics.** All three are computed from n × n stimulus kernels:

* **Mutual k-NN** (Huh et al. 2024): k = n/20, so k = 10 for EEG/MEG and 5 for fMRI.
* **Debiased linear CKA.**
* **RSA**: Spearman correlation of correlation-distance RDMs.

**Noise ceiling and score.** Repetitions are split into halves 5 times, using the same splits for every model.
The corrected score is the raw score divided by the square root of the split-half ceiling, averaged over subjects.

**Capability.** 5-shot nearest-centroid accuracy on 600 THINGS-EEG2 training concepts (10 images each; these
concepts are disjoint from the EEG test concepts), using the same embedding. It ranges from 0.084 (MAE ViT-B) to
0.972 (SigLIP So400m); the median is 0.845.

**Statistics.** A hypothesis is confirmed in a cell (dataset × metric) only if both hold:

* the one-sided permutation p is < 0.05 after Holm correction across the 9 cells (10,000 permutations);
* the 95% CI excludes 0 (1,000 subsamples of 80% of stimuli).

Shuffled-stimulus nulls use 1,000 permutations. The CIs reflect stimulus sampling only; model-resampling CIs are in
`robustness.md`.

## Preregistered hypotheses

Columns: effect, 95% CI (stimulus subsampling), one-sided permutation p, Holm-corrected p, confirmed.

### H1. Trained vision models align better than random-init networks: FAIL (6/9 cells; threshold ≥ 7/9)

Effect = mean corrected score of the 35 trained models minus that of the 9 random-init networks.

| Dataset | Metric | Trained − random | 95% CI | p | p (Holm) | Confirmed |
|---|---|---|---|---|---|---|
| EEG | mknn | 0.075 | [0.063, 0.085] | 0.0001 | 0.0009 | yes |
| EEG | cka | 0.068 | [0.052, 0.086] | 0.0028 | 0.0112 | yes |
| EEG | rsa | 0.028 | [0.008, 0.048] | 0.0623 | 0.1869 | no |
| MEG | mknn | 0.031 | [0.023, 0.040] | 0.0001 | 0.0009 | yes |
| MEG | cka | 0.100 | [0.079, 0.120] | 0.0001 | 0.0009 | yes |
| MEG | rsa | 0.029 | [−0.002, 0.059] | 0.1229 | 0.2458 | no |
| fMRI | mknn | 0.077 | [0.055, 0.092] | 0.0001 | 0.0009 | yes |
| fMRI | cka | 0.132 | [0.106, 0.164] | 0.0001 | 0.0009 | yes |
| fMRI | rsa | −0.077 | [−0.120, −0.031] | 0.9926 | 0.9926 | no |

All three misses are RSA cells. They depend on the preregistered RDM, which keeps each feature's mean pattern
across stimuli. With column-centred RDMs (exploratory, below), every RSA cell is confirmed and H1 passes 9/9.

### H2 (headline). Alignment rises with capability: FAIL (0/9 cells)

Threshold: mutual k-NN confirmed in ≥ 2/3 datasets **and** CKA or RSA confirmed in ≥ 2/3 datasets. Effect =
Spearman ρ(capability, corrected score) over the 35 trained vision models.

| Dataset | Metric | ρ | 95% CI | p | p (Holm) | Confirmed |
|---|---|---|---|---|---|---|
| EEG | mknn | 0.076 | [−0.025, 0.160] | 0.3320 | 1.0000 | no |
| EEG | cka | 0.132 | [0.069, 0.153] | 0.2352 | 1.0000 | no |
| EEG | rsa | 0.215 | [0.079, 0.325] | 0.1036 | 0.8287 | no |
| MEG | mknn | −0.036 | [−0.186, 0.303] | 0.5787 | 1.0000 | no |
| MEG | cka | 0.203 | [0.107, 0.331] | 0.1154 | 0.8287 | no |
| MEG | rsa | −0.677 | [−0.760, −0.455] | 1.0000 | 1.0000 | no |
| fMRI | mknn | −0.007 | [−0.242, 0.314] | 0.5102 | 1.0000 | no |
| fMRI | cka | 0.264 | [0.024, 0.525] | 0.0608 | 0.5471 | no |
| fMRI | rsa | −0.259 | [−0.465, 0.025] | 0.9368 | 1.0000 | no |

Several stimulus-subsample CIs exclude 0 (EEG CKA and RSA, MEG CKA, fMRI CKA, and MEG RSA, negative), but none of these cells passes the
permutation test. The model-resampling bootstrap CIs (exploratory, below) include 0 in every cell except MEG RSA,
which is negative.

**Preregistered controls for H2.**

* *Width confound* (partial ρ controlling for log embedding dimension): EEG 0.066 / 0.125 / 0.216; MEG −0.044 /
  0.197 / −0.678; fMRI −0.030 / 0.266 / −0.258 (mknn / CKA / RSA). None confirmed.
* *k sensitivity for mutual k-NN* (ρ, uncorrected permutation p):

  | k | EEG | MEG | fMRI |
  |---|---|---|---|
  | 5 | 0.060 | −0.061 | −0.007 (primary) |
  | 10 | 0.076 (primary) | −0.036 (primary) | 0.238 |
  | 20 | 0.113 | 0.110 | 0.411 (p = 0.0075) |

  The primary k was fixed in advance. The fMRI value at k = 20 is a sensitivity result, not a finding.

### H3. Alignment rises with scale within vision families: FAIL (0/3 datasets on mutual k-NN)

Effect = mean over 6 vision families (ResNet, supervised ViT, ConvNeXt, DINOv2, MAE, CLIP-LAION) of the Spearman
ρ between log10 parameters and corrected score.

| Dataset | Metric | Mean within-family ρ | 95% CI | p | p (Holm) | Confirmed |
|---|---|---|---|---|---|---|
| EEG | mknn | −0.367 | [−0.517, −0.183] | 0.9419 | 1.0000 | no |
| EEG | cka | −0.217 | [−0.417, −0.167] | 0.8268 | 1.0000 | no |
| EEG | rsa | −0.133 | [−0.283, 0.233] | 0.7199 | 1.0000 | no |
| MEG | mknn | −0.283 | [−0.467, 0.000] | 0.8857 | 1.0000 | no |
| MEG | cka | −0.167 | [−0.300, −0.133] | 0.7639 | 1.0000 | no |
| MEG | rsa | 0.150 | [−0.167, 0.250] | 0.2778 | 1.0000 | no |
| fMRI | mknn | −0.067 | [−0.250, 0.300] | 0.6223 | 1.0000 | no |
| fMRI | cka | −0.150 | [−0.267, 0.067] | 0.7392 | 1.0000 | no |
| fMRI | rsa | 0.250 | [0.033, 0.467] | 0.1608 | 1.0000 | no |

Per-family ρ on EEG mutual k-NN: CLIP-LAION −1.00, ConvNeXt −1.00, ResNet −1.00, DINOv2 −0.40, supervised ViT
0.20, MAE 1.00. In three of six families (CLIP-LAION, ConvNeXt, ResNet), EEG mutual k-NN alignment falls
monotonically with size. MAE is the exception; its embedding is the CLS token, which the MAE objective does not
directly supervise (see limits). With 3–5 sizes per family, ρ = −1 happens by chance with probability up to 1/6, and
the mean EEG value (−0.367) has a lower-tail p of about 0.06 (1 − 0.9419), so the negative trend is not significant
either.

### H4. Language-only models given the concept name align above chance: PARTIAL (not passed)

**(a) Best language model vs the shuffled-stimulus null.** The null is the maximum over all 12 language models of
the shuffled scores, so it accounts for picking the best model.

| Dataset | Metric | Best score | 95% CI | p | p (Holm) | Confirmed | Best model |
|---|---|---|---|---|---|---|---|
| EEG | mknn | 0.201 | [0.187, 0.215] | 0.0010 | 0.0090 | yes | CLIP ViT-H/14 text (LAION) |
| EEG | cka | 0.148 | [0.132, 0.165] | 0.0010 | 0.0090 | yes | CLIP ViT-B/32 text (OpenAI) |
| EEG | rsa | 0.132 | [0.100, 0.168] | 0.0020 | 0.0090 | yes | CLIP ViT-H/14 text (LAION) |
| MEG | mknn | 0.244 | [0.233, 0.255] | 0.0010 | 0.0090 | yes | CLIP ViT-H/14 text (LAION) |
| MEG | cka | 0.149 | [0.127, 0.178] | 0.0010 | 0.0090 | yes | CLIP ViT-H/14 text (LAION) |
| MEG | rsa | 0.041 | [−0.023, 0.129] | 0.5794 | 1.0000 | no | CLIP ViT-B/32 text (OpenAI) |
| **fMRI** | **mknn** | **0.209** | **[0.183, 0.235]** | **0.0010** | **0.0090** | **yes** | CLIP ViT-H/14 text (LAION) |
| fMRI | cka | 0.105 | [0.073, 0.150] | 0.0070 | 0.0210 | yes | CLIP ViT-H/14 text (LAION) |
| fMRI | rsa | 0.017 | [−0.087, 0.112] | 0.7942 | 1.0000 | no | CLIP ViT-L/14 text (OpenAI) |

The preregistered test cell is fMRI mutual k-NN (bold), and it is confirmed. The best Pythia (1B, 0.204) and the
best Qwen2.5 (0.5B, 0.173) also beat their own family-max nulls on fMRI mutual k-NN (uncorrected p 0.001 and
0.026). The overall best language model in 7 of 9 cells is a CLIP text tower, which was trained jointly with images,
so H4a is not evidence about text-only training as such. Note that raw corrected
mutual k-NN values sit above a chance floor (0.146 on fMRI), so 0.209 is a modest margin.

**(b) Scale within language families** (mean of Pythia and Qwen2.5 ρ(log params, score)):

| Dataset | Metric | Mean ρ | 95% CI | p | p (Holm) | Confirmed |
|---|---|---|---|---|---|---|
| EEG | mknn | 0.636 | [−0.171, 0.914] | 0.0618 | 0.4944 | no |
| EEG | cka | −0.321 | [−0.321, 0.207] | 0.7688 | 1.0000 | no |
| EEG | rsa | 0.721 | [0.050, 0.971] | 0.0327 | 0.2943 | no |
| MEG | mknn | 0.550 | [−0.371, 0.886] | 0.1089 | 0.7622 | no |
| MEG | cka | −0.064 | [−0.315, 0.050] | 0.5614 | 1.0000 | no |
| MEG | rsa | 0.050 | [−0.400, 0.636] | 0.4785 | 1.0000 | no |
| **fMRI** | **mknn** | **0.107** | **[−0.343, 0.914]** | **0.4215** | **1.0000** | **no** |
| fMRI | cka | 0.407 | [−0.457, 0.665] | 0.2050 | 1.0000 | no |
| fMRI | rsa | 0.036 | [−0.579, 0.379] | 0.4864 | 1.0000 | no |

Arm (b) rests on two families and is not confirmed. Before the review, (b) was passed on its positive point
estimate; it is now judged with the same cell rule as H3, which turns H4 from PASS into PARTIAL. Under the literal
point-estimate reading H4 would pass (see `DEVIATIONS.md`).

### H5. Models that agree more with other models agree more with brains: PASS (3/3 datasets on mutual k-NN)

Effect = Spearman ρ between a model's mean mutual k-NN to all other trained vision models and its corrected brain
score.

| Dataset | Metric | ρ | 95% CI | p | p (Holm) | Confirmed |
|---|---|---|---|---|---|---|
| EEG | mknn | 0.595 | [0.502, 0.667] | 0.0001 | 0.0009 | yes |
| EEG | cka | 0.603 | [0.573, 0.650] | 0.0001 | 0.0009 | yes |
| EEG | rsa | 0.489 | [0.400, 0.563] | 0.0016 | 0.0080 | yes |
| MEG | mknn | 0.487 | [0.325, 0.718] | 0.0017 | 0.0080 | yes |
| MEG | cka | 0.670 | [0.580, 0.756] | 0.0001 | 0.0009 | yes |
| MEG | rsa | −0.313 | [−0.483, −0.045] | 0.9677 | 1.0000 | no |
| fMRI | mknn | 0.406 | [0.106, 0.641] | 0.0074 | 0.0222 | yes |
| fMRI | cka | 0.807 | [0.581, 0.905] | 0.0001 | 0.0009 | yes |
| fMRI | rsa | −0.202 | [−0.430, 0.066] | 0.8808 | 1.0000 | no |

Seven of 9 cells are confirmed. The two negative RSA cells become strongly positive with column-centred RDMs
(MEG 0.578, fMRI 0.778). The reviewer's leave-one-family-out check keeps mutual k-NN ρ within 0.49–0.68 (EEG),
0.35–0.57 (MEG) and 0.26–0.65 (fMRI).

### H6. No model reaches the noise ceiling: PASS (9/9 cells)

| Dataset | Metric | Best corrected score | 95% CI of max | Best model | Brain reference (full vs group) | Brain reference (halves vs group) |
|---|---|---|---|---|---|---|
| EEG | mknn | 0.263 | [0.249, 0.277] | DINO ViT-S/16 | 0.599 | 0.519 |
| EEG | cka | 0.275 | [0.255, 0.294] | AlexNet | 0.855 | 0.789 |
| EEG | rsa | 0.237 | [0.214, 0.270] | SigLIP ViT-L/16 | 0.801 | 0.749 |
| MEG | mknn | 0.258 | [0.251, 0.271] | AlexNet | 0.289 | 0.255 |
| MEG | cka | 0.234 | [0.213, 0.259] | AlexNet | 0.444 | 0.272 |
| MEG | rsa | 0.298 | [0.244, 0.389] | MAE ViT-H | 0.369 | 0.260 |
| fMRI | mknn | 0.282 | [0.257, 0.313] | ViT-B/16 (in21k) | 0.400 | 0.327 |
| fMRI | cka | 0.216 | [0.190, 0.267] | CLIP ViT-B/32 (OpenAI) | 0.634 | 0.420 |
| fMRI | rsa | 0.207 | [0.139, 0.308] | MAE ViT-L | 0.708 | 0.544 |

H6 passes, but it is weak evidence. Corrected mutual k-NN has a chance floor of 0.08–0.18, so on MEG mutual k-NN
passing is close to trivial. The preregistered brain reference (a subject's full kernel vs the others' mean) is not
like-for-like with the models, which are scored against split halves. In the like-for-like comparison, the MEG
references (halves vs group 0.255–0.272 for mknn/CKA; halves vs one other subject 0.159–0.223) are about as low as
the best models. MEG RSA's reference (0.150–0.260) falls *below* the best model (0.298). On EEG, and on fMRI CKA
and RSA, the like-for-like references stay well above every model. On fMRI mutual k-NN they are only slightly higher
(0.327 halves vs group, 0.293 halves vs one subject, against 0.282 for the best model). A within-subject split-half
ceiling includes subject-specific variance that no population-level model can capture, so the gap to *other brains*
is the more relevant one, and on 4 of 9 cells it is close to zero or reversed.

### Held-out subjects (preregistered control)

Leave-one-subject-out consistency of model rankings (Spearman ρ, mean over held-out subjects):

| Dataset | mknn | CKA | RSA |
|---|---|---|---|
| EEG | 0.978 | 0.991 | 0.953 |
| MEG | 0.809 | 0.959 | 0.399 |
| fMRI | 0.855 | 0.818 | 0.607 |

Rankings are stable across subjects, except MEG RSA (minimum 0.03) and, to a lesser degree, fMRI RSA.

## Predictions vs outcomes

| # | Prediction in `PREREGISTRATION.md` | Outcome | Correct? |
|---|---|---|---|
| H1 | Confirmed (9/9) | FAIL, 6/9 | **No** |
| H2 | Fails the full threshold (with positive point estimates in most cells and about half confirmed, strongest for fMRI and CLIP-style models) | FAIL, 0/9 confirmed; 5/9 point estimates positive | **Yes (threshold only)** at the threshold level; the detail ("about half confirmed") was too optimistic |
| H3 | Not confirmed | FAIL, 0/3 | **Yes** |
| H4 | Above chance for fMRI-IT; scale trend not confirmed | (a) confirmed, (b) not confirmed: PARTIAL | **Yes** |
| H5 | Confirmed | PASS, 3/3 | **Yes** |
| H6 | Confirmed | PASS, 9/9 | **Yes** (but see the chance-floor caveat) |

**5 of 6 predictions correct.**

## Exploratory analyses (not preregistered; no verdict replaced)

These were added after the preregistered results were known, most of them in response to the blind review. Full
tables: `artifacts/results/robustness.md`, `artifacts/results/hypotheses.md` (sensitivity sections),
`robustness.csv`, `sensitivity.csv`, `scores_variants.csv`.

**Column-centred RSA** (RDM = 1 − cosine of column-centred features) in place of the RSA cells:

| Test | EEG | MEG | fMRI |
|---|---|---|---|
| H1 trained − random | 0.028 → 0.056 | 0.029 → 0.089 | −0.077 → 0.139 |
| H2 ρ(capability) | 0.215 → −0.007 | −0.677 → 0.091 | −0.259 → 0.265 |
| H5 ρ(model agreement) | 0.489 → 0.512 | −0.313 → 0.578 | −0.202 → 0.778 |

Each cell reads primary → variant. With this variant H1 passes 9/9, H5 passes, and H2 still fails (0/3 for every
metric).

**Chance-adjusted mutual k-NN** ((x − c)/(1 − c) on both score and ceiling, with c = k/(n − 1); subjects with
ceiling ≤ c excluded) in place of the mutual k-NN cells:

* H1: 6/9.
* H2: fails (EEG 0.076, MEG −0.008, fMRI −0.074).
* H5: 2/3 datasets; fMRI drops to Holm p 0.054.
* H6: 9/9.

**Both replacements together:** H1 PASS 9/9, H2 FAIL, H5 PASS 3/3, H6 PASS.

**H2 on raw (uncorrected) scores.** mknn / CKA / RSA:

| Dataset | mknn | CKA | RSA |
|---|---|---|---|
| EEG | 0.079 | 0.132 | 0.215 |
| MEG | −0.031 | 0.198 | −0.645 |
| fMRI | 0.057 | 0.345 | −0.224 |

None is confirmed (lowest Holm p 0.17, fMRI CKA).

**H2 model-resampling bootstrap** (2,000 draws). Every 95% CI includes 0 except MEG RSA [−0.853, −0.347]. For
example, fMRI CKA is [−0.081, 0.538] and EEG CKA is [−0.242, 0.488].

**Without the three MAE models.** H2 still FAILS (0/3 for every metric), H3 FAILS, and H5 PASSES 2/3 (fMRI mutual
k-NN drops to 0.263). Dropping MAE moves single cells a lot: EEG RSA H2 goes from 0.215 to 0.348, and fMRI mutual
k-NN H2 from −0.007 to −0.233.

**Subjects with ceiling ≥ 0.1 only.** This excludes MEG sub-01/03/04 for some metrics and fMRI sub-03 for mknn and
CKA. H1 6/9, H2 FAIL, H3 FAIL, H4 FAIL, H5 PASS, H6 PASS. One H2 cell is confirmed here: fMRI CKA, ρ = 0.469
[0.257, 0.594], Holm p 0.025. It is the only positive capability signal that reaches significance anywhere, and only
in this post-hoc subset of 2 fMRI subjects.

**Variants checked by the reviewer that do not rescue H2:**

* leave-one-family-out;
* family means;
* within-family (demeaned);
* per-subject analyses.

**What does produce a positive H2.** Only two things do:

* Adding the 9 random-init networks to the model set (ρ 0.36–0.61 in 7 of 9 cells). H2 is preregistered over
  trained models only; untrained networks sit at near-zero capability and low alignment, so including them
  measures training versus no training, which H1 already tests.
* Picking k = 20 for fMRI mutual k-NN after seeing the results (ρ 0.411).

**Layer-selected H2 (preregistered secondary representation; H2 on it applied post hoc).** The best of 5
depth-fraction layers is selected on one random half of the stimuli and the corrected score is taken on the other
half (34 models; AlexNet was not swept). p = one-sided model-label permutation, Holm over 9 cells; CI = model
bootstrap. Full table: `robustness.md`.

| Dataset | mknn | CKA | RSA |
|---|---|---|---|
| EEG | **0.598** [0.209, 0.863], Holm p 0.002 | −0.022 | −0.228 |
| MEG | 0.155 | 0.384 [−0.063, 0.730], Holm p 0.090 | **0.478** [0.090, 0.782], Holm p 0.022 |
| fMRI | 0.120 | −0.121 | −0.151 |

Two cells are confirmed (bold). Under the H2 threshold (mknn in ≥ 2/3 datasets and CKA or RSA in ≥ 2/3) this
would still FAIL (1/3 and 1/3). The depth-1.0 layer scored on the same held-out half gives EEG mknn 0.087 and MEG
RSA −0.050, so the two positive cells come from layer choice, not from the half-split. Half-split estimates are noisy,
though: on MEG mknn the depth-1.0 layer alone gives 0.325. The signal is real enough to report but mixed. It is
positive where early-to-middle layers win (EEG mknn: best layer at depth 0.4–0.6 for 27/34 models), and absent or
negative on fMRI IT. Best-layer scores also lift the best model in a cell above the primary 0.298 (for example EEG CKA
0.485, CLIP RN50; MEG RSA 0.349). These are maxima over 34 models on 50–100 stimuli, so they are biased upward.
H6 refers to the primary embedding only.

**Capability vs model–model agreement (PRH positive control, exploratory).** On the same stimuli, capability
predicts a model's mean mutual k-NN with the other trained vision models: ρ = 0.580 (EEG stimuli), 0.629 (MEG),
0.624 (fMRI), and 0.467 / 0.532 / 0.525 without MAE (computed from `model_similarity.csv` and
`models_vision.csv`). So the capability proxy does recover PRH's model–model convergence. Brain alignment tracks
that agreement (H5) but not capability itself.

**Restricted capability range.** 24 of 35 models have capability 0.80–0.97. Within them, H2 is 0.15 / 0.10 /
−0.10 (mknn, EEG / MEG / fMRI) and 0.05 / 0.13 / 0.01 (CKA). Most of the spread in capability comes from MAE,
AlexNet and the small ResNets.

**Other exploratory outputs:** audio models on spoken names, time-resolved EEG/MEG curves and the
map of minds (`figures/`). The audio models all rank at the bottom of the leaderboard (PBI 0.033–0.042). No claim in
this document rests on these outputs.

## Deviations from the preregistration

The full list is in `DEVIATIONS.md`. The ones that matter for interpretation:

* **Datasets.** NSD, NOD and BOLD5000 are not included, so all claims are restricted to THINGS (see `BLOCKERS.md`).
* **Language models.** SBERT models were added (exploratory, excluded from H4). The CLIP text towers use their
  pooled projected output rather than a mean-pooled hidden state.
* **MAE.** The pooled timm output is the CLS token, which the MAE objective does not directly supervise (MAE
  features are known to be weak under frozen k-NN or linear readout). This was kept as
  preregistered, and a no-MAE sensitivity analysis is reported.
* **H4 reading (changed after review; makes the result worse).** Both arms are now judged with the general cell
  rule, which turns H4 from PASS into PARTIAL.
* **H3 rule.** The H3 threshold is applied with the stricter general cell rule (Holm p and CI). The CI-only count
  is also 0/3, so this changes nothing.
* **Where deviations are listed.** The preregistration promised a deviations section in this file; the full list
  lives in `DEVIATIONS.md`, and this section summarizes it.

## Independent review

A blind reviewer (an independent AI reviewer agent, not a human peer reviewer) cloned the repository at commit
5627443, installed only `requirements.txt`, and tried to
overturn the headline. The report is committed unedited in `REVIEW.md`, with the response.

* **Reproduction:** `python -m pbi reproduce` ran in 94 s, and every CSV, `hypotheses.md`, `LEADERBOARD.md` and the
  map-of-minds figure was byte-identical to the committed files. The first run had read a stats cache outside the
  repository. With the cache disabled the rerun took 222 s and was again byte-identical; 27 tests passed. The cache
  has since moved into the repository and is keyed on a hash of the engine code.
* **Headline:** not overturned. H2 failed under every defensible variant the reviewer tried. The reviewer did not
  examine the layer sweep. Everything added after commit 5627443 (H4 rule, `robustness.md`, the layer-selected test,
  figures) is regenerated by CI from the committed kernels (which compares `scores.csv`, `hypotheses.csv` and
  `leaderboard.csv`), not by the review.
* **Secondary points partially overturned:**
  * H1's failure is an artefact of how the RDMs are built (column-centred RSA gives 9/9).
  * The negative RSA capability correlations are the same artefact.
  * H4 had been passed on a point estimate; it is now PARTIAL.
  * Corrected mutual k-NN has a chance floor of 0.08–0.18, which weakens H6 and cross-dataset comparisons.
  * The leaderboard ranking is shaped by the last two issues and is now labelled descriptive.

## Limitations

* **Data.** Only three THINGS datasets, with 10, 4 and 3 subjects and 200, 200 and 100 test images. MEG
  reliability is low (split-half ceilings 0.03–0.23; participants 1 and 4 lowest), and MEG RSA rankings are
  unstable across subjects.
* **Brain coverage.** fMRI covers IT cortex only. There is no early visual cortex and no whole-brain analysis.
* **Metric choices.**
  * RSA verdicts depend on the RDM definition.
  * Corrected mutual k-NN is not zero at chance, so corrected values are not comparable across datasets.
  * The square-root correction is a heuristic for mutual k-NN.
* **Models.**
  * Capability is a single proxy (5-shot nearest-centroid on THINGS concepts) measured on the same embedding.
    Other capability measures (ImageNet accuracy, zero-shot transfer) might order models differently.
  * Only the final embedding enters the primary analysis. Layer selection changes two H2 cells (see above).
  * MAE's embedding is the CLS token, which its objective does not directly supervise.
  * Capability is compressed (24 of 35 models at 0.80–0.97).
  * EEG/MEG patterns are flattened over 80–500 ms, which weights early responses. No time-resolved H2 was run.
  * Scale is confounded with training data and recipe across families, and within-family size ranges are small
    (3–5 sizes).
  * No audio-visual models, and no models beyond about 1B vision / 3B language parameters (16 GB RAM).
* **Cross-modal tests.** Language and audio models see only the concept name, not the image. Their scores measure
  concept-level structure, not visual similarity.
* **Statistical power.** With 35 models a Spearman ρ of roughly 0.4 or more is needed for the smallest p to pass Holm across 9 cells. Small
  positive capability effects (the CKA cells sit at ρ ≈ 0.13–0.26) cannot be ruled out; model-resampling CI upper bounds
  are 0.31–0.44 for mknn and 0.49–0.54 for CKA.
