# Deviations from the preregistration

`PREREGISTRATION.md` (commit cbd0a89) said deviations would be listed in `RESULTS.md`. `RESULTS.md` (the reader-facing
write-up) has a "Deviations from the preregistration" section that summarizes the ones that matter for interpretation;
the complete list is kept here.
This file lists every place where the executed analysis differs from, or had to specify, what the preregistration
says. Items marked **(deviation)** change something the preregistration states; items marked **(specification)**
pin down something it leaves open. The final section lists the statistics implementation choices recorded when
`pbi/stats.py` was written.

## Data

* **(deviation) Datasets.** Only THINGS-EEG2 (10 subjects), THINGS-MEG (4) and THINGS-fMRI (3) are analysed. NOD,
  BOLD5000 and NSD (listed as conditional in the preregistration) were not included; reasons in `BLOCKERS.md`.
  All claims are restricted to the THINGS datasets.
* **(specification) MEG window.** "Resampled to 50 Hz, 80-500 ms" was implemented as FFT resampling
  (`mne.filter.resample`, 200 -> 50 Hz) of the full -100..1300 ms epoch followed by the half-open window
  80 <= t < 500 ms, which gives **21 samples** (80, 100, ..., 480 ms) x 271 MEG sensors (reference channels
  excluded) = 5,691 features. The 500 ms sample is not included. (The `pbi/brains.py` docstring said 22 samples;
  corrected to 21 — the committed kernels always used 21, see `artifacts/brains.csv`.)
* **(specification) EEG window.** 17 channels x 42 samples (100 Hz, 80 <= t < 500 ms) = 714 features.
* **(specification) fMRI voxels.** Released IT mask, with voxels that are NaN in any trial dropped
  (sub-01 4,145, sub-02 3,720, sub-03 3,022 voxels).
* **(specification) Split halves.** EEG2 (80 repetitions for every image): one permutation of the repetition indices
  per split, shared by all images. MEG / fMRI: repetitions permuted independently per image; A = first n_i // 2.
  Seeds `default_rng(1000 + split)`.

## Systems

* **(deviation) SBERT added.** Two sentence-transformers models (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`) were scored;
  they are not in the preregistration. They are treated as exploratory: excluded from H4 (language set = Pythia,
  Qwen2.5, CLIP text towers) but listed on the leaderboard. Their embeddings are computed "manually" with
  `transformers` (attention-masked mean pooling + L2 normalisation, i.e. the sentence-transformers pooling of those
  checkpoints), not with the `sentence-transformers` library.
* **(deviation) CLIP text towers** use open_clip `encode_text` (the projected end-of-text token embedding, the tower's
  pooled output) rather than the "mean-pooled final hidden state" the preregistration specifies for language models.
  Pythia and Qwen2.5 use the attention-masked mean of the final hidden state, as preregistered.
* **(specification) Language input.** The fixed sentence is "A photo of a/an {concept}."
* **(specification) MAE embedding.** For `vit_*_patch*_224.mae` timm's pooled output (num_classes=0) is the CLS token,
  which the MAE objective does not directly supervise (MAE features are known to be weak under frozen readout); MAE capability is therefore very low (0.08-0.36). Kept as preregistered
  ("final pooled embedding as exposed by timm") in all primary analyses; an exploratory sensitivity analysis without
  MAE is reported (below).
* **(specification) Preprocessing.** Each model uses its library's default eval transform (timm `resolve_data_config`,
  open_clip transforms, torchvision weights transforms). Consequences: DINOv2 runs at 518 px; SigLIP-L is the 256 px
  checkpoint (`ViT-L-16-SigLIP-256`); the random-init ConvNeXt-T uses crop 0.95 (timm default without pretrained
  config) while trained ConvNeXt uses crop 0.875. AlexNet's embedding is fc7 (after ReLU; final classifier removed).
* **(specification) Precision.** Vision models with > 300 M parameters run in fp16 on Apple MPS; Pythia >= 410M, all
  Qwen2.5 models, Whisper-medium and HuBERT-L were loaded in fp16 (language models fall back to bf16 if fp16
  overflows).
* **(specification) Parameter counts** are of the module that produced the embedding (vision: image tower;
  CLIP text: text tower; Pythia/Qwen: base model without the LM head).
* **(specification) Layer sweep (secondary).** Run for 43 of the 44 vision models; AlexNet (torchvision) was not
  swept. Layer features are the token/spatial mean of the block output at depth fractions 0.2..1.0, so the 1.0 layer
  is not identical to the primary embedding. No layer sweep for language/audio models.
* **(exploratory, added after the write-up checks) H2 on the layer sweep.** The preregistration named the layer
  sweep as a secondary representation but did not say which hypothesis tests to run on it. H2 on the held-out-half
  best-layer score (model-label permutation, Holm over 9 cells, model-bootstrap CI) is now reported in
  `robustness.md`. It confirms EEG mutual k-NN (ρ 0.598) and MEG RSA (ρ 0.478) and would still fail the H2 threshold.
  It replaces no verdict.

## Statistics

* **(specification) H3 confirmation, H2 reading, H4 operationalization, H6 CI, subsample k and kernels, shuffled
  null:** see "Scoring / statistics implementation choices" below. Note in particular that the H3 table threshold
  ("mean rho > 0 with CI excluding 0 in >= 2/3 datasets") is applied with the general cell rule (Holm-corrected
  permutation p < 0.05 **and** CI excluding 0), which is stricter than the table text; `hypotheses.md` also reports
  the CI-only count.
* **(specification) LOSO** rankings use all non-random models (random-init networks excluded).
* **(addition, exploratory) Sensitivity analyses** (not preregistered; added after the capability values and the
  split-half ceilings were known, labelled exploratory in `hypotheses.md` and `sensitivity.csv`; they do not replace
  any verdict): (a) H2, H3, H5 without the three MAE models; (b) every hypothesis after excluding, per dataset x
  metric, subjects whose split-half ceiling is < 0.1 (MEG and fMRI subjects; EEG2 ceilings are all >= 0.23).
* **Independent review.** `REVIEW.md` holds the blind reviewer's report (unedited) and the response to it.
* **(changed after review; makes the result worse) H4 reading.** The H4 table threshold says "above-null (p < 0.05)
  for fMRI-IT; scale rho > 0". Until the review, arm (b) passed on a bare positive point estimate while H3 was judged
  with the full cell rule. Both H4 arms are now judged with the general cell rule (Holm-corrected p < 0.05 and CI
  excluding 0), consistent with H3 and with the definition of "confirmed" in the preregistration. Result: (a) is
  confirmed, (b) is not (fMRI mutual k-NN rho = 0.107, CI [-0.343, 0.914], Holm p = 1; only two LM families), so H4
  is reported as **PARTIAL (not passed)**. Under the literal point-estimate reading H4 would pass; `hypotheses.md`
  states both. (a) now uses the Holm-corrected p and CI instead of the raw p; this changes nothing (Holm p 0.009).
* **(addition, exploratory) Robustness analyses** after the review (`pbi/robustness.py`, `robustness.md`,
  `robustness.csv`, `scores_variants.csv`; not preregistered, no verdict replaced): column-centred RSA
  (RDM = 1 - cosine of column-centred features), chance-adjusted mutual k-NN ((x - c)/(1 - c) on score and ceiling,
  c = k/(n-1), subjects with ceiling <= c excluded), both together, H2 on raw (uncorrected) scores, model-resampling
  bootstrap CIs for H2, and a like-for-like brain-brain reference (subject split halves vs the other subjects).
  The leaderboard shows an exploratory `PBI alt` (chance-adjusted mutual k-NN + column-centred RSA) next to the
  preregistered PBI, which still defines the ranking.
* **(specification) RDM definition kept.** The preregistration specifies "correlation-distance RDMs" (1 - Pearson r
  between stimulus patterns), which is what the primary analysis uses. This RDM keeps every feature's mean over
  stimuli; the review showed that the RSA-based H1 misses and the negative H2/H5 RSA correlations depend on that.
  The primary analysis is unchanged; `hypotheses.md` flags the dependence and `robustness.md` gives the numbers.


## Scoring / statistics implementation choices (pbi/score.py, pbi/stats.py)

None of these change the preregistered plan; they pin down details the plan leaves open.

* **Subsample k for mutual k-NN.** In the 80 % stimulus subsamples k follows the preregistered rule
  k = round(n/20) for the subsample size (k = 8 for 160 of 200 stimuli, k = 4 for 80 of 100), rather than
  keeping the full-set k.
* **Subsample kernels.** Kernels are restricted to the stimulus subset; cosine and Gram kernels keep the
  feature centring of the full set. Debiased CKA is exactly invariant to this (tested); mutual k-NN on the
  cosine kernel is affected only marginally. RSA ranks are recomputed on the subset (ordinal tie-breaking in
  subsamples and nulls; average ranks for point estimates).
* **Shuffled-stimulus null.** One random stimulus permutation per draw is applied to every model kernel
  (each model's marginal null is exact; shared draws make max-over-models nulls well defined). All 1000
  permutations use all subjects and all split halves.
* **H4 "best-of-family" null** = distribution of the maximum over the family's models of the shuffled
  scores (accounts for picking the best model). H4 is judged on the primary metric (mutual k-NN), since
  the threshold names no metric: (a) best language model (null = max over all language models) confirmed
  on THINGS-fMRI-IT, and (b) mean within-LM-family rho(log10 params, score) confirmed there (both with the
  cell rule; see "H4 reading" above). Every family's own best model also beats its family-max null on fMRI
  mutual k-NN (`H4_best:*` rows), so using the best of all language models changes nothing.
* **H3 confirmation** uses the general cell rule (Holm-corrected one-sided permutation p < 0.05, with
  params permuted within each family, plus a 95 % CI excluding 0). hypotheses.md also reports the
  CI-only reading.
* **H2 "≥ 2/3 datasets for CKA or RSA"** is read as (CKA confirmed in ≥ 2 datasets) OR (RSA confirmed in
  ≥ 2 datasets). hypotheses.md also reports the looser per-dataset reading (CKA or RSA in the same dataset).
* **H6 CI** is the percentile interval of the *maximum* over non-random models in each subsample (the
  conservative choice for "no model reaches the ceiling").
* **Holm** always uses a family of 9 cells. Cells with no data (e.g. a dataset not yet processed) enter
  as p = 1 and count as not confirmed.
* **Stored precision.** `scores*.csv`, `ceilings.csv` and `model_similarity.csv` are written with 10 significant
  digits (6 before the review), so they can be compared more tightly than the 1e-6 storage precision.
* **Engine cache.** Moved from `$PBI_DATA/cache/stats` to `.cache/stats` in the repository (git-ignored) and keyed on
  a hash of the engine source code in addition to kernels and settings; `PBI_CACHE=0` disables it.
* **Runtime.** The full 1000 subsamples / 1000 null permutations / 10,000 label permutations are run (no
  reduction was needed). `python -m pbi reproduce --jobs 5` on all data (17 subjects x 64 models, no engine cache)
  took 251 s on an M2 Pro.
