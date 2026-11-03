
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
  the threshold names no metric: (a) best language model (null = max over all language models) p < 0.05
  on THINGS-fMRI-IT, and (b) mean within-LM-family rho(log10 params, score) > 0 there.
* **H3 confirmation** uses the general cell rule (Holm-corrected one-sided permutation p < 0.05, with
  params permuted within each family, plus a 95 % CI excluding 0). hypotheses.md also reports the
  CI-only reading.
* **H2 "≥ 2/3 datasets for CKA or RSA"** is read as (CKA confirmed in ≥ 2 datasets) OR (RSA confirmed in
  ≥ 2 datasets). hypotheses.md also reports the looser per-dataset reading (CKA or RSA in the same dataset).
* **H6 CI** is the percentile interval of the *maximum* over non-random models in each subsample (the
  conservative choice for "no model reaches the ceiling").
* **Holm** always uses a family of 9 cells. Cells with no data (e.g. a dataset not yet processed) enter
  as p = 1 and count as not confirmed.
* **Runtime.** The full 1000 subsamples / 1000 null permutations / 10,000 label permutations are run (no
  reduction was needed).
