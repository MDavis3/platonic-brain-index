# Independent review (blind agent, clean clone)

*The review below is committed unedited, as promised in `PREREGISTRATION.md` ("Reporting commitments"). The
author's response is appended at the end under "Response to review".*

---

# External review: platonic-brain-index (clean-clone reproduction and attempt to overturn the headline)

Reviewer clone: `~/pbi_data/cache/review_clone` (commit 5627443). Fresh venv (`/opt/homebrew/bin/python3.13`), only `requirements.txt` installed. My analysis scripts are in `review_clone/review/a1.py` to `a4.py`. The original repository was not modified.

## Summary verdict

**Headline overturned? No for the headline itself; partially for the secondary verdicts.** The headline is a null result: H2 fails ("alignment does not reliably rise with capability", so no convergence claim). I could not make H2 pass under any reasonable variant I tried:
- uncorrected raw scores
- chance-corrected mutual k-NN
- per-subject analyses
- dropping each of the 10 vision families in turn
- family-level or within-family analyses
- a model-resampling bootstrap
- an alternative RSA definition

In every case the capability–alignment Spearman ρ stays small (|ρ| ≤ 0.41 in any one cell) and inconsistent across datasets and metrics. Two kinds of change do produce a strong positive H2:
- adding the random-init networks to the model set (ρ = 0.36 to 0.61), which the preregistration rightly excludes;
- choosing k after seeing the results (fMRI k = 20).

Several secondary conclusions are fragile:
- **The H1 FAIL is a by-product of how the RDM is built.** It comes from RSA on correlation-distance RDMs computed without column-centring. With column-centred RDMs, trained models beat random-init networks in all 3 RSA cells (p ≤ 0.0075), and H1 would pass 9/9.
- **The strongly negative MEG-RSA and fMRI-RSA capability correlations come from the same RDM choice.** These are ρ = −0.68 and −0.26, and they turn positive with column-centred RDMs.
- **H4 is marked PASS on a non-significant point estimate.** Its scale arm has ρ = 0.11, p = 0.42, CI [−0.34, 0.91].
- **The corrected mutual k-NN has a chance floor of 0.08 to 0.18 in corrected units.** This undermines the leaderboard's statement that "1.0 = ceiling" and makes averaging across datasets questionable.

Reproduction is exact (byte-identical), including with the hidden stats cache turned off.

## Reproduction

| Check | Result |
|---|---|
| `python -m pbi reproduce` from the clean clone (venv with only requirements.txt) | Ran in 94 s, no errors. |
| `python -m pbi check --against <committed results>` | `[check] OK` |
| All 15 CSVs in `artifacts/results/`, `hypotheses.md`, `LEADERBOARD.md`, `figures/map_of_minds.png` | **Byte-identical** to the committed files. |
| **Caveat found:** the first run printed `engine results from cache`. The stats engine reads and writes a cache **outside the repo** (`$PBI_DATA/cache/stats`, default `~/pbi_data/cache/stats`, the author's own cache on this machine). So the first "clean-clone" run did not recompute the subsamples, nulls or permutations. | I reran with `PBI_CACHE=0 python -m pbi reproduce --root <copy>`. It took 222 s, and **every CSV, LEADERBOARD.md and map_of_minds.png was again byte-identical.** The claim holds, but only after this extra step. |
| `pytest tests` | 27 passed. |
| Stimulus-order alignment | On the model side, the RDM entries for the 100 images shared by MEG and fMRI are identical across the two datasets' kernels (max diff 2e−7), checked for 5 models. On the brain side, the per-model shuffled-stimulus nulls are beaten by every trained vision model in most cells, which rules out a gross misalignment. A direct MEG-vs-fMRI brain RSA on the shared images was weak (0.044, p = 0.23), but MEG reliability is low, so this check is not informative either way. |
| Capability leakage | 0 of the MEG, fMRI or EEG2 test images are among the capability images (the first 10 training images of each concept). EEG2 test concepts are excluded by design. MEG and fMRI test *concepts* can appear among the 600 capability concepts, but the images differ. Capability is a property of the model, not fitted to brains, so I found no circularity. |

Figure: `map_of_minds.png` regenerated identically. It is a 2-D MDS layout:
- **Left side:** brain subjects (EEG stars in an upper-left arc, MEG diamonds, fMRI crosses), placed next to the random-init networks, MAE and the audio models.
- **Right side:** trained vision families, with CLIP and SigLIP tightly clustered near the consensus marker.
- **Bottom:** language models.
- Several labels overlap (EEG s02/s03/s05, MEG s01/s03, mae_vit_huge).

## Issues found

1. **(Major) The RSA's RDM definition flips the H1 verdict and creates spurious negative H2/H5 RSA effects.**
   - **What the code does:** `kernels_from_features` builds the RDM as 1 − Pearson correlation between stimulus rows, without removing the per-feature mean. The mean pattern across stimuli (for example a voxel's average response) is therefore left in.
   - **Evidence with the shipped RDM:**
     - fMRI RSA: trained − random = −0.077 (random nets win), and the H1 fMRI-RSA cell has p = 0.99.
     - H2 RSA: ρ = −0.68 on MEG and −0.26 on fMRI.
     - H5 RSA: ρ = −0.31 on MEG and −0.20 on fMRI.
   - **Evidence with a column-centred RDM** (1 − cosine of column-centred features, from the shipped `cos` kernels, corrected by the same split-half ceiling):

     | Dataset | Trained − random | Permutation p (10k) | CI (100 × 80 % subsamples) |
     |---|---|---|---|
     | EEG | +0.056 | 0.0075 | [0.038, 0.073] |
     | MEG | +0.089 | 0.0003 | [0.068, 0.111] |
     | fMRI | +0.139 | 0.0001 | [0.117, 0.168] |

     All pass Holm across the 9 cells, so H1 = 9/9. H2-RSA becomes −0.007 (EEG), +0.09 (MEG) and +0.27 (fMRI).
   - **Assessment:** the preregistration does say "correlation-distance RDMs", so the shipped verdict follows the protocol. But "H1 FAIL: trained models are not better than random" and the headline branch "H1/H5 did not both pass" are not robust to this choice.
   - **Fix:** report column-centred RSA as a labelled deviation or sensitivity analysis, and qualify the H1 FAIL as driven by the RDM choice.

2. **(Major) The noise-ceiling correction inflates chance for mutual k-NN and over-weights unreliable subjects.**
   - **Chance level:** for raw mutual k-NN, chance is k/(n−1) ≈ 0.050. The ceilings are themselves near chance: MEG 0.058 to 0.123, fMRI 0.082 to 0.172. Dividing by √ceiling therefore turns chance into **0.084 (EEG), 0.146 (fMRI) and 0.183 (MEG)** in corrected units.
   - **Leaderboard effect:** in the MEG_mknn column, every model lies between 0.19 and 0.26, including Whisper on spoken words at 0.197. Its spread across models is SD 0.020, against 0.07 for the other cells.
   - **Weighting:** subjects with ceiling 0.03 to 0.06 get roughly 4 to 5 times the weight of reliable subjects in each dataset mean.
   - **What does not change:** the H2 ranks barely move (chance-corrected mknn H2: EEG 0.076, MEG −0.008, fMRI −0.074; raw-score H2 is similar). So this does not rescue the headline.
   - **What it does break:** the "1.0 = as aligned as reliability allows" wording, comparisons of PBI values across datasets, and H6's informativeness.
   - **Fix:** subtract chance from both the score and the ceiling, (x − c)/(1 − c), before correcting. Exclude, or down-weight by reliability, subjects whose ceiling does not significantly exceed chance.

3. **(Major) The H4 PASS rests on an untested point estimate, and the rule is applied inconsistently with H3.**
   - **Shipped rule:** arm (b) of H4 passes if the mean within-LM-family ρ is > 0. The value is 0.107 with p = 0.42, Holm p = 1.0, CI [−0.34, 0.91], from only 2 families.
   - **Inconsistency:** H3 is judged with the strict rule (Holm p plus CI), while H4(b) uses the point estimate alone.
   - **Contradiction with the prediction:** the author predicted "scale trend: not confirmed". Under the stricter rule the scale arm is not confirmed, and H4 overall would be FAIL.
   - **Fix:** apply the same confirmation rule to H4(b), or report H4 as "(a) confirmed, (b) not confirmed".

4. **(Minor) The H2 confidence intervals do not reflect uncertainty over models.**
   - **Stimulus-subsample CIs exclude 0** in EEG-CKA [0.069, 0.153], EEG-RSA, MEG-CKA and fMRI-CKA, which suggests reliable positive effects.
   - **Model-resampling bootstrap CIs all include 0:** EEG-CKA [−0.24, 0.48], fMRI-CKA [−0.05, 0.54].
   - **Asymmetry:** EEG-CKA's point estimate (0.132) sits near the top of its CI, a sign of bias from the subsampling scheme.
   - The verdict requires the permutation p as well, so the conclusion is unaffected, but the tables invite misreading.
   - **Fix:** add a model-bootstrap CI for correlation tests, or label the CIs "stimulus-sampling only".

5. **(Minor) The leaderboard is shaped by issues 1 and 2.**
   - The #1 model is `mae_vit_huge` (capability 0.36), which is #13 when the RSA cells are dropped. AlexNet is #3.
   - `random_resnet50_s2` (#24) outranks ResNet-34/50/101/152 and all ConvNeXts.
   - Overall PBI vs capability across trained vision models: ρ = −0.04.
   - The PBI is honestly defined, but because it is an unweighted mean of cells with different chance floors and an RDM artefact, the ranking is not a "brain-likeness" ranking you can rely on.
   - **Fix:** use chance-corrected mknn and column-centred RSA, or z-score each cell before averaging (that variant still puts AlexNet and MAE-H in the top two).

6. **(Minor) The MAE capability values are measurement artefacts and give MAE high leverage.**
   - Capability is read from the untrained CLS token (0.08 to 0.36), yet MAE ranks among the most brain-aligned models.
   - The no-MAE sensitivity analysis still fails H2 (the author reports this, which is good).
   - Dropping MAE moves EEG-RSA H2 from 0.215 to 0.348 and fMRI-mknn from −0.007 to −0.233. The verdict stands, but the sign of individual cells is unstable.

7. **(Minor) Mutual k-NN H2 is sensitive to k on fMRI.**
   - fMRI H2-mknn: ρ = −0.007 at k = 5 (primary), 0.238 at k = 10, 0.411 at k = 20 (uncorrected p = 0.0075, Holm 0.068).
   - The primary k was preregistered, so this is not a reporting problem, but the fMRI mknn null is fragile.

8. **(Minor) Reproducibility and documentation gaps.**
   - The stats cache lives outside the repo and is keyed on a manual `ENGINE_VERSION` rather than a hash of the code, so a code change without a version bump would silently reuse stale results. Fix: put the cache under the repo or `.cache/`, hash `stats.py`, and document `PBI_CACHE=0`.
   - There is no README, even though the preregistration and the review brief point to one.
   - The `Makefile` comment says `reproduce` runs the tests; it does not.
   - The preregistration promises `RESULTS.md`; deviations went to `DEVIATIONS.md` instead (disclosed).
   - `pbi full` references a `pbi.download` module that does not exist (it prints instructions instead).

9. **(Minor) The map-of-minds layout mixes incomparable profiles.**
   - Brain rows are chance-inflated corrected subject scores (issue 2); model rows are raw model–model mknn.
   - The columns are z-scored across this mixed population.
   - As a result, brains sitting next to random-init, MAE and audio models may reflect the difference in profile types rather than shared structure.
   - It is labelled exploratory, but no interpretation of it should be drawn without a check that compares like with like.

## Robustness checks I ran (H2 = Spearman ρ between capability and corrected score, 35 trained vision models)

| Variant | EEG mknn / cka / rsa | MEG mknn / cka / rsa | fMRI mknn / cka / rsa |
|---|---|---|---|
| Primary (reproduced) | 0.076 / 0.132 / 0.215 | −0.036 / 0.203 / −0.677 | −0.007 / 0.264 / −0.259 |
| Raw (no ceiling correction) | 0.079 / 0.132 / 0.215 | −0.031 / 0.198 / −0.645 | 0.057 / 0.345 / −0.224 |
| Chance-corrected mknn | 0.076 | −0.008 | −0.074 |
| Column-centred RSA | rsa −0.007 | rsa 0.091 | rsa 0.265 |
| Leave-one-family-out (min..max) | −0.06..0.18 / 0.02..0.24 / 0.00..0.36 | −0.20..0.04 / 0.09..0.31 / −0.78..−0.59 | −0.23..0.14 / 0.08..0.41 / −0.42..−0.06 |
| Family means (10 points) | 0.07 / −0.03 / 0.13 | 0.03 / 0.03 / −0.83 | −0.04 / 0.38 / −0.52 |
| Within-family (family-demeaned) | −0.19 / −0.21 / 0.11 | 0.02 / −0.02 / 0.07 | 0.22 / 0.09 / 0.21 |
| Model-bootstrap 95 % CI | [−0.28, 0.41] / [−0.24, 0.48] / [−0.20, 0.58] | [−0.38, 0.30] / [−0.13, 0.50] / [−0.85, −0.37] | [−0.36, 0.33] / [−0.05, 0.54] / [−0.59, 0.12] |
| Including 9 random-init nets (not preregistered) | 0.52 / 0.43 / 0.36 | 0.45 / 0.56 / −0.27 | 0.49 / 0.61 / −0.36 |

Per-subject ρ:
- EEG-CKA is positive in 10/10 subjects, but small (0.04 to 0.15).
- MEG-RSA is negative in 3/4 subjects.
- fMRI sub-03 (ceiling 0.05 to 0.08) has the opposite sign to sub-01/02 for mknn and CKA.

Other checks:
- **H1 with column-centred RSA:** the RSA cells reach p = 0.0075 / 0.0003 / 0.0001, so H1 = 9/9.
- **H5 leave-one-family-out (mknn):**

  | Dataset | Primary ρ | Leave-one-family-out range |
  |---|---|---|
  | EEG | 0.59 | 0.49 to 0.68 |
  | MEG | 0.49 | 0.35 to 0.57 |
  | fMRI | 0.41 | 0.26 to 0.65 |

  H5 is robust.
- **H6:** it passes trivially, because the best corrected scores (0.21 to 0.30) contain the chance floor from issue 2 and are far below the brain-to-brain reference (0.29 to 0.86).

**Conclusion of the checks:** no defensible variant gives H2 confirmed in ≥ 2/3 datasets for mknn. The only consistently positive (but non-significant) H2 signal is in CKA (ρ 0.13 to 0.26 in all three datasets), and it becomes significant only on fMRI with low-ceiling subjects excluded (exploratory, already reported by the author).

## Verdict on how the preregistered predictions are reported

The reporting is largely honest and transparent:
- **Preregistration is clean:** `PREREGISTRATION.md` has a single commit (cbd0a89), made before any analysis commits.
- **Misses are reported, against the author's own predictions:**
  - H1 was predicted "confirmed 9/9" and is reported FAIL 6/9.
  - H2 was predicted to fail and does fail.
  - H3 was predicted not confirmed and is not confirmed.
  - H5 and H6 were predicted confirmed and pass.
- **Deviations and post-hoc sensitivity analyses** are listed and labelled exploratory, and they do not replace verdicts. The author also chose the stricter reading for H3.

Two exceptions:
- **H4 is reported PASS** although its scale arm is a non-significant point estimate (ρ = 0.11, p = 0.42). This contradicts the author's own prediction and the stricter rule used elsewhere. It should read "partially confirmed".
- **The H1 FAIL is reported correctly under the protocol,** but it should be flagged as driven by the RDM definition (issue 1), because it changes which headline branch applies.

---

## Response to review

All changes are in the commit that adds this section ("Address independent review"), unless noted. No
preregistered analysis was changed to improve a result; the only verdict change (H4) makes the result worse.
New exploratory outputs: `pbi/robustness.py` -> `artifacts/results/robustness.md`, `robustness.csv`,
`scores_variants.csv`.

1. **RDM definition drives the H1 FAIL and the negative H2/H5 RSA effects. Fixed (reporting); primary kept.**
   I agree with the evidence and reproduced it with the full machinery (1000 subsamples, 10k permutations,
   Holm over 9 cells): with column-centred RDMs H1 is confirmed in 9/9 cells (trained - random RSA EEG .056,
   MEG .089, fMRI .139, Holm p <= .0075), H2-RSA becomes -.007 / .091 / .265 (none confirmed), H5-RSA becomes
   .51 / .58 / .78. The preregistration specifies correlation-distance RDMs, so the preregistered H1 verdict
   (FAIL, 6/9) and the headline branch stay as they are; `hypotheses.md` now opens with a generated
   "How to read these verdicts" section that says H1's misses and the negative RSA correlations depend on the
   RDM definition. Files: `pbi/robustness.py`, `pbi/stats.py` (`_reading_notes`), `DEVIATIONS.md`.
2. **Chance floor of corrected mutual k-NN; weighting of unreliable subjects. Fixed (reporting + exploratory
   variant); primary kept.** Added chance-adjusted mutual k-NN ((x - c)/(1 - c) on score and ceiling,
   c = k/(n-1), subjects with ceiling <= c excluded) as an exploratory variant of every test: H1 6/9, H2 0/3,
   H5 2/3 (fMRI drops to Holm p .054), H6 9/9. The chance floor (EEG .084, MEG .183, fMRI .146) is reported in
   `robustness.md`, `hypotheses.md` and the leaderboard header, which no longer says "1.0 = as aligned as
   reliability allows" without qualification and flags MEG mutual k-NN and H6 there as near-uninformative.
   The weighting concern is covered by H2 on raw uncorrected scores (unweighted mean; no cell confirmed) and by
   the existing ceiling >= 0.1 sensitivity analysis. The preregistered correction is kept as the primary.
3. **H4 PASS rests on a point estimate. Fixed.** Both H4 arms are now judged with the general cell rule, as H3
   is: (a) confirmed (.209, Holm p .009), (b) not confirmed (rho .107, CI [-.343, .914], Holm p 1). H4 is
   reported as **PARTIAL (FAIL)**; the detail states that the literal point-estimate reading would pass.
   Files: `pbi/stats.py` (`verdicts`), `tests/test_stats.py`, `DEVIATIONS.md`.
4. **H2 CIs ignore model sampling. Fixed.** The CI caption in `hypotheses.md` now says "stimulus sampling
   only"; model-resampling bootstrap CIs (2000 draws) are in `robustness.md` and all include 0 except
   MEG-RSA (negative).
5. **Leaderboard shaped by issues 1-2. Partly fixed.** The preregistered PBI still defines the ranking (it is
   the preregistered index), but `LEADERBOARD.md` now says to read it as descriptive, explains the chance floor
   and the RDM effect, and adds an exploratory `PBI alt` column (chance-adjusted mutual k-NN + column-centred
   RSA) with its own rank. Files: `pbi/leaderboard.py`, `LEADERBOARD.md`.
6. **MAE capability artefact. Not changed (already handled).** The MAE embedding is the preregistered "pooled
   embedding as exposed by timm"; the no-MAE sensitivity analysis is reported and H2 still fails without MAE.
7. **k sensitivity on fMRI. Not changed (disclosed).** The primary k was preregistered. `hypotheses.md` now
   states that the primary k is 5 for fMRI and 10 for EEG/MEG, so the fMRI H2_k10/H2_k20 rows are sensitivity
   values, not the primary analysis.
8. **Reproducibility and documentation gaps. Fixed, except RESULTS.md and pbi.download.**
   - Engine cache moved into the repository (`.cache/stats`, git-ignored) and keyed on a hash of the engine
     source code (`stats._code_hash`); `PBI_CACHE=0` is documented in `README.md`. A fresh clone now always
     recomputes.
   - `README.md` added.
   - `Makefile` comment corrected (reproduce does not run the tests; `make test` does).
   - RESULTS.md: not created; deviations stay in `DEVIATIONS.md` (disclosed there) and results in
     `hypotheses.md` / `robustness.md`, linked from `README.md`.
   - `pbi.download`: not written (raw data needs manual, licence-gated downloads); `README.md` and the CLI say so.
9. **Map of minds mixes incomparable profiles. Fixed (labelling).** The figure title now states that brain rows
   are ceiling-corrected brain-model scores and model rows raw model-model scores, so positions are not
   like-for-like. No interpretation of the map is used for any claim.

**Additional fixes from the verifiers.** The brain-brain reference was not like-for-like with model scores; a
split-half version (halves vs group mean, halves vs one other subject) is now in `robustness.md`
(e.g. MEG RSA .369 -> .260 / .150, below the best model .298), and the leaderboard and `hypotheses.md` say the
original reference overstates brain-brain agreement. Score CSVs are now stored with 10 significant digits.
