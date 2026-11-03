# X thread (draft)

Five posts, each 280 characters or fewer (counts in the HTML comments; the URL is counted at full length). Numbers are
from `RESULTS.md` and `artifacts/results/robustness.md`. Attach `figures/headline.png` to post 1 and
`figures/scaling.png` (all three metrics) to post 2. `figures/map_of_minds.png` is deliberately not attached:
brain and model positions in it are not like-for-like.
The repository linked in post 5 is currently **private**; make it public before posting.

## 1/5

Do AI vision models get more brain-like as they get more capable? I scored 35 trained vision models against human EEG, MEG and fMRI (THINGS). Final embedding: no detectable trend. Capability vs ceiling-corrected mutual k-NN: rho 0.08 / -0.04 / -0.01; 0/9 cells significant.

[attach: figures/headline.png]

<!-- 273 characters -->

## 2/5

Caveats: with 35 models I can't rule out rho below ~0.4, and CKA is weakly positive in all 3 datasets (0.13-0.26). Using each model's best layer (picked on held-out stimuli, post hoc), capability does predict EEG mutual k-NN (rho 0.60) and MEG RSA (0.48). Not fMRI.

[attach: figures/scaling.png]

<!-- 265 characters -->

## 3/5

Scale within a family doesn't help either. Mean within-family rho(params, mutual k-NN): EEG -0.37, MEG -0.28, fMRI -0.07. Best model in any cell: 0.30 (ceiling-corrected). Other brains score higher on EEG and fMRI (0.33 to 0.79); MEG is too noisy to tell.

<!-- 255 characters -->

## 4/5

What held: trained nets beat random-init on mutual k-NN and CKA, and models that agree more with other models agree more with brains (rho 0.59/0.49/0.41). Capability predicts model-model agreement (rho ~0.6) but not brain alignment. Fits Linsley '23, Gokce & Schrimpf '25.

<!-- 272 characters -->

## 5/5

Preregistered; 5/6 predictions held (H1 wrong; on H2 I expected ~half the cells to pass, got 0/9). An AI reviewer agent re-ran a pre-review commit from a clean clone (byte-identical). THINGS only, 4 MEG / 3 fMRI subjects. Code, one command: github.com/MDavis3/platonic-brain-index

<!-- 280 characters -->
