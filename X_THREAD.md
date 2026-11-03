# X thread (draft)

Five posts, each under 280 characters (counts in the HTML comments). Numbers are from `RESULTS.md`.
Attach `figures/headline.png` to post 1 and `figures/map_of_minds.png` to post 3.
The repository linked in post 5 is currently **private**; make it public before posting.

## 1/5

Do AI models become more brain-like as they get more capable? I scored 35 trained vision models against human EEG, MEG and fMRI. Answer: no. Capability vs. noise-ceiling-corrected alignment: Spearman rho from -0.68 to 0.26 across 9 dataset x metric cells, 0/9 significant.

[attach: figures/headline.png]

<!-- 272 characters -->

## 2/5

Scale within a family doesn't help either. Mean within-family rho(params, alignment) on mutual k-NN: EEG -0.37, MEG -0.28, fMRI -0.07. The best model in any cell reaches 0.30 of the noise ceiling. Brain-to-brain reference on EEG and fMRI: 0.40 to 0.86.

<!-- 252 characters -->

## 3/5

What did hold: trained nets beat random-init nets on mutual k-NN and CKA in all 3 datasets, and models that agree more with other models agree more with brains (rho 0.59 EEG, 0.49 MEG, 0.41 fMRI). Language models given only the concept name beat chance on fMRI IT.

[attach: figures/map_of_minds.png]

<!-- 264 characters -->

## 4/5

All 6 hypotheses were preregistered; 5/6 of my predictions held (I wrongly expected trained > random on RSA too). A blind reviewer reproduced it byte-for-byte and couldn't flip the null. Limits: THINGS only, 4 MEG and 3 fMRI subjects, IT only, one capability proxy.

<!-- 265 characters -->

## 5/5

Code, kernels, leaderboard and review: github.com/MDavis3/platonic-brain-index (one command, ~3 min on a CPU). Next: NSD, NOD and BOLD5000, encoding-model metrics, layer-selected scores, more subjects, and model families that vary only in size.

<!-- 244 characters -->
