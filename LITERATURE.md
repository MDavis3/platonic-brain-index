# Literature check (2026-11-02)

Built from abstracts and landing pages, not full texts; effect sizes below are not independently verified.
Items marked [uncertain] need a check before citing.

## Anchors
- **Huh, Cheung, Wang, Isola (2024)**, *The Platonic Representation Hypothesis*, ICML. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987). Mutual k-NN alignment; stronger vision and language models align more with each other. Brain data only mentioned in passing.
- **Schrimpf et al. (2018/2020)**, Brain-Score, *Neuron* [10.1016/j.neuron.2020.07.040](https://doi.org/10.1016/j.neuron.2020.07.040). Encoding-based benchmark, mostly macaque.
- **Conwell, Prince, Kay, Alvarez, Konkle (2024)**, *Nat. Commun.* 15:9383 [10.1038/s41467-024-53147-y](https://doi.org/10.1038/s41467-024-53147-y). 224 models on NSD: architecture/objective matter little once controlled; training diet matters most; many models tie.

## Direct tests of PRH with brains
- **Raugel et al. (2025/26)**, ICLR 2026, [arXiv:2508.18226](https://arxiv.org/abs/2508.18226). Controlled DINOv3 variants vs fMRI and MEG: size, training amount and human-centric data each raise brain similarity.
- **Yu et al. (2025)**, [arXiv:2506.12117](https://arxiv.org/abs/2506.12117), scale-invariance of embeddings tracks NSD alignment (ConvNeXt variants).
- **Yu et al. (2026)**, [arXiv:2601.22722](https://arxiv.org/abs/2601.22722). Generalization, model–model and model–brain alignment co-vary; lower local intrinsic dimension predicts all three.
- **López-Cardona et al. (2025)**, [arXiv:2510.17833](https://arxiv.org/abs/2510.17833), review of language–fMRI studies; qualitative support for PRH.
- **Marcos-Manchón et al. (2026)**, [arXiv:2605.20496](https://arxiv.org/abs/2605.20496), brain-to-brain near-isometry in NSD.
- **Zhang et al. (2026)**, [arXiv:2609.10947](https://arxiv.org/abs/2609.10947), omni-modal models more brain-like with more input channels.

## Scaling of brain alignment
- **Gokce & Schrimpf (2025)**, ICML, [arXiv:2411.05712](https://arxiv.org/abs/2411.05712). 600+ controlled models: behavioural alignment keeps improving with scale, **neural alignment saturates**.
- **Linsley et al. (2023)**, NeurIPS, [arXiv:2306.03779](https://arxiv.org/abs/2306.03779). Higher ImageNet accuracy → **worse** IT prediction.
- **Antonello, Vaidya, Huth (2023)**, NeurIPS, [arXiv:2305.11863](https://arxiv.org/abs/2305.11863). Language-fMRI encoding improves log-linearly with LM size.
- **Chen & Bonner (2025)**, *Sci. Adv.* [10.1126/sciadv.adw7697](https://doi.org/10.1126/sciadv.adw7697). A few "universal" dimensions shared across networks carry most of the brain alignment.
- **Wang et al. (2023)**, *Nat. Mach. Intell.* [10.1038/s42256-023-00753-y](https://doi.org/10.1038/s42256-023-00753-y). Language supervision + diverse data → better high-level visual cortex models.
- **Muttenthaler et al. (2023)**, ICLR, [arXiv:2211.01201](https://arxiv.org/abs/2211.01201). THINGS odd-one-out: data/objective matter, size barely does.
- **Pepino et al. (2025)**, [arXiv:2511.16849](https://arxiv.org/abs/2511.16849). Audio models: task performance correlates with auditory-cortex alignment.

## Critiques of PRH / mutual k-NN
- **Gröger, Wen, Brbić (2026)**, ICML, [arXiv:2602.14486](https://arxiv.org/abs/2602.14486). Similarity metrics are inflated by width/depth; after permutation calibration, global convergence largely vanishes, local-neighbourhood agreement survives.
- **Koepke et al. (2026)**, [arXiv:2604.18572](https://arxiv.org/abs/2604.18572). Mutual k-NN depends on N and k; gains saturate.
- **You et al. (2026)**, [arXiv:2609.27252](https://arxiv.org/abs/2609.27252). Relational structure converges after calibration; metric geometry much less.

## Datasets / tools
- THINGS-data (Hebart et al. 2023, eLife [10.7554/eLife.82580](https://doi.org/10.7554/eLife.82580)); THINGS-EEG2 (Gifford et al. 2022, NeuroImage). Mostly used for decoding with a few CLIP encoders; no systematic many-model scaling study found on them.
- Net2Brain (600+ models, RSA/encoding, NSD/BOLDMoments; toolbox, no leaderboard, no mutual k-NN). Brain-Score (macaque, encoding). THINGSvision (feature extraction). Algonauts 2025 (movie fMRI encoding).

## Gap this project fills
No open benchmark scores a broad zoo (supervised, SSL, CLIP-style, language-only, audio, random-init) on THINGS EEG, MEG and fMRI with PRH's own metric next to CKA and RSA, noise-ceiling corrected, with permutation nulls, width controls and held-out-subject checks, and asks whether alignment rises with capability. Prior results disagree (rises: Raugel, Yu, Antonello; saturates: Gokce & Schrimpf, Conwell; falls: Linsley), so a calibrated cross-modality test is informative either way.
