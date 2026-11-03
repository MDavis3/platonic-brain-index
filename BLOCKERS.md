# Blockers and workarounds

Logged as they happen. Format: date — blocker — what I did instead.

* 2026-11-02 — **Repo name not given** (the brief says `Repo: <REPO_NAME>`, a placeholder). Creating a new
  repository on the user's GitHub account is an outward-facing action I won't guess at. — Working in a local git
  repo named `platonic-brain-index`, committing after every milestone; push is the one step left for the user
  (`gh repo create <name> --source . --push`).
* 2026-11-02 — **Colab not used.** Stimulus sets are small (100–200 test images; 6,000 capability images), so all
  feature extraction runs locally on an Apple M2 Pro (MPS) in minutes per model. Driving Colab through a browser
  would add fragility without adding compute that this design needs. Larger models (ViT-bigG, 7B+ LLMs) were
  dropped for memory (16 GB RAM), not compute.
* 2026-11-02 — **NSD** requires signing a data-use agreement; I can't accept terms on the user's behalf. — Excluded.
* 2026-11-02 — **NOD (177 GB) and BOLD5000 (83 GB raw / 122 GB release 2.0)** ship as raw or whole-brain data that
  need a GLM and ROI pipeline; THINGS-MEG/fMRI derivatives on figshare download at ~0.8–5 MB/s here. — Deferred;
  claims restricted to THINGS-EEG2 / THINGS-MEG / THINGS-fMRI. Listed as next step.
* 2026-11-02 — **Audio-vision models.** Common open audio-visual models (ImageBind, Wav2CLIP, AudioCLIP) reuse a
  frozen or lightly tuned CLIP image tower, so scoring their image encoder would duplicate CLIP. THINGS has no
  audio stimuli. — Cross-modal test done instead with (a) language-only models given the concept name and
  (b) audio models given the spoken concept name (macOS TTS), labelled exploratory.
* 2026-11-02 — **THINGS images can't be redistributed** (research-only license). — Images are never committed;
  the pipeline downloads them from OSF. The committed artifacts are n×n stimulus kernels, which are derived data.
* 2026-11-02 21:15 — **Push to GitHub (MDavis3).** The user approved pushing to their personal MDavis3 account. The `gh` CLI on this
  machine is only logged into the work account (`manavdavis-sama`), and adding MDavis3 via device login stopped at GitHub's
  passkey re-authentication ("sudo mode"), which only the user can complete. I did not push to the work account. — Everything is
  committed locally; `scripts/push_to_github.sh` creates `MDavis3/platonic-brain-index` and pushes in one step after
  `gh auth login` as MDavis3.
