import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pbi import kernels as kn  # noqa: E402

SYN_DATASETS = {"things_eeg2": (40, 3, 8), "things_meg": (40, 2, 6), "things_fmri": (30, 2, 6)}  # n, subjects, reps


def _save(path, X):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kn.pack(kn.kernels_from_features(X)))


def make_synthetic(root: Path, seed=0):
    """Synthetic artifacts following the file contract. A latent 'world' Z drives brains and models; model
    alignment with Z grows with size/capability, random-init models see none of it."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    models = []
    #            id, family, modality, objective, params, emb, signal
    spec = []
    for fam, obj, sizes in (("resnet", "supervised", [(11e6, 0.6), (25e6, 0.9), (60e6, 1.3)]),
                            ("vit", "supervised", [(5e6, 0.5), (22e6, 0.8), (86e6, 1.2), (300e6, 1.5)]),
                            ("dinov2", "ssl", [(21e6, 0.9), (86e6, 1.2), (300e6, 1.6)]),
                            ("clip", "clip", [(38e6, 1.0), (88e6, 1.4)])):
        for i, (p, s) in enumerate(sizes):
            spec.append((f"{fam}_{i}", fam, "vision", obj, p, 64 * (i + 1), s))
    for i in range(3):
        spec.append((f"random_resnet_s{i}", "random_resnet", "vision", "random", 25e6, 128, 0.0))
    for i, (p, s) in enumerate([(70e6, 0.2), (400e6, 0.3), (1.4e9, 0.35)]):
        spec.append((f"pythia_{i}", "pythia", "language", "lm", p, 512, s))
    spec.append(("whisper_0", "whisper", "audio", "asr", 39e6, 384, 0.1))
    vis_rows, la_rows = [], []
    for ds, (n, nsub, reps) in SYN_DATASETS.items():
        Z = rng.normal(size=(n, 6))
        stim = pd.DataFrame(dict(idx=range(n), image=[f"img{i:03d}.jpg" for i in range(n)],
                                 concept=[f"c{i}" for i in range(n)], path_rel=[f"img{i:03d}.jpg" for i in range(n)]))
        (root / "stimuli").mkdir(parents=True, exist_ok=True)
        stim.to_csv(root / "stimuli" / f"{ds}.csv", index=False)
        for s in range(1, nsub + 1):
            W = rng.normal(size=(6, 20))
            T = (Z @ W)[:, None, :] + rng.normal(scale=4.0, size=(n, reps, 20))
            _save(root / "kernels" / ds / "brain" / f"sub-{s:02d}_full.npz", T.mean(1))
            for sp in range(5):
                perm = np.random.default_rng(1000 + sp).permutation(reps)
                A, B = perm[:reps // 2], perm[reps // 2:]
                _save(root / "kernels" / ds / "brain" / f"sub-{s:02d}_split{sp}_A.npz", T[:, A].mean(1))
                _save(root / "kernels" / ds / "brain" / f"sub-{s:02d}_split{sp}_B.npz", T[:, B].mean(1))
        for mid, fam, mod, obj, p, emb, sig in spec:
            X = sig * (Z @ rng.normal(size=(6, emb))) + rng.normal(size=(n, emb))
            _save(root / "kernels" / ds / "model" / f"{mid}.npz", X)
    for mid, fam, mod, obj, p, emb, sig in spec:
        cap = (0.2 + 0.4 * sig / 1.6 + 0.01 * rng.normal()) if mod == "vision" and obj != "random" else np.nan
        row = dict(model_id=mid, family=fam, modality=mod, objective=obj, training_data="synthetic", params=int(p),
                   emb_dim=emb, capability=cap, size_label="", source="synthetic")
        (vis_rows if mod == "vision" else la_rows).append(row)
    pd.DataFrame(vis_rows).to_csv(root / "models_vision.csv", index=False)
    pd.DataFrame(la_rows).to_csv(root / "models_langaudio.csv", index=False)
    return root


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("pbi") / "artifacts")
