"""Capability of vision models (preregistered).

5-shot nearest-centroid concept classification on THINGS-EEG2 *training* images:
  * concepts = sorted training-image concept folders, minus any concept that is also an EEG2 test concept
  * 600 concepts chosen with np.random.default_rng(0).choice(concepts, 600, replace=False)
  * per concept, the first 10 images in sorted filename order: first 5 = support, last 5 = query
  * features = the model's primary embedding (same as for kernels, pbi.models)
  * cosine similarity: embeddings L2-normalized, class centroid = mean of normalized support embeddings
    (renormalized), each query assigned to the most similar of the 600 centroids
  * capability = mean accuracy over the 3000 queries

CLI: python -m pbi.capability [--models all] [--force]
"""
import argparse
import time
import traceback

import numpy as np

from pbi import models as M

TRAIN_DIR = M.DATA / "things_eeg2" / "training_images"
N_CONCEPTS, N_IMG, N_SUPPORT = 600, 10, 5


def _concept(folder):
    return folder.split("_", 1)[1] if "_" in folder and folder.split("_", 1)[0].isdigit() else folder


def episode():
    """Return (paths, labels, is_support) for the fixed preregistered episode."""
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(TRAIN_DIR)
    test_dir = M.DATA / "things_eeg2" / "test_images"
    test_concepts = {_concept(d.name) for d in test_dir.iterdir() if d.is_dir()} if test_dir.exists() else set()
    dirs = sorted(d for d in TRAIN_DIR.iterdir() if d.is_dir() and _concept(d.name) not in test_concepts)
    rng = np.random.default_rng(0)
    chosen = rng.choice(np.array([d.name for d in dirs]), N_CONCEPTS, replace=False)
    paths, labels, sup = [], [], []
    for ci, name in enumerate(chosen):
        imgs = sorted(p for p in (TRAIN_DIR / name).iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:N_IMG]
        if len(imgs) < N_IMG:
            raise RuntimeError(f"{name}: only {len(imgs)} images (download incomplete?)")
        paths += imgs
        labels += [ci] * N_IMG
        sup += [True] * N_SUPPORT + [False] * (N_IMG - N_SUPPORT)
    return paths, np.array(labels), np.array(sup)


def five_shot_accuracy(X, labels, sup):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    C = np.stack([X[(labels == c) & sup].mean(0) for c in range(N_CONCEPTS)])
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    q = ~sup
    pred = (X[q] @ C.T).argmax(1)
    return float((pred == labels[q]).mean())


def run(model_ids, force=False):
    paths, labels, sup = episode()
    meta = M.load_meta()
    for mid in sorted(model_ids, key=lambda m: M.REGISTRY[m].approx_params):
        if mid not in meta:
            print(f"[{mid}] no metadata yet (run pbi.models first); skipping")
            continue
        if meta[mid].get("capability") not in ("", None) and not force:
            continue
        spec = M.REGISTRY[mid]
        L = None
        t0 = time.time()
        try:
            def getL():
                nonlocal L
                if L is None:
                    L = M.load(spec)
                return L
            X = M.cached_features(spec, getL, paths, "capability_eeg2train")
            assert np.isfinite(X).all()
            acc = five_shot_accuracy(X, labels, sup)
            meta = M.load_meta()
            meta[mid]["capability"] = round(acc, 5)
            M.save_meta(meta)
            print(f"[{mid}] capability={acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:  # noqa
            traceback.print_exc()
            M.log_blocker(f"capability for `{mid}` — {type(e).__name__}: {str(e)[:200]} — left blank.")
        finally:
            M.free(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    ms = list(M.REGISTRY) if a.models == "all" else a.models.split(",")
    run(ms, a.force)


if __name__ == "__main__":
    main()
