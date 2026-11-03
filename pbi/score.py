"""Model-brain scores with split-half noise-ceiling correction (PREREGISTRATION.md, "Metrics" / "Noise ceiling").

For dataset d, model m, subject s, metric M:
    raw(m, s)       = mean over 5 splits x 2 halves of M(K_m, K_s,half)
    ceiling(s)      = mean over 5 splits of M(K_s,A, K_s,B)
    corrected(m, s) = raw / sqrt(ceiling)          (subject excluded for M if ceiling <= 0)
    score(m, d)     = mean over included subjects of corrected(m, s)
Brain-brain reference ("held-out subject"):
    corrected_bb(s) = M(K_s,full, mean_{s' != s} K_s',full) / sqrt(ceiling(s))

All three metrics are inner products of per-system feature vectors built from the n x n kernels
(`Feats`), so every model x half score of a dataset is one matrix product. The same code serves the
point estimate (exact ranks), the 80 % stimulus subsamples and the shuffled-stimulus null (both use
an O(N) re-ranking from a precomputed sort order). `tests/test_score.py` checks that Feats reproduces
pbi.kernels.score exactly.

Outputs (artifacts/results/):
  scores_subject.csv   dataset,model,subject,metric,raw,ceiling,corrected,included
  scores.csv           dataset,model,metric,raw,corrected,n_subjects
  scores_reference.csv dataset,system,subject,metric,raw,ceiling,corrected   (brain-brain + consensus)
  ceilings.csv         dataset,subject,metric,ceiling
  model_similarity.csv dataset,model_a,model_b,mknn  (model-model mutual k-NN, incl. 'consensus')
and artifacts/models.csv (merge of artifacts/models_*.csv).
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import kernels as kn

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("things_eeg2", "things_meg", "things_fmri")
DS_SHORT = {"things_eeg2": "EEG", "things_meg": "MEG", "things_fmri": "fMRI"}
METRICS = kn.METRICS
N_SPLITS = 5
MODEL_COLS = ["model_id", "family", "modality", "objective", "training_data", "params", "emb_dim",
              "capability", "size_label", "source"]


def art_root(root=None) -> Path:
    return Path(root or os.environ.get("PBI_ARTIFACTS") or REPO / "artifacts")


def results_dir(root=None) -> Path:
    p = art_root(root) / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ registry

def merge_models(root=None) -> pd.DataFrame:
    """artifacts/models_*.csv -> artifacts/models.csv (later files win on duplicate model_id)."""
    root = art_root(root)
    parts = []
    for f in sorted(glob.glob(str(root / "models_*.csv"))):
        df = pd.read_csv(f)
        if "model_id" in df and len(df):
            parts.append(df)
    if not parts:
        df = pd.read_csv(root / "models.csv") if (root / "models.csv").exists() else pd.DataFrame(columns=MODEL_COLS)
    else:
        df = pd.concat(parts, ignore_index=True).drop_duplicates("model_id", keep="last")
        for c in MODEL_COLS:
            if c not in df:
                df[c] = np.nan
        df = df[MODEL_COLS + [c for c in df.columns if c not in MODEL_COLS]]
        df.to_csv(root / "models.csv", index=False)
    return df.reset_index(drop=True)


def load_models(root=None) -> pd.DataFrame:
    root = art_root(root)
    df = pd.read_csv(root / "models.csv") if (root / "models.csv").exists() else merge_models(root)
    for c in ("params", "emb_dim", "capability"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["objective"] = df["objective"].fillna("").astype(str)
    df["modality"] = df["modality"].fillna("").astype(str)
    df["family"] = df["family"].fillna(df["model_id"]).astype(str)
    return df.set_index("model_id", drop=False)


def is_random(meta, mid):
    return mid in meta.index and meta.at[mid, "objective"] == "random"


def is_trained_vision(meta, mid):
    return mid in meta.index and meta.at[mid, "modality"] == "vision" and meta.at[mid, "objective"] != "random"


# ------------------------------------------------------------------ data

def load_kernel(path):
    with np.load(path) as d:
        return kn.unpack(d)


class Dataset:
    """All kernels of one dataset: models (idx order) and per-subject full + split halves."""

    def __init__(self, ds, root=None, model_ids=None, subject_ids=None):
        root = art_root(root)
        self.ds = ds
        kdir = root / "kernels" / ds
        stim = root / "stimuli" / f"{ds}.csv"
        self.n = len(pd.read_csv(stim)) if stim.exists() else None
        self.subjects, self.KF, self.KH = [], [], []
        for f in sorted(glob.glob(str(kdir / "brain" / "sub-*_full.npz"))):
            sub = re.match(r"(sub-\d+)_full", os.path.basename(f)).group(1)
            if subject_ids is not None and sub not in subject_ids:
                continue
            halves = [kdir / "brain" / f"{sub}_split{s}_{h}.npz" for s in range(N_SPLITS) for h in "AB"]
            if not all(p.exists() for p in halves):
                continue
            try:
                KF = load_kernel(f)
                KH = [load_kernel(p) for p in halves]  # order: s0A s0B s1A s1B ...
            except Exception as e:
                print(f"[score] {ds}/{sub}: unreadable kernel ({type(e).__name__}); skipped")
                continue
            if self.n is None:
                self.n = len(KF["cos"])
            if len(KF["cos"]) != self.n:
                print(f"[score] {ds}/{sub}: n={len(KF['cos'])} != {self.n}; skipped")
                continue
            self.subjects.append(sub)
            self.KF.append(KF)
            self.KH.append(KH)
        self.models, self.KM = [], []
        files = {os.path.basename(f)[:-4]: f for f in glob.glob(str(kdir / "model" / "*.npz"))}
        for mid in sorted(files):
            f = files[mid]
            if model_ids is not None and mid not in model_ids:
                continue
            try:
                K = load_kernel(f)
            except Exception as e:  # e.g. a file another process is still writing
                print(f"[score] {ds}/model/{mid}: unreadable ({type(e).__name__}); skipped")
                continue
            if self.n is not None and len(K["cos"]) != self.n:
                print(f"[score] {ds}/model/{mid}: n={len(K['cos'])} != {self.n}; skipped")
                continue
            if not all(np.isfinite(K[k]).all() for k in kn.KINDS):
                print(f"[score] {ds}/model/{mid}: non-finite kernel; skipped")
                continue
            self.models.append(mid)
            self.KM.append(K)

    @property
    def ok(self):
        return len(self.subjects) > 0 and len(self.models) > 0

    def bb_means(self):
        """Mean full kernel of all other subjects, per subject (None if only one subject)."""
        if len(self.KF) < 2:
            return []
        return [kn.mean_kernel([K for j, K in enumerate(self.KF) if j != i]) for i in range(len(self.KF))]

    def brain_kernels(self):
        """Brain systems in a fixed layout: halves (nsub*10), then full (nsub), then bb means (nsub or 0)."""
        Ks = [K for H in self.KH for K in H] + list(self.KF) + self.bb_means()
        return Ks

    def banks(self):
        """Stacked kernels + RDM sort orders for brains and models (computed once, cached)."""
        if not hasattr(self, "_banks"):
            bB, bM = stack(self.brain_kernels()), stack(self.KM)
            self._banks = (bB, rank_orders(bB), bM, rank_orders(bM))
        return self._banks


# ------------------------------------------------------------------ features

_IU = {}


def _iu(n):
    if n not in _IU:
        _IU[n] = np.triu_indices(n, 1)
    return _IU[n]


def pair_positions(n0, idx):
    """Positions in the original upper triangle (row-major, k=1) of the subset's upper-triangle pairs."""
    a, b = _iu(len(idx))
    i, j = idx[a], idx[b]
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return lo * n0 - lo * (lo + 1) // 2 + (hi - lo - 1)


def stack(Ks):
    """List of kernel dicts -> dict of (S, n, n) arrays."""
    return {k: np.stack([K[k] for K in Ks]) for k in kn.KINDS}


def rank_orders(Ks):
    """Sort order of each kernel's RDM upper triangle (reused for every subsample / permutation)."""
    B = Ks if isinstance(Ks, dict) else stack(Ks)
    iu = _iu(B["rdm"].shape[1])
    return np.stack([np.argsort(R[iu], kind="stable") for R in B["rdm"]]).astype(np.int32)


class Feats:
    """Per-system vectors from which mknn / CKA / RSA follow as inner products.

    mknn: N = k-NN indicator matrix (flattened);   mknn(a,b) = N_a . N_b / (n k)
    cka : G = Gram with zero diagonal (flattened), row sums r, total t;
          HSIC_u(a,b) = (G_a.G_b + t_a t_b/((n-1)(n-2)) - 2 r_a.r_b/(n-2)) / (n(n-3))
    rsa : V = centered, unit-norm ranks of the RDM upper triangle;  rsa(a,b) = V_a . V_b
    """

    def __init__(self, Ks, idx=None, k=None, exact=False, metrics=METRICS, orders=None):
        B = Ks if isinstance(Ks, dict) else stack(Ks)
        S, n0 = B["cos"].shape[:2]
        idx = np.arange(n0) if idx is None else np.asarray(idx)
        n = len(idx)
        self.n, self.metrics = n, tuple(metrics)
        self.k = int(k or kn.default_k(n))
        full = len(idx) == n0 and np.array_equal(idx, np.arange(n0))
        sub = (lambda A: A) if full else (lambda A: A[:, idx][:, :, idx])
        d = np.arange(n)
        if "mknn" in metrics:
            C = sub(B["cos"]).copy() if full else sub(B["cos"])
            C[:, d, d] = -np.inf  # as pbi.kernels._knn
            nb = np.argpartition(-C, self.k, axis=2)[:, :, :self.k]
            del C
            N = np.zeros((S, n, n), np.float32)  # 0/1 entries: float32 dot products are exact
            np.put_along_axis(N, nb, 1.0, axis=2)
            self.N = N.reshape(S, -1)
        if "cka" in metrics:
            G = sub(B["gram"]).copy() if full else sub(B["gram"])
            G[:, d, d] = 0.0
            self.r = G.sum(2)
            self.t = self.r.sum(1)
            self.G = G.reshape(S, -1)
            self.hself = self._hsic(np.einsum("ij,ij->i", self.G, self.G), self.t * self.t,
                                    np.einsum("ij,ij->i", self.r, self.r))
        if "rsa" in metrics:
            a, b = _iu(n)
            m = len(a)
            if exact or orders is None:
                Rd = sub(B["rdm"])
                V = np.stack([rankdata(Rd[s][a, b]) for s in range(S)])
            else:
                pos = pair_positions(n0, idx)
                N0 = n0 * (n0 - 1) // 2
                mask = np.zeros(N0, bool)
                mask[pos] = True
                srt = orders[mask[orders]].reshape(S, m)
                R = np.empty((S, N0))
                np.put_along_axis(R, srt, np.broadcast_to(np.arange(1, m + 1, dtype=float), (S, m)), axis=1)
                V = R[:, pos]
            V = V - V.mean(1, keepdims=True)
            nrm = np.linalg.norm(V, axis=1, keepdims=True)
            self.V = np.divide(V, nrm, out=np.zeros_like(V), where=nrm > 0)

    def _hsic(self, gg, tt, rr):
        n = self.n
        return (gg + tt / ((n - 1) * (n - 2)) - 2 * rr / (n - 2)) / (n * (n - 3))

    def pair(self, ia, other, ib, metric):
        """Matrix of metric(system ia[i] of self, system ib[j] of other)."""
        ia, ib = np.asarray(ia), np.asarray(ib)
        if metric == "mknn":
            return (self.N[ia] @ other.N[ib].T).astype(np.float64) / (self.n * self.k)
        if metric == "cka":
            h = self._hsic(self.G[ia] @ other.G[ib].T, np.outer(self.t[ia], other.t[ib]),
                           self.r[ia] @ other.r[ib].T)
            den = np.outer(self.hself[ia], other.hself[ib])
            out = np.full(h.shape, np.nan)
            ok = (np.outer(self.hself[ia] > 0, other.hself[ib] > 0))
            out[ok] = h[ok] / np.sqrt(den[ok])
            return out
        if metric == "rsa":
            return self.V[ia] @ other.V[ib].T
        raise ValueError(metric)

    def rowpair(self, ia, ib, metric):
        """metric(system ia[i], system ib[i]) within self, elementwise."""
        ia, ib = np.asarray(ia), np.asarray(ib)
        if metric == "mknn":
            return np.einsum("ij,ij->i", self.N[ia], self.N[ib]).astype(np.float64) / (self.n * self.k)
        if metric == "cka":
            h = self._hsic(np.einsum("ij,ij->i", self.G[ia], self.G[ib]), self.t[ia] * self.t[ib],
                           np.einsum("ij,ij->i", self.r[ia], self.r[ib]))
            den = self.hself[ia] * self.hself[ib]
            out = np.full(h.shape, np.nan)
            ok = (self.hself[ia] > 0) & (self.hself[ib] > 0)
            out[ok] = h[ok] / np.sqrt(den[ok])
            return out
        if metric == "rsa":
            return np.einsum("ij,ij->i", self.V[ia], self.V[ib])
        raise ValueError(metric)


class BrainLayout:
    def __init__(self, nsub, has_bb):
        self.nsub = nsub
        self.halves = np.arange(nsub * 2 * N_SPLITS).reshape(nsub, N_SPLITS, 2)
        self.full = nsub * 2 * N_SPLITS + np.arange(nsub)
        self.bb = (self.full[-1] + 1 + np.arange(nsub)) if has_bb else None


def brain_feats(D: Dataset, idx=None, k=None, exact=False, metrics=METRICS):
    bB, oB, _, _ = D.banks()
    return Feats(bB, idx, k, exact, metrics, oB), BrainLayout(len(D.subjects), len(D.subjects) > 1)


def corrected(raw, ceil):
    """raw (..., nsub), ceil (nsub,) -> corrected with NaN for excluded subjects (ceiling <= 0 or NaN)."""
    ceil = np.asarray(ceil, float)
    ok = np.isfinite(ceil) & (ceil > 0)
    out = np.full(np.shape(raw), np.nan)
    out[..., ok] = np.asarray(raw)[..., ok] / np.sqrt(ceil[ok])
    return out


def nanmean0(x, axis=-1):
    x = np.asarray(x, float)
    cnt = np.isfinite(x).sum(axis)
    s = np.where(np.isfinite(x), x, 0).sum(axis)
    return np.divide(s, cnt, out=np.full(np.shape(s), np.nan), where=cnt > 0)


def block_scores(FM, im, FB, L: BrainLayout, metric):
    """raw (len(im), nsub), ceiling (nsub,), bb raw (nsub,) for one metric."""
    H = L.halves.reshape(-1)
    raw = FM.pair(im, FB, H, metric).reshape(len(im), L.nsub, 2 * N_SPLITS).mean(-1)
    ceil = FB.rowpair(L.halves[:, :, 0].reshape(-1), L.halves[:, :, 1].reshape(-1), metric)
    ceil = ceil.reshape(L.nsub, N_SPLITS).mean(1)
    bb = FB.rowpair(L.full, L.bb, metric) if L.bb is not None else np.full(L.nsub, np.nan)
    return raw, ceil, bb


def dataset_scores(raw, ceil):
    c = corrected(raw, ceil)
    return nanmean0(c, -1), np.isfinite(c).sum(-1)


# ------------------------------------------------------------------ point estimates

def score_all(root=None, datasets=DATASETS, verbose=True):
    root = art_root(root)
    meta = merge_models(root)
    meta = load_models(root)
    out = results_dir(root)
    rows_s, rows_d, rows_r, rows_c, rows_mm = [], [], [], [], []
    for ds in datasets:
        D = Dataset(ds, root)
        if not D.ok:
            if verbose:
                print(f"[score] {ds}: no complete brain+model kernels yet; skipped")
            continue
        tv = [i for i, m in enumerate(D.models) if is_trained_vision(meta, m)]
        extra = {}
        if tv:
            extra["consensus"] = kn.mean_kernel([D.KM[i] for i in tv])
        KM = D.KM + list(extra.values())
        names = D.models + list(extra)
        FM = Feats(KM, exact=True)
        FB, L = brain_feats(D, exact=True)
        im = np.arange(len(KM))
        mm = FM.pair(im, FM, im, "mknn")
        for a, ma in enumerate(names):
            for b, mb in enumerate(names):
                rows_mm.append((ds, ma, mb, mm[a, b]))
        for metric in METRICS:
            raw, ceil, bb = block_scores(FM, im, FB, L, metric)
            cor = corrected(raw, ceil)
            for si, sub in enumerate(D.subjects):
                rows_c.append((ds, sub, metric, ceil[si]))
                for mi, mid in enumerate(names):
                    row = (ds, mid, sub, metric, raw[mi, si], ceil[si], cor[mi, si], bool(np.isfinite(cor[mi, si])))
                    (rows_s if mi < len(D.models) else rows_r).append(row)
                rows_r.append((ds, "brain_heldout", sub, metric, bb[si], ceil[si],
                               corrected(bb[si:si + 1], ceil[si:si + 1])[0], bool(np.isfinite(ceil[si]) and ceil[si] > 0)))
        if verbose:
            print(f"[score] {ds}: {len(D.models)} models x {len(D.subjects)} subjects (n={D.n})")
    cols = ["dataset", "model", "subject", "metric", "raw", "ceiling", "corrected", "included"]
    S = pd.DataFrame(rows_s, columns=cols)
    R = pd.DataFrame(rows_r, columns=cols).rename(columns={"model": "system"})
    S.to_csv(out / "scores_subject.csv", index=False, float_format="%.6g")
    R.to_csv(out / "scores_reference.csv", index=False, float_format="%.6g")
    pd.DataFrame(rows_c, columns=["dataset", "subject", "metric", "ceiling"]).to_csv(
        out / "ceilings.csv", index=False, float_format="%.6g")
    pd.DataFrame(rows_mm, columns=["dataset", "model_a", "model_b", "mknn"]).to_csv(
        out / "model_similarity.csv", index=False, float_format="%.6g")
    agg = summarize(S, "model")
    agg.to_csv(out / "scores.csv", index=False, float_format="%.6g")
    return agg


def summarize(S, key):
    rows = []
    for (ds, mid, metric), g in S.groupby(["dataset", key, "metric"], sort=False):
        inc = g[g["included"].astype(bool)]
        rows.append((ds, mid, metric, inc["raw"].mean() if len(inc) else np.nan,
                     inc["corrected"].mean() if len(inc) else np.nan, len(inc)))
    return pd.DataFrame(rows, columns=["dataset", key, "metric", "raw", "corrected", "n_subjects"])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Score every model against every brain (point estimates).")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    score_all(a.root)


if __name__ == "__main__":
    main()
