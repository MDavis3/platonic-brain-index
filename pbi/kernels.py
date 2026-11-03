"""Stimulus kernels and the three alignment metrics.

Every system (brain subject, model) is summarized on a dataset by three n x n matrices:
  cos  - cosine similarity of column-centered features   (mutual k-NN)
  gram - linear Gram matrix of column-centered features  (debiased CKA)
  rdm  - correlation-distance RDM                        (RSA)
All metrics below take these kernels, so scoring, nulls and subsampling never need raw features.
"""
import numpy as np
from scipy.stats import rankdata

KINDS = ("cos", "gram", "rdm")


def kernels_from_features(X):
    X = np.asarray(X, dtype=np.float64).reshape(len(X), -1)
    Xc = X - X.mean(0, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    gram = Xc @ Xc.T
    gram /= np.trace(gram) + 1e-12  # CKA is scale invariant; normalize for storage
    Xr = X - X.mean(1, keepdims=True)
    Xr /= np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12
    return {"cos": Xn @ Xn.T, "gram": gram, "rdm": 1.0 - Xr @ Xr.T}


def pack(K):
    """Upper triangle (incl. diagonal) as float32 for compact storage."""
    n = len(K["cos"])
    iu = np.triu_indices(n)
    return {k: K[k][iu].astype(np.float32) for k in KINDS} | {"n": np.array(n)}


def unpack(d):
    n = int(d["n"])
    iu = np.triu_indices(n)
    out = {}
    for k in KINDS:
        M = np.zeros((n, n))
        M[iu] = d[k]
        out[k] = M + M.T - np.diag(np.diag(M))
    return out


def subset(K, idx):
    return {k: K[k][np.ix_(idx, idx)] for k in KINDS}


def mean_kernel(Ks):
    return {k: np.mean([K[k] for K in Ks], 0) for k in KINDS}


# ---------------------------------------------------------------- metrics

def _knn(S, k):
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def mutual_knn(A, B, k):
    na, nb = _knn(A, k), _knn(B, k)
    n = len(A)
    ma = np.zeros((n, n), bool)
    mb = np.zeros((n, n), bool)
    r = np.arange(n)[:, None]
    ma[r, na] = True
    mb[r, nb] = True
    return float((ma & mb).sum(1).mean() / k)


def _hsic_u(K, L):
    n = len(K)
    K = K.copy(); L = L.copy()
    np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
    KL = K @ L
    return (np.trace(KL) + K.sum() * L.sum() / ((n - 1) * (n - 2)) - 2 * KL.sum() / (n - 2)) / (n * (n - 3))


def cka_debiased(K, L):
    kl, kk, ll = _hsic_u(K, L), _hsic_u(K, K), _hsic_u(L, L)
    if kk <= 0 or ll <= 0:
        return np.nan
    return float(kl / np.sqrt(kk * ll))


def rsa(D1, D2):
    iu = np.triu_indices(len(D1), 1)
    a, b = rankdata(D1[iu]), rankdata(D2[iu])
    return float(np.corrcoef(a, b)[0, 1])


def default_k(n):
    return max(1, int(round(n / 20)))


def score(KA, KB, metric, k=None):
    if metric == "mknn":
        return mutual_knn(KA["cos"], KB["cos"], k or default_k(len(KA["cos"])))
    if metric == "cka":
        return cka_debiased(KA["gram"], KB["gram"])
    if metric == "rsa":
        return rsa(KA["rdm"], KB["rdm"])
    raise ValueError(metric)


METRICS = ("mknn", "cka", "rsa")
