import numpy as np
import pytest

from pbi import kernels as kn
from pbi import score as sc


def _K(seed, n=60, d=12):
    return kn.kernels_from_features(np.random.default_rng(seed).normal(size=(n, d)))


@pytest.mark.parametrize("metric", kn.METRICS)
def test_identity_is_one(metric):
    K = _K(0)
    assert kn.score(K, K, metric) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("metric", kn.METRICS)
def test_permutation_invariance(metric):
    A, B = _K(1), _K(2)
    X = np.random.default_rng(3).normal(size=(60, 12))
    B = kn.kernels_from_features(X + 0.5 * np.random.default_rng(9).normal(size=X.shape))
    A = kn.kernels_from_features(X)
    p = np.random.default_rng(4).permutation(60)
    assert kn.score(kn.subset(A, p), kn.subset(B, p), metric) == pytest.approx(kn.score(A, B, metric), abs=1e-9)


@pytest.mark.parametrize("metric", kn.METRICS)
def test_shuffled_is_near_chance(metric):
    X = np.random.default_rng(5).normal(size=(80, 10))
    A = kn.kernels_from_features(X)
    B = kn.kernels_from_features(X + 0.3 * np.random.default_rng(6).normal(size=X.shape))
    rng = np.random.default_rng(7)
    vals = [kn.score(kn.subset(A, rng.permutation(80)), B, metric) for _ in range(200)]
    chance = kn.default_k(80) / 79 if metric == "mknn" else 0.0
    assert abs(np.mean(vals) - chance) < 0.03
    assert kn.score(A, B, metric) > np.mean(vals) + 0.3


@pytest.mark.parametrize("metric", kn.METRICS)
def test_feats_match_kernels(metric):
    Ks = [_K(s) for s in range(5)]
    O = sc.rank_orders(Ks)
    rng = np.random.default_rng(0)
    for idx in (None, np.sort(rng.choice(60, 48, replace=False)), rng.permutation(60)):
        F = sc.Feats(Ks, idx, orders=O)
        Fe = sc.Feats(Ks, idx, exact=True)
        sub = (lambda K: K) if idx is None else (lambda K: kn.subset(K, idx))
        E = np.array([[kn.score(sub(Ks[a]), sub(Ks[b]), metric) for b in (3, 4)] for a in (0, 1, 2)])
        np.testing.assert_allclose(F.pair([0, 1, 2], F, [3, 4], metric), E, atol=1e-12)
        np.testing.assert_allclose(Fe.pair([0, 1, 2], Fe, [3, 4], metric), E, atol=1e-12)
        np.testing.assert_allclose(F.rowpair([0, 1], [3, 4], metric), E[[0, 1], [0, 1]], atol=1e-12)


def test_cka_invariant_to_feature_offset():
    """Subsampling kernels of features centred on the full set is exact for debiased CKA."""
    X = np.random.default_rng(1).normal(size=(50, 8))
    Y = X @ np.random.default_rng(2).normal(size=(8, 8)) + np.random.default_rng(3).normal(size=(50, 8))
    idx = np.arange(0, 50, 2)
    a = kn.score(kn.subset(kn.kernels_from_features(X), idx), kn.subset(kn.kernels_from_features(Y), idx), "cka")
    b = kn.score(kn.kernels_from_features(X[idx]), kn.kernels_from_features(Y[idx]), "cka")
    assert a == pytest.approx(b, abs=1e-9)


def test_ceiling_correction():
    raw = np.array([[0.2, 0.3, 0.1], [0.4, 0.6, 0.2]])
    ceil = np.array([0.25, 0.36, -0.1])
    c = sc.corrected(raw, ceil)
    np.testing.assert_allclose(c[:, :2], raw[:, :2] / np.sqrt(ceil[:2]))
    assert np.isnan(c[:, 2]).all()  # ceiling <= 0 -> subject excluded
    s, n = sc.dataset_scores(raw, ceil)
    np.testing.assert_allclose(s, c[:, :2].mean(1))
    assert list(n) == [2, 2]


def test_ceiling_correction_recovers_true_model():
    """A noise-free model of the latent signal should score ~1 after correction (RSA/CKA), below 1 before."""
    rng = np.random.default_rng(0)
    n, reps = 100, 40
    Z = rng.normal(size=(n, 30))
    T = Z[:, None, :] + rng.normal(scale=3.0, size=(n, reps, 30))
    Km = kn.kernels_from_features(Z)
    for metric in ("cka", "rsa"):
        raws, ceils = [], []
        for s in range(5):
            p = np.random.default_rng(1000 + s).permutation(reps)
            A, B = (kn.kernels_from_features(T[:, p[:reps // 2]].mean(1)),
                    kn.kernels_from_features(T[:, p[reps // 2:]].mean(1)))
            raws += [kn.score(Km, A, metric), kn.score(Km, B, metric)]
            ceils.append(kn.score(A, B, metric))
        raw, ceil = np.mean(raws), np.mean(ceils)
        assert raw < 0.95
        assert raw / np.sqrt(ceil) == pytest.approx(1.0, abs=0.12)
