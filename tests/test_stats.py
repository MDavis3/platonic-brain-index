import numpy as np
import pandas as pd
import pytest

from pbi import stats as st


def test_holm():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    adj = st.holm(p, 4)
    # sorted 0.01, 0.03, 0.04, 1(nan) -> 0.04, 0.09, 0.09 (monotone), nan
    np.testing.assert_allclose(adj[:3], [0.04, 0.09, 0.09])
    assert np.isnan(adj[3])
    # missing cells count against the family size of 9
    assert st.holm(np.array([0.01]), 9)[0] == pytest.approx(0.09)


def test_perm_p_and_spearman_perm():
    assert st.perm_p(1.0, np.zeros(99)) == pytest.approx(0.01)
    assert st.perm_p(-1.0, np.zeros(99)) == pytest.approx(1.0)
    x = np.arange(20.0)
    rho, p = st.spearman_perm(x, x + np.random.default_rng(0).normal(scale=0.1, size=20), 2000, 0)
    assert rho > 0.95 and p < 0.01
    rho, p = st.spearman_perm(x, -x, 2000, 0)
    assert rho == pytest.approx(-1) and p > 0.99


def test_partial_spearman_removes_confound():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    x = z + 0.3 * rng.normal(size=200)
    y = z + 0.3 * rng.normal(size=200)
    assert st.spearman(x, y) > 0.7
    assert abs(st.partial_spearman(x, y, z)) < 0.2


def _H(rows):
    base = dict(effect=0.5, ci_lo=0.1, ci_hi=0.9, p=0.001, n=10, note="")
    H = pd.DataFrame([{**base, **r} for r in rows])
    return st._apply_confirmation(H)


def _cells(h, ok_cells, **kw):
    rows = []
    for d in st.DATASETS:
        for m in st.METRICS:
            good = (d, m) in ok_cells
            rows.append(dict(hypothesis=h, dataset=d, metric=m, p=0.001 if good else 0.5,
                             ci_lo=0.1 if good else -0.1, **kw))
    return rows


ALL = {(d, m) for d in st.DATASETS for m in st.METRICS}


def test_verdict_h1_threshold():
    seven = set(sorted(ALL)[:7])
    six = set(sorted(ALL)[:6])
    assert st.verdicts(_H(_cells("H1", seven)))["H1"]["passed"]
    assert not st.verdicts(_H(_cells("H1", six)))["H1"]["passed"]


def test_verdict_h2_threshold():
    D = st.DATASETS
    ok = {(D[0], "mknn"), (D[1], "mknn"), (D[0], "rsa"), (D[2], "rsa")}
    assert st.verdicts(_H(_cells("H2", ok)))["H2"]["passed"]
    ok = {(D[0], "mknn"), (D[1], "mknn"), (D[0], "rsa"), (D[2], "cka")}  # CKA 1/3, RSA 1/3 -> strict fail
    v = st.verdicts(_H(_cells("H2", ok)))["H2"]
    assert not v["passed"] and "-> pass" in v["detail"]  # lenient per-dataset reading would pass
    ok = {(D[0], "mknn"), (D[0], "rsa"), (D[1], "rsa"), (D[0], "cka"), (D[1], "cka")}
    assert not st.verdicts(_H(_cells("H2", ok)))["H2"]["passed"]  # mknn only 1/3


def test_confirmation_needs_ci_and_holm():
    rows = _cells("H5", ALL)
    rows[0]["ci_lo"] = -0.01  # CI includes 0
    rows[1]["p"] = 0.01      # Holm with 9 cells: 0.01*... > 0.05? 9*0.001=0.009 for others; this one 0.01*1 ok
    H = _H(rows)
    h5 = H[H["hypothesis"] == "H5"]
    assert not h5.iloc[0]["confirmed"]
    rows = _cells("H5", ALL)
    for r in rows:
        r["p"] = 0.02  # Holm: 9 * 0.02 = 0.18 -> nothing confirmed
    assert not _H(rows)["confirmed"].any()


def test_verdict_h6_and_h4():
    rows = [dict(hypothesis="H6", dataset=d, metric=m, ci_hi=0.8) for d in st.DATASETS for m in st.METRICS]
    assert st.verdicts(_H(rows))["H6"]["passed"]
    rows[3]["ci_hi"] = 1.05
    assert not st.verdicts(_H(rows))["H6"]["passed"]
    rows = [dict(hypothesis="H4_bestLM", dataset="things_fmri", metric="mknn", p=0.001),
            dict(hypothesis="H4_scale", dataset="things_fmri", metric="mknn", effect=0.2)]
    assert st.verdicts(_H(rows))["H4"]["passed"]
    rows[1]["effect"] = -0.2
    assert not st.verdicts(_H(rows))["H4"]["passed"]
    # (b) is judged with the cell rule: a positive but non-significant point estimate does not pass
    rows[1].update(effect=0.1, p=0.42, ci_lo=-0.3)
    v = st.verdicts(_H(rows))["H4"]
    assert not v["passed"] and v["label"].startswith("PARTIAL") and "would PASS" in v["detail"]


def test_chance_corrected_mknn():
    from pbi import robustness as rb
    c = 0.05
    out = rb.chance_corrected(np.array([[c, 0.3, 0.3]]), np.array([0.5, 0.5, 0.04]), c)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 1] == pytest.approx(((0.3 - c) / (1 - c)) / np.sqrt((0.5 - c) / (1 - c)))
    assert np.isnan(out[0, 2])  # ceiling below chance -> excluded


def test_derive_subject_and_model_restriction():
    rng = np.random.default_rng(0)
    r = dict(sub_s=rng.random((4, 3, 5, 3)), bb_s=rng.random((4, 3, 3)), null_s=rng.random((6, 3, 5, 3)),
             mm_mat=rng.random((4, 4, 4)))
    for M in r["mm_mat"]:
        np.fill_diagonal(M, np.nan)
    st._derive(r)
    np.testing.assert_allclose(r["sub"], r["sub_s"].mean(-1))
    np.testing.assert_allclose(r["mm"], np.nanmean(r["mm_mat"], 2))
    keep = {1: np.array([True, False, True])}
    st._derive(r, keep_subjects=keep, keep_tv=np.array([True, True, False, True]))
    np.testing.assert_allclose(r["sub"][:, 0], r["sub_s"][:, 0].mean(-1))
    np.testing.assert_allclose(r["sub"][:, 1], r["sub_s"][:, 1][..., [0, 2]].mean(-1))
    np.testing.assert_allclose(r["null"][:, 1], r["null_s"][:, 1][..., [0, 2]].mean(-1))
    np.testing.assert_allclose(r["bb"][:, 1], r["bb_s"][:, 1][:, [0, 2]].mean(-1))
    sel = [0, 1, 3]
    np.testing.assert_allclose(r["mm"], np.nanmean(r["mm_mat"][:, sel][:, :, sel], 2))
