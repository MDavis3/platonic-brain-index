"""End-to-end `python -m pbi reproduce` on synthetic artifacts (small subsample / permutation counts)."""
import shutil

import pandas as pd
import pytest

from pbi import __main__ as cli
from pbi import score as sc
from pbi import stats as st

FAST = dict(n_sub=40, n_null=40, n_perm=500)


@pytest.fixture(scope="module")
def reproduced(synth_root, monkeypatch_module):
    monkeypatch_module.setenv("PBI_CACHE", "0")
    monkeypatch_module.setenv("PBI_DATA", str(synth_root.parent / "nodata"))
    cli.reproduce(str(synth_root), jobs=1, **FAST)
    return synth_root


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_outputs_exist(reproduced):
    r = reproduced / "results"
    for f in ("scores.csv", "scores_subject.csv", "scores_reference.csv", "hypotheses.csv", "hypotheses.md",
              "null_pvalues.csv", "loso.csv", "leaderboard.csv", "model_ci.csv", "family_scaling.csv"):
        assert (r / f).exists(), f
    assert (reproduced / "models.csv").exists()
    assert (reproduced.parent / "LEADERBOARD.md").exists()
    for f in ("map_of_minds.png", "map_of_minds.svg", "scaling.png", "family_scaling.png"):
        assert (reproduced.parent / "figures" / f).exists(), f


def test_synthetic_effects_detected(reproduced):
    S = pd.read_csv(reproduced / "results" / "scores.csv")
    assert set(S["metric"]) == set(sc.METRICS)
    H = pd.read_csv(reproduced / "results" / "hypotheses.csv")
    h1 = H[H["hypothesis"] == "H1"]
    assert len(h1) == 9 and (h1["effect"] > 0).all()
    h2 = H[(H["hypothesis"] == "H2") & (H["metric"] == "rsa")]
    assert (h2["effect"] > 0).all() and h2["effect"].mean() > 0.4
    nul = pd.read_csv(reproduced / "results" / "null_pvalues.csv")
    rnd = nul[nul["model"].str.startswith("random")]
    assert rnd["p"].median() > 0.05  # random-init models are at chance
    ref = H[H["hypothesis"] == "REF_brain_heldout"]
    assert len(ref) == 9
    md = (reproduced / "results" / "hypotheses.md").read_text()
    assert "non-preregistered settings" in md and "| H1" in md


def test_ceilings_match_contract(reproduced):
    C = pd.read_csv(reproduced / "results" / "ceilings.csv")
    D = sc.Dataset("things_eeg2", reproduced)
    from pbi import kernels as kn
    import numpy as np
    s0 = np.mean([kn.score(D.KH[0][2 * s], D.KH[0][2 * s + 1], "rsa") for s in range(5)])
    got = C[(C["dataset"] == "things_eeg2") & (C["subject"] == "sub-01") & (C["metric"] == "rsa")]["ceiling"].iloc[0]
    assert got == pytest.approx(s0, abs=1e-5)


def test_deterministic_and_job_independent(reproduced, tmp_path, monkeypatch_module):
    first = pd.read_csv(reproduced / "results" / "hypotheses.csv")
    other = tmp_path / "artifacts"
    shutil.copytree(reproduced, other, ignore=shutil.ignore_patterns("results"))
    sc.score_all(str(other), verbose=False)
    st.run(str(other), jobs=2, verbose=False, **FAST)
    second = pd.read_csv(other / "results" / "hypotheses.csv")
    pd.testing.assert_frame_equal(first, second)
    from pbi import leaderboard
    leaderboard.build(str(other))
    assert cli.check(reproduced / "results", other)
