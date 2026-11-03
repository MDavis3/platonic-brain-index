"""Platonic Brain Index command line.

  python -m pbi reproduce   committed kernels -> scores -> hypothesis tests -> leaderboard -> figures
                            (CPU only, no downloads; needs only requirements.txt)
  python -m pbi full        raw data -> stimuli -> brain kernels -> model kernels -> reproduce
                            (needs requirements-full.txt, ~40 GB of data under $PBI_DATA, a GPU helps)
  python -m pbi score | stats | leaderboard | figures      single steps
  python -m pbi check --against DIR                        compare regenerated results with a saved copy

Options for reproduce/stats: --jobs N (worker processes; default min(8, cpus-2)), --root DIR (artifacts dir).
--n-sub / --n-null / --n-perm exist only for fast tests; non-default values are flagged in hypotheses.md.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import score as sc

CHECK_FILES = ("scores.csv", "hypotheses.csv", "leaderboard.csv")


def reproduce(root=None, jobs=None, n_sub=None, n_null=None, n_perm=None):
    from . import figures, leaderboard, stats
    t0 = time.time()
    print("== score ==", flush=True)
    sc.score_all(root)
    print("== stats ==", flush=True)
    stats.run(root, n_sub or stats.N_SUB, n_null or stats.N_NULL, n_perm or stats.N_PERM, jobs)
    print("== leaderboard ==", flush=True)
    leaderboard.build(root)
    print("== figures ==", flush=True)
    figures.make_all(root)
    print(f"reproduce finished in {time.time() - t0:.0f} s", flush=True)


def _run_module(mod, args):
    cmd = [sys.executable, "-m", mod] + args
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=sc.REPO)


def full(jobs=None):
    """Rebuild everything from raw data, then reproduce. Each step is another agent's CLI and is rerunnable."""
    try:
        importlib.import_module("pbi.download")
        _run_module("pbi.download", [])
    except ModuleNotFoundError:
        print("No pbi.download module: raw data must already be under $PBI_DATA (default ~/pbi_data):\n"
              "  things_eeg2/dl/sub-XX.zip (Gifford et al. 2022, OSF 3jk45), things_eeg2/test_images/,\n"
              "  things_meg/preprocessed_P{1..4}-epo*.fif (Hebart et al. 2023, figshare / OpenNeuro ds004212),\n"
              "  things_fmri/testset/betas_csv_testset/ (Hebart et al. 2023, figshare).", flush=True)
    _run_module("pbi.stimuli", [])
    _run_module("pbi.brains", ["--dataset", "all"])
    _run_module("pbi.models", ["--dataset", "all", "--models", "all", "--layers"])
    _run_module("pbi.text_audio", ["--dataset", "all"])
    reproduce(jobs=jobs)


def check(against, root=None, tol=2e-3):
    """Numerically compare regenerated result CSVs with a saved copy (exact on non-numeric columns)."""
    new_dir, old_dir = sc.results_dir(root), Path(against)
    bad = []
    for f in CHECK_FILES:
        if not (old_dir / f).exists():
            print(f"[check] {f}: no committed copy; skipped")
            continue
        if not (new_dir / f).exists():
            bad.append(f"{f}: not regenerated")
            continue
        a, b = pd.read_csv(new_dir / f), pd.read_csv(old_dir / f)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            bad.append(f"{f}: shape/columns differ ({a.shape} vs {b.shape})")
            continue
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                x, y = a[c].to_numpy(float), b[c].to_numpy(float)
                diff = ~((np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= tol))
                if diff.any():
                    bad.append(f"{f}:{c} differs in {diff.sum()} rows (max |d|={np.nanmax(np.abs(x - y)):.3g})")
            elif not a[c].astype(str).equals(b[c].astype(str)):
                bad.append(f"{f}:{c} differs")
    for m in bad:
        print("[check] " + m)
    print("[check] OK" if not bad else f"[check] {len(bad)} differences")
    return not bad


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m pbi", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["reproduce", "full", "score", "stats", "leaderboard", "figures", "check"])
    ap.add_argument("--root", default=None, help="artifacts directory (default: repo/artifacts or $PBI_ARTIFACTS)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--n-sub", type=int, default=None)
    ap.add_argument("--n-null", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--against", default=None, help="for check: directory with the reference result CSVs")
    a = ap.parse_args(argv)
    if a.command == "reproduce":
        reproduce(a.root, a.jobs, a.n_sub, a.n_null, a.n_perm)
    elif a.command == "full":
        full(a.jobs)
    elif a.command == "score":
        sc.score_all(a.root)
    elif a.command == "stats":
        from . import stats
        stats.run(a.root, a.n_sub or stats.N_SUB, a.n_null or stats.N_NULL, a.n_perm or stats.N_PERM, a.jobs)
    elif a.command == "leaderboard":
        from . import leaderboard
        leaderboard.build(a.root)
    elif a.command == "figures":
        from . import figures
        figures.make_all(a.root)
    elif a.command == "check":
        if not a.against:
            ap.error("check needs --against DIR")
        sys.exit(0 if check(a.against, a.root) else 1)


if __name__ == "__main__":
    main()
