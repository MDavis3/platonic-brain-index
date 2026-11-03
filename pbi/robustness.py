"""EXPLORATORY robustness analyses added in response to the independent review (REVIEW.md) and the verifiers.

None of this is preregistered and none of it replaces a preregistered verdict. Everything reuses the primary
machinery (same split halves, same 1000 80 % stimulus subsamples, 10,000 label permutations, Holm over 9 cells):

  rsa_cc       RSA with column-centred RDMs: RDM = 1 - cosine similarity of column-centred features (the `cos`
               kernel already stored for mutual k-NN). The preregistered RDM (1 - Pearson r between stimulus rows)
               keeps every feature's mean across stimuli, so a shared mean pattern can dominate it.
  mknn_chance  chance-adjusted mutual k-NN: chance c = k/(n-1) is removed from score and ceiling before the
               attenuation correction, corrected = ((raw-c)/(1-c)) / sqrt((ceil-c)/(1-c)); subjects with
               ceil <= c are excluded (the analogue of the preregistered "ceiling <= 0" rule). 0 = chance.
  both         rsa_cc and mknn_chance together.
  H2_raw       H2 on uncorrected raw scores (unweighted mean over subjects; no ceiling division).
  H2_model_bootstrap  95 % CI of the H2 rho from resampling *models* with replacement (the primary CI only
               resamples stimuli).
  REF_*        brain-brain reference computed like-for-like with the model scores (split halves of a subject
               against (i) the mean full kernel of the other subjects, (ii) each other subject's full kernel).

Outputs: artifacts/results/robustness.csv, robustness.md, scores_variants.csv (per-model variant scores).
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

from . import score as sc
from . import stats as st

METRICS = sc.METRICS
IR, IM = METRICS.index("rsa"), METRICS.index("mknn")
N_BOOT = 2000
SEED_BOOT = 60261102
VARIANTS = {"rsa_cc": "RSA on column-centred RDMs (1 - cosine of column-centred features) replaces the RSA cells",
            "mknn_chance": "chance-adjusted mutual k-NN ((x-c)/(1-c) on score and ceiling, c = k/(n-1)) replaces "
                           "the mutual k-NN cells",
            "both": "both replacements"}


def chance_corrected(raw, ceil, c):
    raw, ceil = np.asarray(raw, float), np.asarray(ceil, float)
    ca = (ceil - c) / (1 - c)
    ok = np.isfinite(ca) & (ca > 0)
    out = np.full(np.shape(raw), np.nan)
    out[..., ok] = ((raw[..., ok] - c) / (1 - c)) / np.sqrt(ca[ok])
    return out


def _cc_bank(B):
    return {"cos": B["cos"], "gram": B["gram"], "rdm": 1.0 - B["cos"]}


def _cc(W):
    if "cc" not in W:
        bB, _, bM, _ = W["D"].banks()
        cB, cM = _cc_bank(bB), _cc_bank(bM)
        W["cc"] = (cB, sc.rank_orders(cB), cM, sc.rank_orders(cM))
    return W["cc"]


def _scores(D, W, idx, exact):
    """Per-subject corrected rsa_cc and chance-adjusted mknn (+ brain-brain), for one stimulus subset."""
    cB, oB, cM, oM = _cc(W)
    bB, _, bM, _ = D.banks()
    L = sc.BrainLayout(len(D.subjects), len(D.subjects) > 1)
    im = np.arange(len(D.models))
    FB = sc.Feats(cB, idx, exact=exact, metrics=("rsa",), orders=None if exact else oB)
    FM = sc.Feats(cM, idx, exact=exact, metrics=("rsa",), orders=None if exact else oM)
    r_raw, r_ceil, r_bb = sc.block_scores(FM, im, FB, L, "rsa")
    FB = sc.Feats(bB, idx, exact=exact, metrics=("mknn",))
    FM = sc.Feats(bM, idx, exact=exact, metrics=("mknn",))
    m_raw, m_ceil, m_bb = sc.block_scores(FM, im, FB, L, "mknn")
    c = FM.k / (FM.n - 1)
    return dict(rsa_cc=(sc.corrected(r_raw, r_ceil), sc.corrected(r_bb, r_ceil), r_raw, r_ceil),
                mknn_chance=(chance_corrected(m_raw, m_ceil, c), chance_corrected(m_bb, m_ceil, c), m_raw, m_ceil, c))


def _task(args):
    ds, root, seeds, snap = args
    W = st._get(ds, root, snap)
    D = W["D"]
    ns, nM = len(D.subjects), len(D.models)
    out = {v: (np.full((len(seeds), nM, ns), np.nan), np.full((len(seeds), ns), np.nan)) for v in ("rsa_cc", "mknn_chance")}
    for j, b in enumerate(seeds):
        r = _scores(D, W, st._subsample_idx(D.n, b), exact=False)
        for v in out:
            out[v][0][j], out[v][1][j] = r[v][0], r[v][1]
    return ds, seeds, out


def _point(D):
    W = {"D": D}
    return _scores(D, W, None, exact=True)


def run_engine(root, eng, n_sub, jobs, verbose):
    """{ds: {variant: (sub (n_sub, nM, ns), bb (n_sub, ns))}} (cached next to the primary engine cache)."""
    cdir = st._cache_dir()
    res, tasks = {}, []
    for ds, E in eng.items():
        snap = (tuple(E["models"]), tuple(E["subjects"]))
        fp = st._fingerprint(root, ds, n_sub, 0, snap)
        cpath = cdir / f"{ds}_{fp}_robust.npz" if cdir else None
        if cpath and cpath.exists():
            z = np.load(cpath)
            res[ds] = {v: (z[f"{v}_sub"], z[f"{v}_bb"]) for v in ("rsa_cc", "mknn_chance")}
            if verbose:
                print(f"[robustness] {ds}: subsample results from cache {cpath.name}")
            continue
        nM, ns = len(E["models"]), len(E["subjects"])
        res[ds] = {v: (np.full((n_sub, nM, ns), np.nan), np.full((n_sub, ns), np.nan)) for v in ("rsa_cc", "mknn_chance")}
        res[ds]["_cpath"] = cpath
        tasks += [(ds, root, list(range(i, min(i + 10, n_sub))), snap) for i in range(0, n_sub, 10)]
    if tasks:
        t0 = time.time()
        if verbose:
            print(f"[robustness] {len(tasks)} subsample tasks on {jobs} processes", flush=True)
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(jobs, initializer=st._init_worker) as pool:
                it = pool.imap(_task, tasks, chunksize=1)
                for ds, seeds, out in it:
                    for v, (s, b) in out.items():
                        res[ds][v][0][seeds], res[ds][v][1][seeds] = s, b
        else:
            for ds, seeds, out in map(_task, tasks):
                for v, (s, b) in out.items():
                    res[ds][v][0][seeds], res[ds][v][1][seeds] = s, b
        if verbose:
            print(f"[robustness] subsamples done in {time.time() - t0:.0f} s", flush=True)
    for ds, r in res.items():
        cpath = r.pop("_cpath", None)
        if cpath is not None:
            try:
                np.savez_compressed(cpath, **{f"{v}_{k}": r[v][i] for v in ("rsa_cc", "mknn_chance")
                                              for i, k in enumerate(("sub", "bb"))})
            except OSError:
                pass
    return res


def _variant(eng, point, pts, sub, which):
    """Copies of (point, eng) with the RSA and/or mutual k-NN slots replaced by the variant metric."""
    pv, ev = {}, {}
    for ds, E in eng.items():
        Ev = dict(E)
        Ev["sub_s"], Ev["bb_s"], Ev["null_s"] = E["sub_s"].copy(), E["bb_s"].copy(), E["null_s"].copy()
        Ev["bb_point"] = np.array(E["bb_point"], float).copy()
        pv[ds] = dict(point[ds])
        for v, mi in (("rsa_cc", IR), ("mknn_chance", IM)):
            if v not in which:
                continue
            s, b = sub[ds][v]
            Ev["sub_s"][:, mi] = s
            Ev["bb_s"][:, mi] = b
            Ev["null_s"][:, mi] = np.nan  # no shuffled-stimulus null for the variants (H4 rows are not used)
            pv[ds][METRICS[mi]] = sc.nanmean0(pts[ds][v][0], -1)
            Ev["bb_point"][mi] = np.nanmean(pts[ds][v][1]) if np.isfinite(pts[ds][v][1]).any() else np.nan
        st._derive(Ev)
        ev[ds] = Ev
    return pv, ev


def _boot_ci(x, y, n_boot=N_BOOT, seed=SEED_BOOT):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = [st.spearman(x[i], y[i]) for i in rng.integers(0, len(x), (n_boot, len(x)))]
    return st.ci(v)


def brain_reference(D):
    """Like-for-like brain-brain reference: split halves of subject s (as for models) against (i) the mean full
    kernel of the other subjects, (ii) each other subject's full kernel (averaged); divided by sqrt(ceiling(s))."""
    FB, L = sc.brain_feats(D, exact=True)
    rows = []
    if L.bb is None:
        return rows
    for metric in METRICS:
        H = L.halves
        ceil = FB.rowpair(H[:, :, 0].reshape(-1), H[:, :, 1].reshape(-1), metric).reshape(L.nsub, -1).mean(1)
        for s in range(L.nsub):
            hs = H[s].reshape(-1)
            others = [o for o in range(L.nsub) if o != s]
            gm = FB.pair(hs, FB, [L.bb[s]], metric).mean()
            single = FB.pair(hs, FB, L.full[others], metric).mean()
            full_gm = FB.rowpair([L.full[s]], [L.bb[s]], metric)[0]
            for name, raw in (("REF_full_vs_groupmean", full_gm), ("REF_halves_vs_groupmean", gm),
                              ("REF_halves_vs_single_subject", single)):
                rows.append(dict(dataset=D.ds, subject=D.subjects[s], metric=metric, reference=name, raw=raw,
                                 ceiling=ceil[s], corrected=raw / np.sqrt(ceil[s]) if ceil[s] > 0 else np.nan))
    return rows


def run(root, meta, point, eng, subj, n_sub=st.N_SUB, n_perm=st.N_PERM, jobs=None, verbose=True):
    root = sc.art_root(root)
    out = sc.results_dir(root)
    jobs = jobs or int(os.environ.get("PBI_JOBS", 0)) or max(1, min(8, (os.cpu_count() or 2) - 2))
    t0 = time.time()
    snaps = {ds: (tuple(E["models"]), tuple(E["subjects"])) for ds, E in eng.items()}
    pts, refs, var_rows, nstim = {}, [], [], {}
    for ds in eng:
        D = sc.Dataset(ds, root, *snaps[ds])
        pts[ds] = _point(D)
        nstim[ds] = D.n
        refs += brain_reference(D)
        for v, mname in (("rsa_cc", "rsa_cc"), ("mknn_chance", "mknn_chance")):
            cor, _, raw, ceil = pts[ds][v][:4]
            for j, m in enumerate(D.models):
                inc = np.isfinite(cor[j])
                var_rows.append(dict(dataset=ds, model=m, metric=mname,
                                     raw=float(np.mean(raw[j][inc])) if inc.any() else np.nan,
                                     corrected=float(np.mean(cor[j][inc])) if inc.any() else np.nan,
                                     n_subjects=int(inc.sum())))
        del D
    sub = run_engine(root, eng, n_sub, jobs, verbose)
    rows, V = [], {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for v, which in (("rsa_cc", ("rsa_cc",)), ("mknn_chance", ("mknn_chance",)), ("both", ("rsa_cc", "mknn_chance"))):
            pv, ev = _variant(eng, point, pts, sub, which)
            Hv, _, _ = st.test_hypotheses(meta, pv, ev, n_perm)
            Hv = Hv[Hv["hypothesis"].isin(["H1", "H2", "H3", "H5", "H6", "REF_brain_heldout"])].assign(analysis=v)
            vv = st.verdicts(Hv)
            V[v] = {h: vv[h] for h in ("H1", "H2", "H5", "H6")}
            rows.append(Hv)
            # model-bootstrap CI for the variant's H2 rows
            for ds in eng:
                tv = [m for m in eng[ds]["models"] if sc.is_trained_vision(meta, m)]
                pos = {m: i for i, m in enumerate(eng[ds]["models"])}
                cap = meta.loc[tv, "capability"].to_numpy(float)
                for mt in METRICS:
                    lo, hi = _boot_ci(cap, pv[ds][mt][[pos[m] for m in tv]])
                    rows.append(pd.DataFrame([dict(st._row("H2_model_bootstrap", ds, mt, st.spearman(
                        cap, pv[ds][mt][[pos[m] for m in tv]]), lo, hi, note="CI: models resampled with replacement"),
                        analysis=v)]))
        # primary: model-bootstrap CI and raw-score H2
        S = subj[subj["included"].astype(bool)].groupby(["dataset", "model", "metric"])["raw"].mean()
        for ds in eng:
            tv = [m for m in eng[ds]["models"] if sc.is_trained_vision(meta, m)]
            pos = {m: i for i, m in enumerate(eng[ds]["models"])}
            cap = meta.loc[tv, "capability"].to_numpy(float)
            for mi, mt in enumerate(METRICS):
                y = point[ds][mt][[pos[m] for m in tv]]
                lo, hi = _boot_ci(cap, y)
                rows.append(pd.DataFrame([dict(st._row("H2_model_bootstrap", ds, mt, st.spearman(cap, y), lo, hi,
                                                       note="CI: models resampled with replacement"), analysis="primary")]))
                yr = np.array([S.get((ds, m, mt), np.nan) for m in tv])
                rho, p = st.spearman_perm(cap, yr, n_perm, st.SEED_PERM + 100 * st.DATASETS.index(ds) + 10 * mi + 2)
                rows.append(pd.DataFrame([dict(st._row("H2_raw", ds, mt, rho, p=p, n=len(tv),
                                                       note="uncorrected raw scores, mean over subjects"),
                                               analysis="raw_uncorrected")]))
    T = pd.concat(rows, ignore_index=True)
    m = (T["analysis"] == "raw_uncorrected")
    T.loc[m, "p_holm"] = st.holm(T.loc[m, "p"].to_numpy(float), 9)
    R = pd.DataFrame(refs)
    Rm = (R.groupby(["dataset", "metric", "reference"], sort=False)["corrected"].mean().reset_index()
          if len(R) else pd.DataFrame(columns=["dataset", "metric", "reference", "corrected"]))
    for _, r in Rm.iterrows():
        T = pd.concat([T, pd.DataFrame([dict(st._row(r["reference"], r["dataset"], r["metric"], r["corrected"],
                                                     note="brain-brain reference, mean over subjects"),
                                             analysis="reference")])], ignore_index=True)
    floor = _chance_floor(subj, nstim)
    cols = ["analysis", "hypothesis", "dataset", "metric", "effect", "ci_lo", "ci_hi", "p", "p_holm", "confirmed",
            "n", "note"]
    T[cols].to_csv(out / "robustness.csv", index=False, float_format="%.4f")
    pd.DataFrame(var_rows).to_csv(out / "scores_variants.csv", index=False, float_format="%.10g")
    write_md(T, V, floor, point, eng, meta, out / "robustness.md")
    if verbose:
        print(f"[robustness] done in {time.time() - t0:.0f} s")
    return T, V, floor


def _chance_floor(subj, nstim):
    """Mean over included subjects of chance (k/(n-1)) expressed in corrected units, c / sqrt(ceiling)."""
    out = {}
    g = subj[(subj["metric"] == "mknn") & subj["included"].astype(bool)]
    for ds, gg in g.groupby("dataset"):
        if ds not in nstim:
            continue
        n = nstim[ds]
        c = sc.kn.default_k(n) / (n - 1)
        ce = gg.groupby("subject")["ceiling"].first()
        out[ds] = dict(chance=c, floor=float(np.mean(c / np.sqrt(ce))), ceil_min=float(ce.min()), ceil_max=float(ce.max()))
    return out


def _cellrow(T, a, h, ds, mt):
    r = T[(T["analysis"] == a) & (T["hypothesis"] == h) & (T["dataset"] == ds) & (T["metric"] == mt)]
    return r.iloc[0] if len(r) else None


def write_md(T, V, floor, point, eng, meta, path):
    f = st._fmt
    L = ["# EXPLORATORY robustness analyses (machine-generated by `python -m pbi stats`; do not edit)", "",
         "Added after the independent review (`REVIEW.md`). **Not preregistered; they do not replace any verdict "
         "in `hypotheses.md`.** Same split halves, 1000 stimulus subsamples, 10,000 permutations and Holm over 9 "
         "cells as the primary analysis. Full rows: `artifacts/results/robustness.csv`; per-model variant scores: "
         "`artifacts/results/scores_variants.csv`. Definitions: see the docstring of `pbi/robustness.py`.", "",
         "## Verdicts under each variant (exploratory)", "", "| Variant | H1 | H2 | H5 | H6 |", "|---|---|---|---|---|"]
    for v, desc in VARIANTS.items():
        L.append(f"| {v}: {desc} | " + " | ".join(
            f"{'PASS' if V[v][h]['passed'] else 'FAIL'} ({V[v][h]['detail'].split(' (')[0]})" for h in ("H1", "H2", "H5", "H6")) + " |")
    L += ["", "## H1 / H2 / H5 per cell: primary vs variants", "",
          "| Test | Dataset | Metric | Primary | rsa_cc / mknn_chance variant | variant 95% CI | variant p (Holm) | confirmed |",
          "|---|---|---|---|---|---|---|---|"]
    for h in ("H1", "H2", "H5"):
        for ds in st.DATASETS:
            for mt in METRICS:
                v = "rsa_cc" if mt == "rsa" else "mknn_chance" if mt == "mknn" else None
                if v is None:
                    continue
                r = _cellrow(T, v, h, ds, mt)
                prim = _cellrow(T, "mknn_chance" if v == "rsa_cc" else "rsa_cc", h, ds, mt)  # untouched slot = primary
                if r is None:
                    continue
                L.append(f"| {h} | {sc.DS_SHORT[ds]} | {mt} -> {v} | {f(prim['effect']) if prim is not None else ''} | "
                         f"{f(r['effect'])} | [{f(r['ci_lo'])}, {f(r['ci_hi'])}] | {f(r['p_holm'], 4)} | "
                         f"{'yes' if r['confirmed'] else 'no'} |")
    L += ["", "## H2 with model-resampling bootstrap CIs and on raw (uncorrected) scores", "",
          "The CIs in `hypotheses.md` resample stimuli only (model set fixed). Here models are resampled with "
          f"replacement ({N_BOOT} draws).", "",
          "| Dataset | Metric | primary rho | model-bootstrap 95% CI | raw-score rho | raw p (Holm) |", "|---|---|---|---|---|---|"]
    for ds in st.DATASETS:
        for mt in METRICS:
            b = _cellrow(T, "primary", "H2_model_bootstrap", ds, mt)
            r = _cellrow(T, "raw_uncorrected", "H2_raw", ds, mt)
            if b is None:
                continue
            L.append(f"| {sc.DS_SHORT[ds]} | {mt} | {f(b['effect'])} | [{f(b['ci_lo'])}, {f(b['ci_hi'])}] | "
                     f"{f(r['effect']) if r is not None else ''} | {f(r['p_holm'], 4) if r is not None else ''} |")
    L += ["", "## Mutual k-NN chance floor", "",
          "Chance for raw mutual k-NN is c = k/(n-1). After the preregistered correction (raw / sqrt(ceiling)) chance "
          "sits at c / sqrt(ceiling), not at 0, so corrected mutual k-NN values are not comparable across datasets "
          "and '1.0 = ceiling' does not mean '0 = chance'.", "",
          "| Dataset | chance c | ceiling range | chance in corrected units (mean over subjects) |", "|---|---|---|---|"]
    for ds in st.DATASETS:
        if ds in floor:
            x = floor[ds]
            L.append(f"| {sc.DS_SHORT[ds]} | {x['chance']:.4f} | {x['ceil_min']:.3f} - {x['ceil_max']:.3f} | {x['floor']:.3f} |")
    L += ["", "## Brain-brain reference, like-for-like", "",
          "`REF_brain_heldout` in `hypotheses.md` compares a subject's *full* kernel with the mean of the other "
          "subjects, while models are scored against *split halves*; that inflates the reference. The like-for-like "
          "versions below score the subject's split halves, exactly as models are scored.", "",
          "| Dataset | Metric | full vs group mean (as in hypotheses.md) | halves vs group mean | halves vs one other subject | best model (primary) |",
          "|---|---|---|---|---|---|"]
    for ds in st.DATASETS:
        for mt in METRICS:
            a = _cellrow(T, "reference", "REF_full_vs_groupmean", ds, mt)
            b = _cellrow(T, "reference", "REF_halves_vs_groupmean", ds, mt)
            c = _cellrow(T, "reference", "REF_halves_vs_single_subject", ds, mt)
            if a is None:
                continue
            E = eng[ds]
            nonrand = [i for i, m in enumerate(E["models"]) if not sc.is_random(meta, m)]
            best = np.nanmax(point[ds][mt][nonrand])
            L.append(f"| {sc.DS_SHORT[ds]} | {mt} | {f(a['effect'])} | {f(b['effect'])} | {f(c['effect'])} | {f(best)} |")
    L.append("")
    path.write_text("\n".join(L))
