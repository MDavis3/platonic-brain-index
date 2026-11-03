"""Preregistered hypothesis tests H1-H6, nulls and controls (PREREGISTRATION.md, "Hypotheses" / "Nulls and controls").

Inputs: artifacts/kernels/** (for subsamples / nulls), artifacts/results/scores*.csv (point estimates from
pbi.score), artifacts/models.csv. All randomness uses fixed seeds; results do not depend on the number of
worker processes (each subsample / permutation has its own seed).

Definitions used throughout (one "cell" = dataset x metric):
  trained vision (TV) models = modality 'vision' and objective != 'random'; random-init = objective 'random'.
  CI      = 2.5 / 97.5 percentiles over N_SUB = 1000 subsamples of 80 % of stimuli; every kernel (models,
            split halves, full subject kernels) is restricted to the subset and all scores, ceilings and
            the statistic are recomputed (mutual k-NN uses k = default_k(n_subset) = round(n_subset/20)).
  perm p  = one-sided (effect > 0), (1 + #{null >= observed}) / (1 + N_PERM), N_PERM = 10,000 permutations
            of model labels (correlations) or trained/random labels (H1).
  Holm    = Holm-Bonferroni across the 9 planned cells (cells without data enter as p = 1).
  confirmed (per cell) = p_holm < 0.05 AND CI excludes 0 (H6: upper CI < 1).
  shuffled-stimulus null = 1000 permutations of the model kernel's stimulus order (one permutation shared by
            all models per draw), scored against the same split halves and ceiling-corrected the same way.

Outputs (artifacts/results/): hypotheses.csv, hypotheses.md, null_pvalues.csv, loso.csv,
family_scaling.csv, model_ci.csv, layers.csv (if artifacts/layers exists).
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import kernels as kn
from . import score as sc

N_SUB, N_NULL, N_PERM = 1000, 1000, 10_000
SUB_FRAC = 0.8
SEED_SUB, SEED_NULL, SEED_PERM, SEED_LAYER = 20261102, 30261102, 40261102, 50261102
K_SENS = (5, 10, 20)
ALPHA = 0.05
ENGINE_VERSION = "3"  # 2: per-subject subsample/null scores, model-model matrix, layer files in fingerprint;
#                       3: source-code hash in fingerprint, cache moved into the repo (.cache/stats)
METRICS = sc.METRICS
DATASETS = sc.DATASETS
# Systems named in PREREGISTRATION.md: H3 families ("ResNet, ViT, ConvNeXt, DINOv2, MAE, CLIP-LAION"), the LM
# families of the H4 scale test ("Pythia, Qwen"), and the preregistered language-only systems (Pythia, Qwen,
# CLIP text towers; SBERT is exploratory and not part of H4).
H3_FAMILIES = ("resnet", "vit_sup", "convnext", "dinov2", "mae", "clip_laion")
H4_SCALE_FAMILIES = ("pythia", "qwen2.5")
H4_LANGUAGE_OBJECTIVES = ("lm", "clip_text")


# ------------------------------------------------------------------ small statistics

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _corr_rows(A, b):
    """Pearson correlation of each row of A with vector b."""
    A = A - A.mean(1, keepdims=True)
    b = b - b.mean()
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(b)
    return np.divide(A @ b, den, out=np.full(len(A), np.nan), where=den > 0)


def perm_matrix(n, n_perm, seed):
    return np.argsort(np.random.default_rng(seed).random((n_perm, n)), axis=1)


def perm_p(obs, null):
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return np.nan
    return float((1 + np.sum(null >= obs - 1e-12)) / (1 + len(null)))


def spearman_perm(x, y, n_perm, seed):
    """Observed Spearman rho and one-sided permutation p (labels of x permuted)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    obs = spearman(x, y)
    P = perm_matrix(ok.sum(), n_perm, seed)
    return obs, perm_p(obs, _corr_rows(rx[P], ry))


def partial_spearman(x, y, z):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 4:
        return np.nan
    rxy, rxz, ryz = spearman(x[ok], y[ok]), spearman(x[ok], z[ok]), spearman(y[ok], z[ok])
    if not np.isfinite(rxz) or not np.isfinite(ryz):
        return rxy
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / den) if den > 0 else np.nan


def partial_spearman_perm(x, y, z, n_perm, seed):
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    obs = partial_spearman(x, y, z)
    if not np.isfinite(obs):
        return obs, np.nan
    rx, ry, rz = rankdata(x[ok]), rankdata(y[ok]), rankdata(z[ok])
    P = perm_matrix(ok.sum(), n_perm, seed)
    rxy, rxz = _corr_rows(rx[P], ry), _corr_rows(rx[P], rz)
    ryz = np.corrcoef(ry, rz)[0, 1] if rz.std() > 0 else 0.0
    null = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return obs, perm_p(obs, null)


def ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def holm(p, m_total=None):
    """Holm-adjusted p-values; NaN p's are treated as untestable cells with p = 1 (conservative)."""
    p = np.asarray(p, float)
    m = max(m_total or len(p), len(p))
    q = np.where(np.isfinite(p), p, 1.0)
    full = np.concatenate([q, np.ones(m - len(p))])
    order = np.argsort(full, kind="stable")
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * full[i]))
        adj[i] = run
    out = adj[:len(p)]
    return np.where(np.isfinite(p), out, np.nan)


# ------------------------------------------------------------------ engine (subsamples, nulls)

_W = {}
_TPL = None


def _init_worker():
    global _TPL
    try:
        from threadpoolctl import threadpool_limits
        _TPL = threadpool_limits(1)
    except Exception:  # pragma: no cover
        pass


def _get(ds, root, snap):
    key = (ds, str(root))
    if key not in _W:
        _W.clear()
        D = sc.Dataset(ds, root, *snap)
        assert D.models == list(snap[0]) and D.subjects == list(snap[1]), "kernel files changed during the run"
        meta = sc.load_models(root)
        tv = np.array([i for i, m in enumerate(D.models) if sc.is_trained_vision(meta, m)], int)
        _W[key] = {"D": D, "tv": tv}
    return _W[key]


def _subsample_idx(n, b):
    rng = np.random.default_rng(SEED_SUB + b)
    return np.sort(rng.choice(n, int(round(SUB_FRAC * n)), replace=False))


def _mm_mean(FM, tv):
    if len(tv) < 2:
        return np.full(len(tv), np.nan)
    M = FM.pair(tv, FM, tv, "mknn")
    np.fill_diagonal(M, np.nan)
    return np.nanmean(M, 1)


def _mm_matrix(FM, tv):
    """Model-model mutual k-NN among trained vision models (diagonal NaN)."""
    M = FM.pair(tv, FM, tv, "mknn") if len(tv) else np.zeros((0, 0))
    np.fill_diagonal(M, np.nan)
    return M


def _task(args):
    kind, ds, root, seeds, snap = args
    W = _get(ds, root, snap)
    D, tv = W["D"], W["tv"]
    bB, oB, bM, oM = D.banks()
    nM = len(D.models)
    im = np.arange(nM)
    if kind == "sub":
        ns = len(D.subjects)
        S = np.full((len(seeds), len(METRICS), nM, ns), np.nan)   # per-subject corrected scores
        BB = np.full((len(seeds), len(METRICS), ns), np.nan)
        MM = np.full((len(seeds), len(tv), len(tv)), np.nan)
        for j, b in enumerate(seeds):
            idx = _subsample_idx(D.n, b)
            FB, L = sc.brain_feats(D, idx)
            FM = sc.Feats(bM, idx, orders=oM)
            for mi, metric in enumerate(METRICS):
                raw, ceil, bb = sc.block_scores(FM, im, FB, L, metric)
                S[j, mi] = sc.corrected(raw, ceil)
                BB[j, mi] = sc.corrected(bb, ceil)
            MM[j] = _mm_matrix(FM, tv)
        return kind, ds, seeds, (S, BB, MM)
    if kind == "null":
        if "FB_full" not in W:
            W["FB_full"] = sc.brain_feats(D, exact=True)
        FB, L = W["FB_full"]
        S = np.full((len(seeds), len(METRICS), nM, len(D.subjects)), np.nan)  # per-subject corrected
        for j, p in enumerate(seeds):
            perm = np.random.default_rng(SEED_NULL + p).permutation(D.n)
            FM = sc.Feats(bM, perm, orders=oM)
            for mi, metric in enumerate(METRICS):
                raw, ceil, _ = sc.block_scores(FM, im, FB, L, metric)
                S[j, mi] = sc.corrected(raw, ceil)
        return kind, ds, seeds, (S,)
    if kind == "k":
        out = {}
        for k in seeds:
            FB, L = sc.brain_feats(D, k=k, exact=True, metrics=("mknn",))
            FM = sc.Feats(bM, k=k, exact=True, metrics=("mknn",))
            raw, ceil, _ = sc.block_scores(FM, im, FB, L, "mknn")
            out[k] = sc.dataset_scores(raw, ceil)[0]
        return kind, ds, seeds, (out,)
    if kind == "layers":
        return kind, ds, seeds, (_layer_sweep(D, root),)
    raise ValueError(kind)


def _layer_sweep(D, root):
    """Secondary analysis: best of 5 depth-fraction layers selected on a random half of stimuli, scored on the other."""
    ldir = sc.art_root(root) / "layers" / D.ds
    if not ldir.exists():
        return []
    rng = np.random.default_rng(SEED_LAYER)
    perm = rng.permutation(D.n)
    sel, ev = np.sort(perm[:D.n // 2]), np.sort(perm[D.n // 2:])
    FBs, Ls = sc.brain_feats(D, sel, exact=True)
    FBe, Le = sc.brain_feats(D, ev, exact=True)
    rows = []
    for mdir in sorted(p for p in ldir.iterdir() if p.is_dir()):
        files = [mdir / f"layer{i}.npz" for i in range(5)]
        if not all(f.exists() for f in files):
            continue
        Ks = [sc.load_kernel(f) for f in files]
        if len(Ks[0]["cos"]) != D.n:
            continue
        Fs, Fe = sc.Feats(Ks, sel, exact=True), sc.Feats(Ks, ev, exact=True)
        for metric in METRICS:
            s_sel = sc.dataset_scores(*sc.block_scores(Fs, np.arange(5), FBs, Ls, metric)[:2])[0]
            s_ev = sc.dataset_scores(*sc.block_scores(Fe, np.arange(5), FBe, Le, metric)[:2])[0]
            best = int(np.nanargmax(s_sel)) if np.isfinite(s_sel).any() else -1
            rows.append(dict(dataset=D.ds, model=mdir.name, metric=metric, best_layer=best,
                             score_best_layer_heldout=s_ev[best] if best >= 0 else np.nan,
                             score_final_layer_heldout=s_ev[4],
                             **{f"sel_layer{i}": s_sel[i] for i in range(5)}))
    return rows


def _code_hash():
    """Hash of the source code the cached engine results depend on (kernels.py and score.py in full, plus the
    engine functions of this module and pbi.robustness), so a code change can never reuse stale cache entries."""
    import inspect
    from . import robustness as rb
    h = hashlib.sha1()
    for mod in (kn, sc):
        h.update(Path(mod.__file__).read_bytes())
    for fn in (_task, _layer_sweep, _subsample_idx, _mm_matrix, _mm_mean, _get, rb._task, rb._scores, rb._cc,
               rb._cc_bank, rb.chance_corrected):
        h.update(inspect.getsource(fn).encode())
    return h.hexdigest()


def _fingerprint(root, ds, n_sub, n_null, snap):
    h = hashlib.sha1(f"{ENGINE_VERSION}|{ds}|{n_sub}|{n_null}|{SUB_FRAC}|{SEED_SUB}|{SEED_NULL}|{_code_hash()}".encode())
    kdir = sc.art_root(root) / "kernels" / ds
    files = [kdir / "model" / f"{m}.npz" for m in snap[0]]
    files += [kdir / "brain" / f"{s}_{t}.npz" for s in snap[1]
              for t in ["full"] + [f"split{i}_{h}" for i in range(sc.N_SPLITS) for h in "AB"]]
    for f in files:
        h.update(str(f.relative_to(kdir)).encode())
        h.update(f.read_bytes())
    ldir = sc.art_root(root) / "layers" / ds  # the layer sweep is cached too
    for f in sorted(ldir.glob("*/layer*.npz")) if ldir.exists() else []:
        h.update(str(f.relative_to(ldir)).encode())
        h.update(f.read_bytes())
    h.update(sc.load_models(root)[["model_id", "modality", "objective"]].to_csv().encode())
    return h.hexdigest()[:16]


def _cache_dir():
    d = os.environ.get("PBI_CACHE")
    if d == "0":
        return None
    # default: inside the repository (git-ignored), so a fresh clone never reuses another checkout's results
    d = Path(d) if d else sc.REPO / ".cache" / "stats"
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        return None


def run_engine(root, datasets, n_sub=N_SUB, n_null=N_NULL, jobs=None, verbose=True, snapshots=None):
    """Subsample and null distributions per dataset: {ds: dict(models, tv, sub, bb, mm, null, k)}.

    snapshots: {ds: (models, subjects)} fixes the systems (the ones pbi.score scored), so kernels that other
    processes add while this runs are ignored."""
    jobs = jobs or int(os.environ.get("PBI_JOBS", 0)) or max(1, min(8, (os.cpu_count() or 2) - 2))
    cdir = _cache_dir()
    res, tasks = {}, []
    for ds in datasets:
        if snapshots and ds in snapshots:
            snap = tuple(tuple(x) for x in snapshots[ds])
            D = sc.Dataset(ds, root, *snap)
        else:
            D = sc.Dataset(ds, root)
            snap = (tuple(D.models), tuple(D.subjects))
        if not D.ok:
            continue
        meta = sc.load_models(root)
        tv = [m for m in D.models if sc.is_trained_vision(meta, m)]
        fp = _fingerprint(root, ds, n_sub, n_null, snap)
        cpath = cdir / f"{ds}_{fp}.npz" if cdir else None
        if cpath and cpath.exists():
            z = np.load(cpath, allow_pickle=True)
            res[ds] = {k: z[k] for k in z.files}
            res[ds]["k"] = z["k"].item()
            res[ds]["layers"] = list(z["layers"])
            res[ds]["models"] = list(z["models"])
            res[ds]["tv"] = list(z["tv"])
            res[ds]["subjects"] = list(z["subjects"])
            _derive(res[ds])
            if verbose:
                print(f"[stats] {ds}: engine results from cache {cpath.name}")
            continue
        M = len(D.models)
        ns = len(D.subjects)
        res[ds] = dict(models=D.models, tv=tv, subjects=list(D.subjects), cpath=cpath,
                       sub_s=np.full((n_sub, len(METRICS), M, ns), np.nan),
                       bb_s=np.full((n_sub, len(METRICS), ns), np.nan),
                       mm_mat=np.full((n_sub, len(tv), len(tv)), np.nan),
                       null_s=np.full((n_null, len(METRICS), M, ns), np.nan),
                       k={}, layers=[])
        del D
        ch = 10
        tasks += [("k", ds, root, list(K_SENS), snap), ("layers", ds, root, [0], snap)]
        tasks += [("sub", ds, root, list(range(i, min(i + ch, n_sub))), snap) for i in range(0, n_sub, ch)]
        tasks += [("null", ds, root, list(range(i, min(i + ch, n_null))), snap) for i in range(0, n_null, ch)]
    if tasks:
        t0 = time.time()
        if verbose:
            print(f"[stats] engine: {len(tasks)} tasks on {jobs} processes "
                  f"({n_sub} subsamples, {n_null} null permutations per dataset)", flush=True)
        if jobs > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(jobs, initializer=_init_worker) as pool:
                it = pool.imap(_task, tasks, chunksize=1)
                _collect(res, it, len(tasks), t0, verbose)
        else:
            _collect(res, map(_task, tasks), len(tasks), t0, verbose)
        for ds, r in res.items():
            cpath = r.pop("cpath", None)
            if cpath is not None:
                try:
                    np.savez_compressed(cpath, sub_s=r["sub_s"], bb_s=r["bb_s"], mm_mat=r["mm_mat"],
                                        null_s=r["null_s"], k=np.array(r["k"], dtype=object),
                                        layers=np.array(r["layers"], dtype=object), models=np.array(r["models"]),
                                        tv=np.array(r["tv"]), subjects=np.array(r["subjects"]))
                except OSError:
                    pass
            _derive(r)
    for r in res.values():
        r.pop("cpath", None)
    return res


def _derive(r, keep_subjects=None, keep_tv=None):
    """Dataset-level arrays from the per-subject engine output (optionally restricted to subjects per metric
    and/or to a subset of the trained vision models for the model-model agreement):
      sub (n_sub, metric, model), bb (n_sub, metric), null (n_null, metric, model), mm (n_sub, n_tv)."""
    S, BB, NL = r["sub_s"], r["bb_s"], r["null_s"]
    if keep_subjects is not None:  # {metric index: boolean mask over subjects}
        S, BB, NL = S.copy(), BB.copy(), NL.copy()
        for mi, keep in keep_subjects.items():
            S[:, mi, :, ~keep] = np.nan
            BB[:, mi, ~keep] = np.nan
            NL[:, mi, :, ~keep] = np.nan
    r["sub"], r["bb"], r["null"] = sc.nanmean0(S, -1), sc.nanmean0(BB, -1), sc.nanmean0(NL, -1)
    M = r["mm_mat"]
    if keep_tv is not None:
        M = M[:, keep_tv][:, :, keep_tv]
    with np.errstate(all="ignore"):
        r["mm"] = np.nanmean(M, 2) if M.shape[1] >= 2 else np.full(M.shape[:2], np.nan)
    return r


def _collect(res, it, n, t0, verbose):
    for i, (kind, ds, seeds, out) in enumerate(it):
        r = res[ds]
        if kind == "sub":
            r["sub_s"][seeds], r["bb_s"][seeds], r["mm_mat"][seeds] = out
        elif kind == "null":
            r["null_s"][seeds] = out[0]
        elif kind == "k":
            r["k"].update(out[0])
        elif kind == "layers":
            r["layers"] = out[0]
        if verbose and (i + 1) % max(1, n // 10) == 0:
            print(f"[stats]   {i + 1}/{n} tasks ({time.time() - t0:.0f} s)", flush=True)


# ------------------------------------------------------------------ hypotheses

def _row(h, ds, metric, effect, lo=np.nan, hi=np.nan, p=np.nan, n=np.nan, note=""):
    return dict(hypothesis=h, dataset=ds, metric=metric, effect=effect, ci_lo=lo, ci_hi=hi, p=p,
                p_holm=np.nan, confirmed=False, n=n, note=note)


def _families(meta, models, pred):
    fam = {}
    for m in models:
        if pred(m) and np.isfinite(meta.at[m, "params"]) and meta.at[m, "params"] > 0:
            fam.setdefault(meta.at[m, "family"], []).append(m)
    return {f: ms for f, ms in fam.items() if len(set(meta.loc[ms, "params"])) >= 3}


def _family_rho_mean(fams, meta, models, score_vec):
    pos = {m: i for i, m in enumerate(models)}
    rhos = {}
    for f, ms in fams.items():
        y = np.array([score_vec[pos[m]] for m in ms])
        rhos[f] = spearman(np.log10(meta.loc[ms, "params"].to_numpy(float)), y)
    v = np.array([r for r in rhos.values() if np.isfinite(r)])
    return (float(v.mean()) if len(v) else np.nan), rhos


def _family_perm(fams, meta, models, score_vec, n_perm, seed):
    """Null of the mean within-family rho: params permuted within each family."""
    pos = {m: i for i, m in enumerate(models)}
    acc, cnt = np.zeros(n_perm), 0
    for j, (f, ms) in enumerate(sorted(fams.items())):
        y = np.array([score_vec[pos[m]] for m in ms])
        x = np.log10(meta.loc[ms, "params"].to_numpy(float))
        ok = np.isfinite(y)
        if ok.sum() < 3:
            continue
        rx, ry = rankdata(x[ok]), rankdata(y[ok])
        if rx.std() == 0 or ry.std() == 0:
            continue
        acc += _corr_rows(rx[perm_matrix(ok.sum(), n_perm, seed + j)], ry)
        cnt += 1
    return acc / cnt if cnt else np.full(n_perm, np.nan)


def test_hypotheses(meta, point, eng, n_perm=N_PERM, subj=None):
    """point: {ds: {metric: np.array over eng[ds]['models']}}; returns hypotheses DataFrame + extra tables."""
    rows, fam_rows, null_rows = [], [], []
    for ds in DATASETS:
        if ds not in eng:
            continue
        E = eng[ds]
        models = list(E["models"])
        pos = {m: i for i, m in enumerate(models)}
        tv = [m for m in models if sc.is_trained_vision(meta, m)]
        rnd = [m for m in models if sc.is_random(meta, m)]
        lm = [m for m in models if m in meta.index and meta.at[m, "modality"] == "language"
              and meta.at[m, "objective"] in H4_LANGUAGE_OBJECTIVES]
        itv, irnd = [pos[m] for m in tv], [pos[m] for m in rnd]
        cap = meta.loc[tv, "capability"].to_numpy(float) if tv else np.array([])
        logw = np.log10(meta.loc[tv, "emb_dim"].to_numpy(float)) if tv else np.array([])
        mm_point = E["mm_point"]
        vfam = _families(meta, models, lambda m: sc.is_trained_vision(meta, m) and meta.at[m, "family"] in H3_FAMILIES)
        lfam = _families(meta, models, lambda m: m in lm and meta.at[m, "family"] in H4_SCALE_FAMILIES)
        for mi, metric in enumerate(METRICS):
            y = point[ds][metric]
            sub = E["sub"][:, mi, :]
            seed = SEED_PERM + 100 * DATASETS.index(ds) + 10 * mi
            # H1
            a, b = y[itv], y[irnd]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) >= 1 and len(b) >= 1:
                eff = a.mean() - b.mean()
                pool = np.concatenate([a, b])
                P = perm_matrix(len(pool), n_perm, seed + 1)
                lab = np.zeros(len(pool), bool)
                lab[:len(a)] = True
                Lp = lab[P]
                null = (np.where(Lp, pool, 0).sum(1) / len(a)) - (np.where(~Lp, pool, 0).sum(1) / len(b))
                d = sc.nanmean0(sub[:, itv]) - sc.nanmean0(sub[:, irnd])
                rows.append(_row("H1", ds, metric, eff, *ci(d), perm_p(eff, null), len(a) + len(b),
                                 f"{len(a)} trained vs {len(b)} random-init"))
            else:
                rows.append(_row("H1", ds, metric, np.nan, note="missing trained or random-init models"))
            # H2 (+ width-partial)
            ok = np.isfinite(cap) & np.isfinite(y[itv]) if tv else np.array([], bool)
            if ok.sum() >= 3:
                ic = np.array(itv)[ok]
                rho, p = spearman_perm(cap[ok], y[ic], n_perm, seed + 2)
                d = [spearman(cap[ok], sub[bb, ic]) for bb in range(len(sub))]
                rows.append(_row("H2", ds, metric, rho, *ci(d), p, ok.sum()))
                prho, pp = partial_spearman_perm(cap[ok], y[ic], logw[ok], n_perm, seed + 3)
                d = [partial_spearman(cap[ok], sub[bb, ic], logw[ok]) for bb in range(len(sub))]
                rows.append(_row("H2_width_partial", ds, metric, prho, *ci(d), pp, ok.sum(),
                                 "partial Spearman controlling log10 emb_dim"))
                if metric == "mknn":
                    for k, sk in sorted(E["k"].items()):
                        rk, pk = spearman_perm(cap[ok], np.asarray(sk)[ic], n_perm, seed + 4 + k)
                        rows.append(_row(f"H2_k{k}", ds, metric, rk, p=pk, n=ok.sum(), note="k sensitivity (point + perm p)"))
            else:
                rows.append(_row("H2", ds, metric, np.nan, note="fewer than 3 trained vision models with capability"))
                rows.append(_row("H2_width_partial", ds, metric, np.nan, note="insufficient models"))
            # H3 (vision families) and LM-family scale (H4b)
            for tag, fams in (("H3", vfam), ("H4_scale", lfam)):
                if fams:
                    eff, rhos = _family_rho_mean(fams, meta, models, y)
                    null = _family_perm(fams, meta, models, y, n_perm, seed + 5 + (tag == "H4_scale"))
                    d = [_family_rho_mean(fams, meta, models, sub[bb])[0] for bb in range(len(sub))]
                    rows.append(_row(tag, ds, metric, eff, *ci(d), perm_p(eff, null), len(fams),
                                     "mean within-family rho(log10 params, score): " +
                                     ", ".join(f"{f}={r:.2f}" for f, r in sorted(rhos.items()))))
                    for f, r in sorted(rhos.items()):
                        fam_rows.append(dict(dataset=ds, metric=metric, family=f,
                                             modality="vision" if tag == "H3" else "language",
                                             n_sizes=len(fams[f]), rho=r))
                else:
                    rows.append(_row(tag, ds, metric, np.nan, note="no family with >= 3 sizes"))
            # shuffled-stimulus null for every model
            null = E["null"][:, mi, :]
            for m in models:
                j = pos[m]
                nv = null[:, j]
                null_rows.append(dict(dataset=ds, model=m, metric=metric, score=y[j],
                                      null_mean=np.nanmean(nv) if np.isfinite(nv).any() else np.nan,
                                      null_sd=np.nanstd(nv) if np.isfinite(nv).any() else np.nan,
                                      p=perm_p(y[j], nv), n_perm=int(np.isfinite(nv).sum())))
            # H4: best-of-family language models vs max-over-family null
            lfam_all = {}
            for m in lm:
                lfam_all.setdefault(meta.at[m, "family"], []).append(m)
            for f, ms in sorted(lfam_all.items()):
                js = [pos[m] for m in ms]
                if not np.isfinite(y[js]).any():
                    continue
                jb = js[int(np.nanargmax(y[js]))]
                pval = perm_p(y[jb], np.nanmax(null[:, js], 1))
                rows.append(_row(f"H4_best:{f}", ds, metric, y[jb], *ci(sub[:, jb]), pval, len(js),
                                 f"best={models[jb]}; null = max over family of shuffled-stimulus scores"))
            if lm:
                js = [pos[m] for m in lm]
                if np.isfinite(y[js]).any():
                    jb = js[int(np.nanargmax(y[js]))]
                    rows.append(_row("H4_bestLM", ds, metric, y[jb], *ci(sub[:, jb]),
                                     perm_p(y[jb], np.nanmax(null[:, js], 1)), len(js),
                                     f"best={models[jb]}; null = max over all language models"))
            else:
                rows.append(_row("H4_bestLM", ds, metric, np.nan, note="no language models"))
            # H5
            if len(tv) >= 3:
                rho, p = spearman_perm(mm_point, y[itv], n_perm, seed + 7)
                d = [spearman(E["mm"][bb], sub[bb, itv]) for bb in range(len(sub))]
                rows.append(_row("H5", ds, metric, rho, *ci(d), p, len(tv)))
            else:
                rows.append(_row("H5", ds, metric, np.nan, note="fewer than 3 trained vision models"))
            # H6
            nonrand = [pos[m] for m in models if not sc.is_random(meta, m)]
            if nonrand and np.isfinite(y[nonrand]).any():
                jb = nonrand[int(np.nanargmax(y[nonrand]))]
                smax = np.nanmax(np.where(np.isfinite(sub[:, nonrand]), sub[:, nonrand], -np.inf), 1)
                smax[~np.isfinite(smax)] = np.nan
                lo, hi = ci(smax)
                rows.append(_row("H6", ds, metric, y[jb], lo, hi, np.nan, len(nonrand),
                                 f"best={models[jb]}; CI of max over models per subsample"))
            # brain-brain reference (held-out subject)
            rows.append(_row("REF_brain_heldout", ds, metric, E["bb_point"][mi], *ci(E["bb"][:, mi]),
                             note="M(subject, mean of other subjects)/sqrt(ceiling), mean over subjects"))
    H = pd.DataFrame(rows)
    H = _apply_confirmation(H)
    return H, pd.DataFrame(fam_rows), pd.DataFrame(null_rows)


CELL_TESTS = ("H1", "H2", "H2_width_partial", "H3", "H4_scale", "H4_bestLM", "H5")


def _apply_confirmation(H):
    H = H.copy()
    H["confirmed"] = False
    for h in CELL_TESTS:
        m = H["hypothesis"] == h
        if m.any():
            H.loc[m, "p_holm"] = holm(H.loc[m, "p"].to_numpy(float), 9)
            H.loc[m, "confirmed"] = ((H.loc[m, "p_holm"] < ALPHA) & (H.loc[m, "ci_lo"] > 0)
                                     & (H.loc[m, "effect"] > 0)).to_numpy()
    m = H["hypothesis"] == "H6"
    H.loc[m, "confirmed"] = (H.loc[m, "ci_hi"] < 1).to_numpy()
    return H


def _cells(H, h, metric=None):
    d = H[(H["hypothesis"] == h) & H["effect"].notna()]
    if metric:
        d = d[d["metric"] == metric]
    return d


def verdicts(H):
    """Preregistered pass/fail per hypothesis (thresholds exactly as in PREREGISTRATION.md).

    Returns {hypothesis: dict(passed: bool, detail: str, available: int)}. Cells without data count as
    not confirmed (thresholds are absolute counts of the planned 3 datasets / 9 cells)."""
    V = {}
    c = _cells(H, "H1")
    V["H1"] = dict(passed=int(c["confirmed"].sum()) >= 7, available=len(c),
                   detail=f"confirmed in {int(c['confirmed'].sum())}/9 cells (threshold >= 7/9)")
    nds = {mt: int(_cells(H, "H2", mt)["confirmed"].sum()) for mt in METRICS}
    strict = nds["mknn"] >= 2 and (nds["cka"] >= 2 or nds["rsa"] >= 2)
    c = _cells(H, "H2")
    per_ds_or = sum(bool(c[(c["dataset"] == ds) & c["metric"].isin(["cka", "rsa"])]["confirmed"].any())
                    for ds in DATASETS)
    lenient = nds["mknn"] >= 2 and per_ds_or >= 2
    V["H2"] = dict(passed=strict, available=len(c),
                   detail=(f"confirmed datasets: mknn {nds['mknn']}/3, CKA {nds['cka']}/3, RSA {nds['rsa']}/3 "
                           f"(threshold: mknn >= 2 AND (CKA >= 2 OR RSA >= 2)); alternative reading counting a "
                           f"dataset if CKA or RSA is confirmed: {per_ds_or}/3 -> "
                           f"{'pass' if lenient else 'fail'}"))
    c = _cells(H, "H3", "mknn")
    ci_only = int(((c["effect"] > 0) & (c["ci_lo"] > 0)).sum())
    V["H3"] = dict(passed=int(c["confirmed"].sum()) >= 2, available=len(c),
                   detail=(f"mutual k-NN: mean within-family rho confirmed in {int(c['confirmed'].sum())}/3 datasets "
                           f"(threshold >= 2/3); mean rho > 0 with CI excluding 0 (ignoring p): {ci_only}/3"))
    a = H[(H["hypothesis"] == "H4_bestLM") & (H["dataset"] == "things_fmri") & (H["metric"] == "mknn")]
    b = H[(H["hypothesis"] == "H4_scale") & (H["dataset"] == "things_fmri") & (H["metric"] == "mknn")]
    # (a) and (b) are both judged with the general cell rule (Holm p < 0.05 and CI excluding 0), as H3 is. The
    # literal table text ("scale rho > 0") would accept a bare positive point estimate for (b); that reading is
    # reported in the detail but is not used for the verdict (changed after the independent review; DEVIATIONS.md).
    h4a = bool(len(a) and a["confirmed"].iloc[0])
    h4b = bool(len(b) and b["confirmed"].iloc[0])
    h4b_point = bool(len(b) and np.isfinite(b["effect"].iloc[0]) and b["effect"].iloc[0] > 0)
    def _d(r):
        if not len(r) or not np.isfinite(r["effect"].iloc[0]):
            return " (no data)"
        r = r.iloc[0]
        return f" ({r['effect']:.3f}, CI [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], p={r['p']:.4f}, Holm p={r['p_holm']:.4f})"
    V["H4"] = dict(passed=h4a and h4b, available=len(a) + len(b),
                   label=("PASS" if h4a and h4b else "PARTIAL (FAIL)" if h4a or h4b else "FAIL"),
                   detail=(f"(a) best language model above the shuffled-stimulus null on fMRI-IT (mutual k-NN): "
                           f"{'confirmed' if h4a else 'not confirmed'}{_d(a)}; "
                           f"(b) mean within-LM-family scale rho on fMRI-IT (mutual k-NN): "
                           f"{'confirmed' if h4b else 'not confirmed'}{_d(b)}. Literal reading of (b) as a bare "
                           f"point estimate rho > 0: {'met' if h4b_point else 'not met'} -> H4 would "
                           f"{'PASS' if h4a and h4b_point else 'FAIL'} under that reading"))
    c = _cells(H, "H5", "mknn")
    V["H5"] = dict(passed=int(c["confirmed"].sum()) >= 2, available=len(c),
                   detail=f"mutual k-NN: confirmed in {int(c['confirmed'].sum())}/3 datasets (threshold >= 2/3)")
    c = _cells(H, "H6")
    n6 = int(c["confirmed"].sum())
    V["H6"] = dict(passed=n6 == 9, available=len(c), detail=f"upper CI < 1 in {n6}/9 cells (threshold 9/9)")
    return V


# ------------------------------------------------------------------ controls

def loso(subj, meta):
    """Leave-one-subject-out: Spearman between held-out subject's model ranking and the others' mean ranking."""
    rows = []
    for (ds, metric), g in subj.groupby(["dataset", "metric"], sort=False):
        W = g.pivot(index="model", columns="subject", values="corrected")
        W = W.loc[[m for m in W.index if not sc.is_random(meta, m)]]
        if W.shape[1] < 2:
            continue
        for s in W.columns:
            others = W.drop(columns=s).mean(axis=1, skipna=True)
            rows.append(dict(dataset=ds, metric=metric, subject=s, rho=spearman(W[s], others),
                             n_models=int(W[s].notna().sum())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ exploratory sensitivity analyses

CEIL_MIN = 0.1           # sensitivity (b): drop subjects whose split-half ceiling for a metric is below this
SENS_EXCLUDE_FAMILY = "mae"  # sensitivity (a): drop MAE models (timm exposes MAE's untrained CLS token)


def _mm_point(mmsim, ds, tv):
    M = mmsim[mmsim["dataset"] == ds].pivot(index="model_a", columns="model_b", values="mknn")
    if len(tv) < 2:
        return np.full(len(tv), np.nan)
    Mt = M.loc[tv, tv].to_numpy(float).copy()
    np.fill_diagonal(Mt, np.nan)
    return np.nanmean(Mt, 1)


def sensitivity(meta, point, eng, subj, ref, mmsim, n_perm=N_PERM):
    """EXPLORATORY (not preregistered; added after the builders' notes, before the final integration run).

    (a) no_mae: H2, H3, H5 recomputed with the MAE models removed from the trained-vision set (and from the
        model-model agreement of H5). Everything else (subsamples, permutations, Holm over 9 cells) as primary.
    (b) ceiling_ge_0.1: every hypothesis recomputed after excluding, per dataset x metric, subjects whose
        split-half ceiling is < 0.1 (point estimate and the same subjects in every subsample / null draw).
    Returns (table, {analysis: verdicts}, {analysis: description})."""
    tabs, V, desc = [], {}, {}
    # (a)
    mae = [m for m in meta.index if meta.at[m, "family"] == SENS_EXCLUDE_FAMILY]
    meta_a = meta.drop(index=mae)
    eng_a = {}
    for ds, E in eng.items():
        Ea = dict(E)
        keep_tv = np.array([m not in mae for m in E["tv"]], bool)
        _derive(Ea, keep_tv=keep_tv)
        Ea["mm_point"] = _mm_point(mmsim, ds, [m for m in E["tv"] if m not in mae])
        eng_a[ds] = Ea
    Ha, _, _ = test_hypotheses(meta_a, point, eng_a, n_perm)
    Ha = Ha[Ha["hypothesis"].isin(["H2", "H3", "H5"])].assign(analysis="no_mae")
    va = verdicts(Ha)
    V["no_mae"] = {h: va[h] for h in ("H2", "H3", "H5")}
    desc["no_mae"] = f"H2/H3/H5 without the MAE models ({', '.join(sorted(mae))})"
    tabs.append(Ha)
    # (b)
    point_b, eng_b, dropped = {}, {}, []
    for ds, E in eng.items():
        subs = list(E["subjects"])
        g = subj[subj["dataset"] == ds]
        ce = g.groupby(["metric", "subject"])["ceiling"].first()
        keep = {mi: np.array([bool(ce.get((mt, s_), np.nan) >= CEIL_MIN) for s_ in subs]) for mi, mt in enumerate(METRICS)}
        for mi, mt in enumerate(METRICS):
            dropped += [f"{sc.DS_SHORT[ds]} {mt}: {s_} ({ce[(mt, s_)]:.3f})" for s_, k in zip(subs, keep[mi]) if not k]
        Eb = _derive(dict(E), keep_subjects=keep)
        point_b[ds] = {}
        rb = ref[(ref["dataset"] == ds) & (ref["system"] == "brain_heldout")]
        bbp = []
        for mi, mt in enumerate(METRICS):
            ok = set(np.array(subs)[keep[mi]])
            gg = g[(g["metric"] == mt) & g["subject"].isin(ok) & g["included"].astype(bool)]
            s = gg.groupby("model")["corrected"].mean()
            point_b[ds][mt] = s.reindex(list(E["models"])).to_numpy(float)
            r = rb[(rb["metric"] == mt) & rb["subject"].isin(ok) & rb["included"].astype(bool)]
            bbp.append(r["corrected"].mean() if len(r) else np.nan)
        Eb["bb_point"] = np.array(bbp)
        eng_b[ds] = Eb
    Hb, _, _ = test_hypotheses(meta, point_b, eng_b, n_perm)
    Hb = Hb.assign(analysis=f"ceiling_ge_{CEIL_MIN}")
    V[f"ceiling_ge_{CEIL_MIN}"] = verdicts(Hb)
    desc[f"ceiling_ge_{CEIL_MIN}"] = (f"all hypotheses after excluding subjects with split-half ceiling < {CEIL_MIN} "
                                      f"for a metric; excluded: " + ("; ".join(dropped) if dropped else "none"))
    T = pd.concat(tabs + [Hb], ignore_index=True)
    return T, V, desc


# ------------------------------------------------------------------ main

def run(root=None, n_sub=N_SUB, n_null=N_NULL, n_perm=N_PERM, jobs=None, verbose=True):
    root = sc.art_root(root)
    out = sc.results_dir(root)
    meta = sc.load_models(root)
    S = pd.read_csv(out / "scores.csv")
    subj = pd.read_csv(out / "scores_subject.csv")
    ref = pd.read_csv(out / "scores_reference.csv")
    mmsim = pd.read_csv(out / "model_similarity.csv")
    datasets = [d for d in DATASETS if d in set(S["dataset"])]
    t0 = time.time()
    snaps = {ds: (sorted(set(g["model"])), sorted(set(g["subject"])))
             for ds, g in subj.groupby("dataset")}
    eng = run_engine(root, datasets, n_sub, n_null, jobs, verbose, snaps)
    point = {}
    for ds, E in eng.items():
        models = list(E["models"])
        point[ds] = {}
        for metric in METRICS:
            s = S[(S["dataset"] == ds) & (S["metric"] == metric)].set_index("model")["corrected"]
            point[ds][metric] = s.reindex(models).to_numpy(float)
        E["mm_point"] = _mm_point(mmsim, ds, list(E["tv"]))
        r = ref[(ref["dataset"] == ds) & (ref["system"] == "brain_heldout") & ref["included"].astype(bool)]
        E["bb_point"] = np.array([r[r["metric"] == mt]["corrected"].mean() for mt in METRICS])
        if "k" in E:
            E["k"] = {k: np.asarray(v) for k, v in E["k"].items()}
    H, fam, nulls = test_hypotheses(meta, point, eng, n_perm)
    L = loso(subj, meta)
    for (ds, metric), g in L.groupby(["dataset", "metric"], sort=False):
        H = pd.concat([H, pd.DataFrame([_row("LOSO", ds, metric, g["rho"].mean(), g["rho"].min(), g["rho"].max(),
                                             n=len(g), note="mean (min, max) over held-out subjects of "
                                             "Spearman(held-out ranking, others' mean ranking)")])], ignore_index=True)
    order = {h: i for i, h in enumerate(["H1", "H2", "H2_width_partial", "H2_k5", "H2_k10", "H2_k20", "H3",
                                         "H4_bestLM", "H4_scale", "H5", "H6", "REF_brain_heldout", "LOSO"])}
    H["_o"] = H["hypothesis"].map(lambda h: order.get(h, 50))
    H["_d"] = H["dataset"].map(DATASETS.index)
    H["_m"] = H["metric"].map(METRICS.index)
    H = H.sort_values(["_o", "hypothesis", "_d", "_m"]).drop(columns=["_o", "_d", "_m"]).reset_index(drop=True)
    cols = ["hypothesis", "dataset", "metric", "effect", "ci_lo", "ci_hi", "p", "p_holm", "confirmed", "n", "note"]
    H[cols].to_csv(out / "hypotheses.csv", index=False, float_format="%.4f")
    fam.to_csv(out / "family_scaling.csv", index=False, float_format="%.4f")
    nulls.to_csv(out / "null_pvalues.csv", index=False, float_format="%.4g")
    L.to_csv(out / "loso.csv", index=False, float_format="%.4f")
    # per-model CIs (and the PBI's, for the leaderboard)
    ci_rows, pbi_sub = [], {}
    for ds, E in eng.items():
        for j, m in enumerate(E["models"]):
            for mi, metric in enumerate(METRICS):
                lo, hi = ci(E["sub"][:, mi, j])
                ci_rows.append(dict(dataset=ds, model=m, metric=metric, ci_lo=lo, ci_hi=hi))
                pbi_sub.setdefault(m, []).append(E["sub"][:, mi, j])
        for mi, metric in enumerate(METRICS):
            lo, hi = ci(E["bb"][:, mi])
            ci_rows.append(dict(dataset=ds, model="brain_heldout", metric=metric, ci_lo=lo, ci_hi=hi))
            pbi_sub.setdefault("brain_heldout", []).append(E["bb"][:, mi])
    for m, arrs in pbi_sub.items():
        lo, hi = ci(sc.nanmean0(np.stack(arrs, 1), 1))
        ci_rows.append(dict(dataset="ALL", model=m, metric="PBI", ci_lo=lo, ci_hi=hi))
    pd.DataFrame(ci_rows).to_csv(out / "model_ci.csv", index=False, float_format="%.4f")
    lay = [r for E in eng.values() for r in E.get("layers", [])]
    if lay:
        pd.DataFrame(lay).to_csv(out / "layers.csv", index=False, float_format="%.4f")
    settings = dict(n_sub=n_sub, n_null=n_null, n_perm=n_perm)
    sens = sensitivity(meta, point, eng, subj, ref, mmsim, n_perm)
    sens[0][["analysis"] + cols].to_csv(out / "sensitivity.csv", index=False, float_format="%.4f")
    from . import robustness
    rob = robustness.run(root, meta, point, eng, subj, n_sub, n_perm, jobs, verbose)
    write_markdown(H, verdicts(H), out / "hypotheses.md", datasets, settings, sens, rob)
    if verbose:
        print(f"[stats] done in {time.time() - t0:.0f} s")
    return H


def _fmt(x, nd=3):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _reading_notes(H, V, rob):
    """Plain-language caveats generated from the exploratory robustness results (pbi/robustness.py)."""
    T, RV, floor = rob
    f = _fmt

    def cell(a, h, ds, mt):
        r = T[(T["analysis"] == a) & (T["hypothesis"] == h) & (T["dataset"] == ds) & (T["metric"] == mt)]
        return r.iloc[0] if len(r) else None

    def prim(h, ds, mt):
        r = H[(H["hypothesis"] == h) & (H["dataset"] == ds) & (H["metric"] == mt)]
        return r.iloc[0] if len(r) else None

    L = ["## How to read these verdicts (exploratory checks; details in `robustness.md`)", ""]
    n1 = int(_cells(H, "H1")["confirmed"].sum())
    h1cc = RV["rsa_cc"]["H1"]
    rs = [(ds, prim("H1", ds, "rsa"), cell("rsa_cc", "H1", ds, "rsa")) for ds in DATASETS]
    L.append(f"* **H1 depends on the RDM definition.** Under the preregistered RDM (1 - Pearson r between stimulus "
             f"rows, feature means kept) H1 is confirmed in {n1}/9 cells; the misses are RSA cells. With column-centred "
             f"RDMs (exploratory) H1 is {'PASS' if h1cc['passed'] else 'FAIL'} ({h1cc['detail'].split(' (')[0]}); "
             "trained - random RSA: " + ", ".join(
                 f"{sc.DS_SHORT[ds]} {f(p['effect'])} -> {f(c['effect'])}" for ds, p, c in rs if p is not None and c is not None)
             + ". The preregistered verdict stands; the RSA-based part of it reflects the shared mean pattern "
               "rather than evidence that trained networks are no better than random ones.")
    h2 = [(ds, prim("H2", ds, "rsa"), cell("rsa_cc", "H2", ds, "rsa")) for ds in DATASETS]
    L.append("* **The negative H2 RSA correlations are an RDM artefact as well**: H2 rho (RSA) " + ", ".join(
        f"{sc.DS_SHORT[ds]} {f(p['effect'])} -> {f(c['effect'])}" for ds, p, c in h2 if p is not None and c is not None)
        + " with column-centred RDMs. Read them as 'not confirmed', not as 'capability hurts'. H2 fails under every "
          f"variant (column-centred RSA: {RV['rsa_cc']['H2']['detail'].split(' (')[0]}; chance-adjusted mutual k-NN: "
          f"{RV['mknn_chance']['H2']['detail'].split(' (')[0]}).")
    if floor:
        L.append("* **Corrected mutual k-NN has a chance floor above 0**: chance k/(n-1) divided by sqrt(ceiling) is " +
                 ", ".join(f"{sc.DS_SHORT[ds]} {x['floor']:.3f}" for ds, x in floor.items() if ds in sc.DS_SHORT) +
                 " in corrected units (MEG ceilings " + (f"{floor['things_meg']['ceil_min']:.3f}-{floor['things_meg']['ceil_max']:.3f}"
                 if "things_meg" in floor else "n/a") + f" vs chance {floor.get('things_meg', {}).get('chance', np.nan):.3f}). "
                 "MEG mutual k-NN cells therefore carry little information, and H6 passing there is close to trivial. "
                 f"With chance-adjusted mutual k-NN (exploratory): H1 {RV['mknn_chance']['H1']['detail'].split(' (')[0]}, "
                 f"H5 {RV['mknn_chance']['H5']['detail']}.")
    rf = []
    for ds in DATASETS:
        a = cell("reference", "REF_full_vs_groupmean", ds, "rsa")
        b = cell("reference", "REF_halves_vs_groupmean", ds, "rsa")
        c = cell("reference", "REF_halves_vs_single_subject", ds, "rsa")
        if a is not None:
            rf.append(f"{sc.DS_SHORT[ds]} {f(a['effect'])} / {f(b['effect'])} / {f(c['effect'])}")
    if rf:
        L.append("* **REF_brain_heldout is not like-for-like** (a subject's full kernel vs the others' mean; models are "
                 "scored against split halves). RSA reference full-vs-mean / halves-vs-mean / halves-vs-one-subject: "
                 + "; ".join(rf) + ". Compare models with the like-for-like columns in `robustness.md`.")
    L.append("* **H4**: arm (a) is confirmed; arm (b) (LM scale trend) rests on two families and is not confirmed under "
             "the cell rule, so H4 is reported as PARTIAL (not passed).")
    L.append("")
    return L


def write_markdown(H, V, path, datasets, settings, sens=None, rob=None):
    pre = dict(n_sub=N_SUB, n_null=N_NULL, n_perm=N_PERM)
    L = ["# Hypothesis tests (machine-generated by `python -m pbi stats`; do not edit)", ""]
    if settings != pre:
        L += [f"**WARNING: non-preregistered settings {settings} (preregistered: {pre}). Not a valid confirmatory run.**", ""]
    missing = [d for d in DATASETS if d not in datasets]
    if missing:
        L += [f"**INCOMPLETE: no scores for {', '.join(missing)}; their cells count as not confirmed.**", ""]
    L += ["Cell = dataset x metric. Confirmed in a cell: one-sided permutation p < 0.05 after Holm correction across",
          "the 9 cells AND 95% subsample CI excluding 0 (H6: upper CI < 1). CIs (stimulus sampling only: the model "
          "set is held fixed, so CIs of correlations across models do not include model-sampling uncertainty; "
          "model-bootstrap CIs are in `robustness.md`): "
          f"{settings['n_sub']} subsamples of 80% of stimuli; permutation tests: {settings['n_perm']} permutations; "
          f"shuffled-stimulus nulls: {settings['n_null']} permutations.", "",
          "## Verdicts (preregistered thresholds)", "", "| Hypothesis | Verdict | Detail |", "|---|---|---|"]
    names = {"H1": "Trained > random-init", "H2": "Alignment rises with capability (headline)",
             "H3": "Alignment rises with scale within vision families", "H4": "Language-only models above chance",
             "H5": "Model-model agreement predicts model-brain agreement", "H6": "No model reaches the noise ceiling"}
    for h in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        v = V[h]
        L.append(f"| {h} {names[h]} | **{v.get('label', 'PASS' if v['passed'] else 'FAIL')}** | {v['detail']} |")
    head = ("Brains and models converge as capability rises (H2 passed; subject to the reviewer's checks)."
            if V["H2"]["passed"] else
            "Models share structure with brains, but more capable models are not reliably more brain-like "
            "(H2 failed; H1 and H5 passed)." if V["H1"]["passed"] and V["H5"]["passed"] else
            "H2 failed and H1/H5 did not both pass: no convergence claim; see cells below.")
    L += ["", f"**Headline decision rule:** {head}", ""]
    if rob is not None:
        L += _reading_notes(H, V, rob)
    L += ["## Cells", "",
          "| Test | Dataset | Metric | Effect | 95% CI | p | p (Holm) | Confirmed | n |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in H.iterrows():
        if r["hypothesis"].startswith("H4_best:"):
            continue
        L.append(f"| {r['hypothesis']} | {sc.DS_SHORT.get(r['dataset'], r['dataset'])} | {r['metric']} | "
                 f"{_fmt(r['effect'])} | [{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | {_fmt(r['p'], 4)} | "
                 f"{_fmt(r['p_holm'], 4)} | {'yes' if r['confirmed'] else 'no'} | "
                 f"{'' if pd.isna(r['n']) else int(r['n'])} |")
    fam = H[H["hypothesis"].str.startswith("H4_best:")]
    if len(fam):
        L += ["", "## H4: best model per language family (shuffled-stimulus null, max over family)", "",
              "| Family | Dataset | Metric | Best score | p | Note |", "|---|---|---|---|---|---|"]
        for _, r in fam.iterrows():
            L.append(f"| {r['hypothesis'][8:]} | {sc.DS_SHORT.get(r['dataset'])} | {r['metric']} | "
                     f"{_fmt(r['effect'])} | {_fmt(r['p'], 4)} | {r['note']} |")
    L += ["", "Notes: H2_k* rows are the k-sensitivity check (mutual k-NN with k = 5, 10, 20). The primary k is "
          "round(n/20): k = 10 for EEG and MEG (n = 200) and **k = 5 for fMRI (n = 100)**, so the primary H2 row equals "
          "H2_k10 for EEG/MEG and H2_k5 for fMRI; the fMRI H2_k10 and H2_k20 rows are sensitivity values, not the "
          "primary analysis. "
          "H2_width_partial is the width-confound control. REF_brain_heldout is the brain-brain reference "
          "(not a test). LOSO = leave-one-subject-out ranking consistency (CI columns hold min/max over subjects). "
          "H4 is operationalized on the primary metric (mutual k-NN) because the threshold names no metric; "
          "the other metrics are reported in the table.", ""]
    if sens is not None:
        T, SV, desc = sens
        L += ["## EXPLORATORY sensitivity analyses (not preregistered; do not replace the verdicts above)", "",
              "Same tests, subsamples, permutations and Holm correction (9 cells) as the primary analysis, on a "
              "restricted set of models or subjects. Full rows: `artifacts/results/sensitivity.csv`.", ""]
        for a in SV:
            L += [f"### {a}: {desc[a]}", "", "| Hypothesis | Verdict (exploratory) | Detail |", "|---|---|---|"]
            for h, v in SV[a].items():
                L.append(f"| {h} | {'PASS' if v['passed'] else 'FAIL'} | {v['detail']} |")
            L += ["", "| Test | Dataset | Metric | Effect | 95% CI | p | p (Holm) | Confirmed | n |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for _, r in T[T["analysis"] == a].iterrows():
                if not r["hypothesis"] in ("H1", "H2", "H3", "H4_bestLM", "H4_scale", "H5", "H6"):
                    continue
                L.append(f"| {r['hypothesis']} | {sc.DS_SHORT.get(r['dataset'], r['dataset'])} | {r['metric']} | "
                         f"{_fmt(r['effect'])} | [{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}] | {_fmt(r['p'], 4)} | "
                         f"{_fmt(r['p_holm'], 4)} | {'yes' if r['confirmed'] else 'no'} | "
                         f"{'' if pd.isna(r['n']) else int(r['n'])} |")
            L.append("")
    Path(path).write_text("\n".join(L))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Preregistered hypothesis tests")
    ap.add_argument("--root", default=None)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--n-sub", type=int, default=N_SUB)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    a = ap.parse_args(argv)
    run(a.root, a.n_sub, a.n_null, a.n_perm, a.jobs)


if __name__ == "__main__":
    main()
