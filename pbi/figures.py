"""Figures (all from artifacts/results/*.csv, so they regenerate on a clean clone).

  figures/map_of_minds.{png,svg}  every brain subject and every model in one 2-D embedding (exploratory)
  figures/scaling.png             corrected score vs capability, dataset x metric (H2)
  figures/family_scaling.png      corrected score vs log10 params within families (H3 / H4 scale)
  figures/timeresolved.png        time-resolved EEG/MEG alignment (exploratory; only if computed)

Colour: hue = training objective (Okabe-Ito, colourblind-safe), families within an objective are lighter /
darker shades of that hue. Marker = modality (models) or recording type (brains).

Map of minds. Each system's profile is its corrected mutual-k-NN alignment with every model: for a brain
subject, its subject-level corrected score with each model on its own dataset; for a model, its mutual k-NN
with each other model averaged over datasets (models are noise-free, so corrected = raw); the consensus point
is the mean kernel of all trained vision models, profiled the same way. Self-similarities and missing entries
are set to NaN, features are z-scored across systems (NaN -> 0 after z-scoring), the distance between two
systems is 1 - Pearson r of their profiles, and the 2-D layout is metric MDS (sklearn, random_state=0).
"""
from __future__ import annotations

import colorsys
import glob
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def _import_pyplot():
    """On some macOS versions `system_profiler SPFontsDataType` returns no '_items' and matplotlib's first
    font-cache build crashes (KeyError). Skip that call (standard font directories are still scanned)."""
    if sys.platform != "darwin":
        import matplotlib.pyplot as plt
        return plt
    orig = subprocess.check_output

    def guarded(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "system_profiler":
            raise OSError("system_profiler skipped")
        return orig(cmd, *a, **k)

    subprocess.check_output = guarded
    try:
        import matplotlib.pyplot as plt
    finally:
        subprocess.check_output = orig
    return plt


plt = _import_pyplot()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import kernels as kn  # noqa: E402
from . import score as sc  # noqa: E402

plt.rcParams.update({"svg.hashsalt": "pbi", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})
SAVE = dict(dpi=200, bbox_inches="tight", metadata={"Software": None})
SAVE_SVG = dict(bbox_inches="tight", metadata={"Date": None, "Creator": None})

OKABE = {"supervised": "#0072B2", "ssl": "#009E73", "clip": "#E69F00", "siglip": "#CC79A7",
         "language": "#D55E00", "audio": "#56B4E9", "random": "#8C8C8C", "other": "#5F4B8B"}
GROUP_LABEL = {"supervised": "supervised", "ssl": "self-supervised", "clip": "CLIP (image)",
               "siglip": "SigLIP (image)", "language": "language", "audio": "audio", "random": "random",
               "other": "other"}
MODALITY_MARKER = {"vision": "o", "language": "s", "audio": "^"}
BRAIN_MARKER = {"things_eeg2": "*", "things_meg": "D", "things_fmri": "P"}
BRAIN_COLOR = "#1A1A1A"


def group_of(meta, m):
    if m not in meta.index:
        return "other"
    obj, mod = meta.at[m, "objective"], meta.at[m, "modality"]
    if obj == "random":
        return "random"
    if mod == "language":
        return "language"
    if mod == "audio":
        return "audio"
    return obj if obj in OKABE else "other"


def _shade(hex_color, f):
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = min(0.9, max(0.15, l + f))
    return matplotlib.colors.to_hex(colorsys.hls_to_rgb(h, l, s))


SHADES = [0.0, -0.22, 0.22, -0.34, 0.32, -0.10, 0.12, 0.40]


def family_colors(meta, models):
    fams = {}
    for m in models:
        if m in meta.index:
            fams.setdefault(group_of(meta, m), [])
            f = meta.at[m, "family"]
            if f not in fams[group_of(meta, m)]:
                fams[group_of(meta, m)].append(f)
    col = {}
    for g, fl in fams.items():
        for i, f in enumerate(sorted(fl)):
            col[f] = _shade(OKABE[g], SHADES[i % len(SHADES)])
    return col


def _fig_dir(root):
    d = sc.art_root(root).parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _size(meta, m, base=18):
    p = meta.at[m, "params"] if m in meta.index else np.nan
    if not np.isfinite(p) or p <= 0:
        return base * 2
    return float(np.clip(base + 34 * (np.log10(p) - 6.5), 10, 260))


# ------------------------------------------------------------------ map of minds

def map_profiles(root=None):
    out = sc.results_dir(root)
    meta = sc.load_models(root)
    subj = pd.read_csv(out / "scores_subject.csv")
    mm = pd.read_csv(out / "model_similarity.csv")
    ref = pd.read_csv(out / "scores_reference.csv")
    feats = sorted(set(subj["model"]))
    rows, info = [], []
    s = subj[subj["metric"] == "mknn"]
    for (ds, sub), g in s.groupby(["dataset", "subject"], sort=False):
        v = g.set_index("model")["corrected"].reindex(feats)
        rows.append(v.to_numpy(float))
        info.append(dict(system=f"{sc.DS_SHORT[ds]} {sub.replace('sub-', 's')}", kind="brain", dataset=ds))
    models = sorted(set(mm["model_a"]) - {"consensus"})
    P = {}
    for ds, g in mm.groupby("dataset", sort=False):
        P[ds] = g.pivot(index="model_a", columns="model_b", values="mknn")
    for m in models + ["consensus"]:
        vs = []
        for ds, W in P.items():
            if m in W.index:
                vs.append(W.loc[m].reindex(feats).to_numpy(float))
        v = np.nanmean(np.stack(vs), 0) if vs else np.full(len(feats), np.nan)
        if m in feats:
            v[feats.index(m)] = np.nan
        rows.append(v)
        info.append(dict(system=m, kind="consensus" if m == "consensus" else "model", dataset=""))
    X = np.array(rows)
    return X, pd.DataFrame(info), feats, meta, ref


def map_embedding(X):
    from sklearn.manifold import MDS
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    sd[~(sd > 0)] = 1.0
    Z = (X - mu) / sd
    Z = np.where(np.isfinite(Z), Z, 0.0)
    C = np.corrcoef(Z)
    C = np.where(np.isfinite(C), C, 0.0)
    D = np.clip(1 - C, 0, 2)
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    try:
        mds = MDS(n_components=2, metric_mds=True, metric="precomputed", random_state=0, n_init=4,
                  init="random", max_iter=1000, normalized_stress="auto")
    except TypeError:  # older sklearn
        mds = MDS(n_components=2, metric=True, dissimilarity="precomputed", random_state=0, n_init=4,
                  max_iter=1000)
    Y = mds.fit_transform(D)
    # fix sign/rotation deterministically: first axis = principal axis, consensus side positive
    Y = Y - Y.mean(0)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    Y = Y @ Vt.T
    return Y, D


def fig_map(root=None, highlight=None):
    X, info, feats, meta, ref = map_profiles(root)
    if len(X) < 4:
        print("[figures] map_of_minds: too few systems; skipped")
        return
    Y, _ = map_embedding(X)
    ci = info.index[info["kind"] == "consensus"]
    if len(ci):
        for a in range(2):
            if Y[ci[0], a] < 0:
                Y[:, a] *= -1
    models = info[info["kind"] == "model"]["system"].tolist()
    fcol = family_colors(meta, models)
    fig, ax = plt.subplots(figsize=(11, 8))
    for i, r in info.iterrows():
        x, y = Y[i]
        if r["kind"] == "brain":
            ax.scatter(x, y, marker=BRAIN_MARKER[r["dataset"]], s=150 if r["dataset"] == "things_eeg2" else 80,
                       c=BRAIN_COLOR, edgecolors="white", linewidths=0.6, zorder=4)
            ax.annotate(r["system"], (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7, color=BRAIN_COLOR)
        elif r["kind"] == "consensus":
            ax.scatter(x, y, marker="o", s=380, facecolors="none", edgecolors="black", linewidths=2, zorder=5)
            ax.scatter(x, y, marker="+", s=120, c="black", zorder=5)
            ax.annotate("consensus\n(mean trained-vision kernel)", (x, y), xytext=(10, -14),
                        textcoords="offset points", fontsize=8, fontweight="bold")
        else:
            m = r["system"]
            mod = meta.at[m, "modality"] if m in meta.index else "vision"
            mk = "X" if sc.is_random(meta, m) else MODALITY_MARKER.get(mod, "o")
            fam = meta.at[m, "family"] if m in meta.index else m
            ax.scatter(x, y, marker=mk, s=_size(meta, m), c=fcol.get(fam, OKABE["other"]),
                       edgecolors="#333333", linewidths=0.4, alpha=0.9, zorder=3)
    for m in highlight or []:
        hit = info.index[info["system"] == m]
        if len(hit):
            ax.annotate(m, Y[hit[0]], xytext=(5, -9), textcoords="offset points", fontsize=7.5, color="#222222",
                        arrowprops=None)
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title("Map of minds: brains and models placed by their alignment profile across all models\n"
                 "(distance = 1 - correlation of z-scored corrected mutual-k-NN profiles; exploratory: brain rows are\n"
                 "ceiling-corrected brain-model scores, model rows are raw model-model scores, so positions are not like-for-like)", fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    # legends: families (colour), markers
    fam_handles = []
    by_group = {}
    for m in models:
        if m in meta.index:
            by_group.setdefault(group_of(meta, m), set()).add(meta.at[m, "family"])
    for g in OKABE:
        for f in sorted(by_group.get(g, [])):
            fam_handles.append(Line2D([], [], marker="o", ls="", markersize=7, markerfacecolor=fcol[f],
                                      markeredgecolor="#333333", label=f"{f} ({GROUP_LABEL[g]})"))
    l1 = ax.legend(handles=fam_handles, title="family (colour)", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   fontsize=7, title_fontsize=8, ncol=1 if len(fam_handles) <= 22 else 2, frameon=False)
    ax.add_artist(l1)
    mk_handles = [Line2D([], [], marker=MODALITY_MARKER[k], ls="", color="#666666", label=f"model: {k}")
                  for k in MODALITY_MARKER]
    mk_handles.append(Line2D([], [], marker="X", ls="", color=OKABE["random"], label="model: random-init"))
    mk_handles += [Line2D([], [], marker=BRAIN_MARKER[d], ls="", color=BRAIN_COLOR, markersize=9 if d == "things_eeg2" else 7,
                          label=f"brain: {sc.DS_SHORT[d]} subject") for d in BRAIN_MARKER]
    mk_handles.append(Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black", markersize=11,
                             label="consensus"))
    ax.legend(handles=mk_handles, title="marker (size = log params)", loc="lower left",
              bbox_to_anchor=(1.01, 0.0), fontsize=7, title_fontsize=8, frameon=False)
    d = _fig_dir(root)
    fig.savefig(d / "map_of_minds.png", bbox_extra_artists=[l1], **SAVE)
    fig.savefig(d / "map_of_minds.svg", bbox_extra_artists=[l1], **SAVE_SVG)
    plt.close(fig)
    coords = info.assign(x=Y[:, 0], y=Y[:, 1])
    coords.to_csv(sc.results_dir(root) / "map_of_minds.csv", index=False, float_format="%.4f")


# ------------------------------------------------------------------ scaling

def _hyp(root):
    p = sc.results_dir(root) / "hypotheses.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["hypothesis", "dataset", "metric"])


def _title(H, h, ds, metric, sym="ρ"):
    r = H[(H["hypothesis"] == h) & (H["dataset"] == ds) & (H["metric"] == metric)]
    if not len(r) or pd.isna(r["effect"].iloc[0]):
        return f"{sc.DS_SHORT[ds]} · {metric}"
    r = r.iloc[0]
    s = f"{sc.DS_SHORT[ds]} · {metric}: {sym}={r['effect']:.2f}"
    if pd.notna(r.get("ci_lo")):
        s += f" [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
    if str(r.get("confirmed")).lower() == "true":
        s += " ✓"
    return s


def fig_scaling(root=None):
    out = sc.results_dir(root)
    meta = sc.load_models(root)
    S = pd.read_csv(out / "scores.csv")
    R = pd.read_csv(out / "scores_reference.csv")
    H = _hyp(root)
    fcol = family_colors(meta, S["model"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    for i, ds in enumerate(sc.DATASETS):
        for j, metric in enumerate(sc.METRICS):
            ax = axes[i, j]
            g = S[(S["dataset"] == ds) & (S["metric"] == metric)].set_index("model")
            if not len(g):
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="#888888")
                ax.set_title(f"{sc.DS_SHORT[ds]} · {metric}")
                continue
            rnd = [m for m in g.index if sc.is_random(meta, m)]
            if rnd:
                v = g.loc[rnd, "corrected"].dropna()
                if len(v):
                    ax.axhspan(v.min(), v.max() if v.max() > v.min() else v.min() + 1e-3, color="#BBBBBB",
                               alpha=0.45, lw=0, label="random-init range")
            bb = R[(R["dataset"] == ds) & (R["metric"] == metric) & (R["system"] == "brain_heldout")
                   & R["included"].astype(bool)]["corrected"]
            if len(bb):
                ax.axhline(bb.mean(), color=BRAIN_COLOR, ls=":", lw=1.2, label="brain held-out subject")
            ax.axhline(0, color="#DDDDDD", lw=0.8, zorder=0)
            tv = [m for m in g.index if sc.is_trained_vision(meta, m) and np.isfinite(meta.at[m, "capability"])]
            for m in tv:
                ax.scatter(meta.at[m, "capability"], g.at[m, "corrected"], s=_size(meta, m, 14),
                           c=fcol.get(meta.at[m, "family"]), edgecolors="#333333", linewidths=0.4, zorder=3)
            ax.set_title(_title(H, "H2", ds, metric), fontsize=9)
            if i == 2:
                ax.set_xlabel("capability (5-shot accuracy)")
            if j == 0:
                ax.set_ylabel("corrected score")
    fams = sorted({meta.at[m, "family"] for m in S["model"].unique()
                   if sc.is_trained_vision(meta, m) and np.isfinite(meta.at[m, "capability"])})
    hs = [Line2D([], [], marker="o", ls="", markerfacecolor=fcol[f], markeredgecolor="#333333", label=f)
          for f in fams]
    hs += [matplotlib.patches.Patch(color="#BBBBBB", alpha=0.45, label="random-init range"),
           Line2D([], [], color=BRAIN_COLOR, ls=":", label="brain held-out subject")]
    fig.legend(handles=hs, loc="lower center", ncol=min(8, len(hs)), fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("H2: noise-ceiling-corrected alignment vs capability (trained vision models)\n"
                 "title: Spearman ρ [95% subsample CI], ✓ = confirmed (Holm p < .05 and CI > 0)", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(_fig_dir(root) / "scaling.png", **SAVE)
    plt.close(fig)


def fig_family_scaling(root=None):
    out = sc.results_dir(root)
    meta = sc.load_models(root)
    S = pd.read_csv(out / "scores.csv")
    F = pd.read_csv(out / "family_scaling.csv") if (out / "family_scaling.csv").exists() else pd.DataFrame()
    H = _hyp(root)
    fcol = family_colors(meta, S["model"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    used = {}
    for i, ds in enumerate(sc.DATASETS):
        for j, metric in enumerate(sc.METRICS):
            ax = axes[i, j]
            g = S[(S["dataset"] == ds) & (S["metric"] == metric)].set_index("model")
            fams = F[(F["dataset"] == ds) & (F["metric"] == metric)] if len(F) else pd.DataFrame()
            if not len(fams):
                ax.text(0.5, 0.5, "no family with ≥3 sizes", ha="center", va="center", transform=ax.transAxes,
                        color="#888888")
            for _, fr in fams.iterrows():
                ms = [m for m in g.index if m in meta.index and meta.at[m, "family"] == fr["family"]
                      and not sc.is_random(meta, m) and np.isfinite(meta.at[m, "params"])]
                ms = sorted(ms, key=lambda m: meta.at[m, "params"])
                x = np.log10(meta.loc[ms, "params"].to_numpy(float))
                ls = "-" if fr["modality"] == "vision" else "--"
                ax.plot(x, g.loc[ms, "corrected"], ls, marker="o" if ls == "-" else "s", ms=4,
                        color=fcol.get(fr["family"]), lw=1.4)
                used[fr["family"]] = ls
            ax.axhline(0, color="#DDDDDD", lw=0.8, zorder=0)
            t = _title(H, "H3", ds, metric, "mean ρ (vision)")
            ax.set_title(t, fontsize=8.5)
            if i == 2:
                ax.set_xlabel("log10 parameters")
            if j == 0:
                ax.set_ylabel("corrected score")
    hs = [Line2D([], [], color=fcol.get(f), ls=ls, marker="o" if ls == "-" else "s", ms=4, label=f)
          for f, ls in sorted(used.items())]
    if hs:
        fig.legend(handles=hs, loc="lower center", ncol=min(8, len(hs)), fontsize=7.5, frameon=False,
                   bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("H3 / H4: alignment vs scale within families (solid = vision, dashed = language)", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(_fig_dir(root) / "family_scaling.png", **SAVE)
    plt.close(fig)


# ------------------------------------------------------------------ time-resolved (exploratory)

def compute_timeresolved(root=None):
    """From ~/pbi_data/cache/timeresolved/{ds}/sub-XX/win{t0}.npz (written by pbi.brains): uncorrected
    mutual k-NN and RSA of a few reference systems against each 20 ms window. Skipped when the cache is absent."""
    data = Path(os.environ.get("PBI_DATA", Path.home() / "pbi_data"))
    base = data / "cache" / "timeresolved"
    if not base.exists():
        return None
    root = sc.art_root(root)
    out = sc.results_dir(root)
    meta = sc.load_models(root)
    S = pd.read_csv(out / "scores.csv")
    rows = []
    for ds in ("things_eeg2", "things_meg"):
        subs = sorted(glob.glob(str(base / ds / "sub-*")))
        if not subs:
            continue
        g = S[(S["dataset"] == ds)].groupby("model")["corrected"].mean()
        pick = {}
        tvs = [m for m in g.index if sc.is_trained_vision(meta, m)]
        lms = [m for m in g.index if m in meta.index and meta.at[m, "modality"] == "language"]
        rnd = [m for m in g.index if sc.is_random(meta, m)]
        if tvs:
            pick["best vision: " + g[tvs].idxmax()] = [g[tvs].idxmax()]
            pick["consensus (trained vision)"] = tvs
        if lms:
            pick["best language: " + g[lms].idxmax()] = [g[lms].idxmax()]
        if rnd:
            pick["random-init: " + rnd[0]] = [rnd[0]]
        kd = root / "kernels" / ds / "model"
        Ks = {lab: kn.mean_kernel([sc.load_kernel(kd / f"{m}.npz") for m in ms]) for lab, ms in pick.items()}
        if not Ks:
            continue
        labs = list(Ks)
        FM = sc.Feats([Ks[l] for l in labs], exact=True, metrics=("mknn", "rsa"))
        for sdir in subs:
            for f in sorted(glob.glob(os.path.join(sdir, "win*.npz"))):
                t0 = int(os.path.basename(f)[3:-4])
                K = sc.load_kernel(f)
                if len(K["cos"]) != FM.n:
                    continue
                FB = sc.Feats([K], exact=True, metrics=("mknn", "rsa"))
                for metric in ("mknn", "rsa"):
                    v = FM.pair(np.arange(len(labs)), FB, [0], metric)[:, 0]
                    for l, x in zip(labs, v):
                        rows.append(dict(dataset=ds, subject=os.path.basename(sdir), system=l, metric=metric,
                                         t0_ms=t0, score=x))
    if not rows:
        return None
    T = pd.DataFrame(rows)
    T.to_csv(out / "timeresolved.csv", index=False, float_format="%.5g")
    return T


def fig_timeresolved(root=None):
    p = sc.results_dir(root) / "timeresolved.csv"
    T = compute_timeresolved(root)
    if T is None:
        if not p.exists():
            print("[figures] timeresolved: no time-resolved data; skipped")
            return
        T = pd.read_csv(p)
    dss = [d for d in ("things_eeg2", "things_meg") if d in set(T["dataset"])]
    fig, axes = plt.subplots(len(dss), 2, figsize=(11, 3.4 * len(dss)), squeeze=False)
    pal = ["#0072B2", "#E69F00", "#D55E00", "#8C8C8C", "#009E73"]
    for i, ds in enumerate(dss):
        for j, metric in enumerate(("mknn", "rsa")):
            ax = axes[i, j]
            g = T[(T["dataset"] == ds) & (T["metric"] == metric)]
            for c, (lab, h) in zip(pal, g.groupby("system", sort=True)):
                a = h.groupby("t0_ms")["score"].agg(["mean", "sem", "count"]).reset_index()
                t = a["t0_ms"] + 10
                ax.plot(t, a["mean"], color=c, lw=1.5, label=lab)
                ax.fill_between(t, a["mean"] - a["sem"].fillna(0), a["mean"] + a["sem"].fillna(0), color=c,
                                alpha=0.2, lw=0)
            ax.axvline(0, color="#999999", lw=0.8)
            ax.axhline(0, color="#DDDDDD", lw=0.8)
            ax.axvspan(80, 500, color="#F3F3F3", zorder=0, label="primary window")
            ax.set_title(f"{sc.DS_SHORT[ds]} · {metric} (uncorrected, mean ± s.e.m. over subjects)", fontsize=9)
            ax.set_xlabel("time (ms, 20 ms windows)")
            ax.set_ylabel(metric)
            if i == 0 and j == 1:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Time-resolved alignment (exploratory)", fontsize=11)
    fig.tight_layout()
    fig.savefig(_fig_dir(root) / "timeresolved.png", **SAVE)
    plt.close(fig)


def make_all(root=None):
    out = sc.results_dir(root)
    meta = sc.load_models(root)
    hl = []
    lb = out / "leaderboard.csv"
    if lb.exists():
        L = pd.read_csv(lb)
        L = L[L["kind"] == "model"]
        hl += L["model"].head(4).tolist()
        for mod in ("language", "audio"):
            x = L[L["modality"] == mod]
            if len(x):
                hl.append(x["model"].iloc[0])
        r = pd.read_csv(lb)
        r = r[r["kind"] == "random-init"]
        if len(r):
            hl.append(r["model"].iloc[0])
    for fn, args in ((fig_map, (root, hl)), (fig_scaling, (root,)), (fig_family_scaling, (root,)),
                     (fig_timeresolved, (root,))):
        try:
            fn(*args)
        except FileNotFoundError as e:
            print(f"[figures] {fn.__name__}: missing input ({e}); skipped")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Make all figures")
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    make_all(a.root)


if __name__ == "__main__":
    main()
