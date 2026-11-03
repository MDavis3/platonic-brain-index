"""Brain representations, split-half kernels and noise ceilings (PREREGISTRATION.md, "Data" + "Noise ceiling").

Per dataset and subject we build a trial array  T = (n_stim, n_reps, n_features)  in the stimulus order of
artifacts/stimuli/{dataset}.csv (rows matched by image filename), then write

  artifacts/kernels/{dataset}/brain/sub-XX_full.npz          kernels of the mean over all reps
  artifacts/kernels/{dataset}/brain/sub-XX_split{s}_{A,B}.npz kernels of the mean over each half of the reps

and a row in artifacts/brains.csv with the split-half ceilings ceil_M = mean_s M(A_s, B_s).

Primary brain representations
  things_eeg2 : 17 posterior channels x samples with 80 <= t < 500 ms (100 Hz -> 42 samples), flattened.
  things_meg  : all 271 MEG sensors (picks='meg'), test trials only, FFT-resampled 200 -> 50 Hz
                (mne.filter.resample, anti-aliased) on the full epoch, then 80 <= t < 500 ms (21 samples: 80, 100, ..., 480 ms), flattened.
  things_fmri : single-trial betas of the released IT voxels; voxels with any NaN dropped.

Splits: s = 0..4, rng = np.random.default_rng(1000 + s).
  EEG (80 reps for every image): one permutation of the 80 repetition indices, shared by all images; A = first 40.
  MEG / fMRI: each image's repetitions are permuted independently (images visited in idx order, same rng);
              A = first n_i // 2, B = the rest (so unequal rep counts are handled).

Time-resolved (exploratory) kernels: `timeresolved(dataset, subject)` writes kernels of the all-rep mean for
20 ms non-overlapping windows [t0, t0+20) ms, t0 = -100, -80, ..., 780, to
~/pbi_data/cache/timeresolved/{dataset}/sub-XX/win{t0}.npz (same packing as everything else; not committed).
Within a window the features are all channels x native-rate samples (EEG 100 Hz: 2 samples; MEG 200 Hz: 4 samples).
To compute a time-resolved model-brain curve: for each t0, unpack(np.load(.../win{t0}.npz)) and
pbi.kernels.score(model_K, brain_K, metric).  `load_trials(dataset, subject)` returns the raw-ish trial array,
and `trials_timeresolved(dataset, subject)` the (n_stim, n_reps, n_ch, n_t) array plus times in ms, if you
want other windows.

CLI:  python -m pbi.brains --dataset {things_eeg2,things_meg,things_fmri,all} [--no-timeresolved]
"""
import argparse
import glob
import os
import re
import zipfile

import numpy as np
import pandas as pd

from pbi.kernels import kernels_from_features, pack, unpack, score, default_k, METRICS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser(os.environ.get("PBI_DATA", "~/pbi_data"))
CACHE = os.path.join(DATA, "cache")
ART = os.path.join(REPO, "artifacts")
N_SPLITS = 5
WIN = (80, 500)  # ms, inclusive start, exclusive end
TR_WINDOWS = list(range(-100, 800, 20))
DATASETS = ("things_eeg2", "things_meg", "things_fmri")


def stimuli(dataset):
    return pd.read_csv(os.path.join(ART, "stimuli", f"{dataset}.csv"))


def _order(dataset, names):
    """Map a list of image filenames (one per stimulus, any order) to stimulus-table idx."""
    st = stimuli(dataset)
    pos = {im: i for im, i in zip(st.image, st.idx)}
    idx = np.array([pos[os.path.basename(n)] for n in names])
    assert sorted(idx) == list(range(len(st))), f"{dataset}: stimulus set mismatch"
    return idx


# ------------------------------------------------------------------ subjects

def subjects(dataset):
    """Subjects whose raw data are completely present (rerunnable while downloads are in progress)."""
    if dataset == "things_eeg2":
        return [int(re.search(r"sub-(\d+)\.zip$", p).group(1))
                for p in sorted(glob.glob(os.path.join(DATA, "things_eeg2/dl/sub-??.zip")))]
    if dataset == "things_meg":
        out = []
        for s in range(1, 5):
            if _meg_complete(s):
                out.append(s)
        return out
    if dataset == "things_fmri":
        return [int(re.search(r"sub-(\d+)_", p).group(1)) for p in
                sorted(glob.glob(os.path.join(DATA, "things_fmri/testset/betas_csv_testset/sub-??_TestResponsesIT.npy")))]
    raise ValueError(dataset)


def _meg_complete(s):
    """True if the split fif chain of participant s exists (no .part) and is fully readable."""
    first = os.path.join(DATA, f"things_meg/preprocessed_P{s}-epo.fif")
    if not os.path.exists(first):
        return False
    if glob.glob(os.path.join(DATA, f"things_meg/preprocessed_P{s}-epo*.part")):
        return False
    try:
        import mne
        ep = mne.read_epochs(first, preload=False, verbose="error")
        return len(ep) == 27048
    except Exception:
        return False


# ------------------------------------------------------------------ raw loaders
# Each returns (T, times_ms, ch_names) with T = (n_stim, n_reps, n_ch, n_t) in stimulus idx order
# (fMRI: n_ch = voxels, n_t = 1, times None).  Trials of one stimulus are in presentation order.

def _eeg_raw(s):
    z = zipfile.ZipFile(os.path.join(DATA, f"things_eeg2/dl/sub-{s:02d}.zip"))
    with z.open(f"sub-{s:02d}/preprocessed_eeg_test.npy") as f:
        d = np.load(f, allow_pickle=True).item()
    X = d["preprocessed_eeg_data"]  # (200 cond, 80 reps, 17 ch, 100 t); cond order = sorted test_images folders
    folders = sorted(os.listdir(os.path.join(DATA, "things_eeg2/test_images")))
    folders = [f for f in folders if os.path.isdir(os.path.join(DATA, "things_eeg2/test_images", f))]
    names = []
    for fo in folders:
        files = [f for f in os.listdir(os.path.join(DATA, "things_eeg2/test_images", fo)) if not f.startswith(".")]
        assert len(files) == 1, fo
        names.append(files[0])
    assert len(names) == X.shape[0] == 200
    idx = _order("things_eeg2", names)
    T = np.empty_like(X, dtype=np.float32)
    T[idx] = X
    times = np.round(np.asarray(d["times"]) * 1000).astype(int)
    return T, times, list(d["ch_names"])


def _meg_cache_path(s):
    return os.path.join(CACHE, "brains", "things_meg", f"P{s}_test_200hz.npz")


def _meg_raw(s):
    """Test trials of THINGS-MEG participant s at native 200 Hz, full epoch (-100..1300 ms). Cached as float32."""
    cp = _meg_cache_path(s)
    if os.path.exists(cp):
        d = np.load(cp, allow_pickle=True)
        return d["T"], d["times"], list(d["ch_names"])
    import mne
    meta = pd.read_csv(os.path.join(DATA, f"things_meg/meta/sample_attributes_P{s}.csv"))
    ep = mne.read_epochs(os.path.join(DATA, f"things_meg/preprocessed_P{s}-epo.fif"), preload=False, verbose="error")
    assert len(ep) == len(meta) == 27048, (len(ep), len(meta))
    # trial -> image: epochs are in the same order as sample_attributes rows; check with the event codes
    # (event value = things_image_nr in the released preprocessing) where available.
    ev = ep.events[:, 2]
    tin = meta.things_image_nr.to_numpy()
    test = np.where(meta.trial_type.to_numpy() == "test")[0]
    # (catch trials carry event code 999999 and no image number; every test trial must match)
    agree = np.mean(ev[test] == tin[test])
    print(f"  P{s}: event code == things_image_nr for {agree:.3f} of test epochs")
    assert agree == 1.0, "MEG epoch order does not match sample_attributes"
    picks = mne.pick_types(ep.info, meg=True, ref_meg=False)
    chunks = []
    for i in range(0, len(test), 300):
        chunks.append(ep[test[i:i + 300]].get_data(picks=picks).astype(np.float32))
    D = np.concatenate(chunks)  # (2400, 271, 281)
    times = np.round(ep.times * 1000).astype(int)
    names = [os.path.basename(p) for p in meta.image_path.to_numpy()[test]]
    st = stimuli("things_meg")
    pos = {im: i for im, i in zip(st.image, st.idx)}
    which = np.array([pos[n] for n in names])
    counts = np.bincount(which, minlength=len(st))
    assert counts.min() > 0, "missing test images"
    R = counts.max()
    T = np.full((len(st), R) + D.shape[1:], np.nan, np.float32)
    fill = np.zeros(len(st), int)
    for t, i in enumerate(which):  # presentation order
        T[i, fill[i]] = D[t]
        fill[i] += 1
    ch = [ep.ch_names[p] for p in picks]
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    np.savez(cp, T=T, times=times, ch_names=np.array(ch), counts=counts)
    del ep, D
    return T, times, ch


def _fmri_raw(s):
    base = os.path.join(DATA, "things_fmri/testset/betas_csv_testset")
    X = np.load(os.path.join(base, f"sub-{s:02d}_TestResponsesIT.npy")).astype(np.float32)  # (1200, voxels)
    meta = pd.read_csv(os.path.join(base, f"sub-{s:02d}_StimulusMetadataTestset.csv"))
    keep = ~np.isnan(X).any(0)
    X = X[:, keep]
    st = stimuli("things_fmri")
    pos = {im: i for im, i in zip(st.image, st.idx)}
    which = np.array([pos[n] for n in meta.stimulus])
    counts = np.bincount(which, minlength=len(st))
    R = counts.max()
    T = np.full((len(st), R, X.shape[1], 1), np.nan, np.float32)
    fill = np.zeros(len(st), int)
    for t, i in enumerate(which):
        T[i, fill[i], :, 0] = X[t]
        fill[i] += 1
    return T, None, [f"vox{j}" for j in np.where(keep)[0]]


def trials_timeresolved(dataset, s):
    return {"things_eeg2": _eeg_raw, "things_meg": _meg_raw, "things_fmri": _fmri_raw}[dataset](s)


def load_trials(dataset, s):
    """Primary representation: (n_stim, n_reps, n_features) float32 (NaN rows = missing reps), plus description."""
    T, times, ch = trials_timeresolved(dataset, s)
    if dataset == "things_eeg2":
        m = (times >= WIN[0]) & (times < WIN[1])
        F = T[..., m]
        desc = f"{len(ch)} ch x {m.sum()} samples (100 Hz) {WIN[0]}-{WIN[1]} ms"
    elif dataset == "things_meg":
        import mne
        # FFT resampling 200 -> 50 Hz on the full -100..1300 ms epoch (anti-aliased), then crop
        n_s, R = T.shape[:2]
        flat = T.reshape(n_s * R, *T.shape[2:])
        valid = ~np.isnan(flat).any((1, 2))
        out = np.full(flat.shape[:2] + (int(round(flat.shape[2] / 4)),), np.nan, np.float32)
        out[valid] = mne.filter.resample(flat[valid].astype(np.float64), down=4.0, npad="auto",
                                         verbose="error").astype(np.float32)
        t50 = times[0] + 20 * np.arange(out.shape[-1])
        m = (t50 >= WIN[0]) & (t50 < WIN[1])
        F = out[..., m].reshape(n_s, R, len(ch), m.sum())
        desc = f"{len(ch)} MEG sensors x {m.sum()} samples (resampled to 50 Hz) {WIN[0]}-{WIN[1]} ms"
    else:
        F = T
        desc = f"{len(ch)} IT voxels (released IT mask, NaN voxels dropped), single-trial betas"
    return F.reshape(F.shape[0], F.shape[1], -1), desc


# ------------------------------------------------------------------ splits / kernels

def split_masks(counts, split, shared):
    """Boolean masks A, B of shape (n_stim, max_reps) for half-split `split`."""
    rng = np.random.default_rng(1000 + split)
    n, R = len(counts), int(counts.max())
    A = np.zeros((n, R), bool)
    B = np.zeros((n, R), bool)
    if shared:
        assert (counts == R).all()
        p = rng.permutation(R)
        A[:, p[: R // 2]] = True
        B[:, p[R // 2:]] = True
    else:
        for i in range(n):
            p = rng.permutation(int(counts[i]))
            A[i, p[: counts[i] // 2]] = True
            B[i, p[counts[i] // 2:]] = True
    return A, B


def _masked_mean(F, mask):
    w = mask[..., None].astype(np.float64)
    return (np.nan_to_num(F.astype(np.float64)) * w).sum(1) / w.sum(1)


def _save(path, K):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **pack(K))


def process_subject(dataset, s, do_tr=True):
    F, desc = load_trials(dataset, s)
    valid = ~np.isnan(F).any(2)
    counts = valid.sum(1)
    out = os.path.join(ART, "kernels", dataset, "brain")
    Kfull = kernels_from_features(_masked_mean(F, valid))
    _save(os.path.join(out, f"sub-{s:02d}_full.npz"), Kfull)
    n = F.shape[0]
    ceil = {m: [] for m in METRICS}
    for sp in range(N_SPLITS):
        A, B = split_masks(counts, sp, shared=(dataset == "things_eeg2"))
        A &= valid
        B &= valid
        KA = kernels_from_features(_masked_mean(F, A))
        KB = kernels_from_features(_masked_mean(F, B))
        _save(os.path.join(out, f"sub-{s:02d}_split{sp}_A.npz"), KA)
        _save(os.path.join(out, f"sub-{s:02d}_split{sp}_B.npz"), KB)
        # ceilings computed from the stored (float32-packed) kernels, exactly what the benchmark will load
        KA, KB = unpack(pack(KA)), unpack(pack(KB))
        for m in METRICS:
            ceil[m].append(score(KA, KB, m, default_k(n)))
    row = dict(dataset=dataset, subject=f"sub-{s:02d}", n_stim=n,
               n_reps=int(np.median(counts)) if counts.min() == counts.max() else f"{counts.min()}-{counts.max()}",
               n_features=F.shape[2], feature_desc=desc,
               **{f"ceil_{m}": float(np.mean(v)) for m, v in ceil.items()})
    print(f"  {dataset} sub-{s:02d}: n={n} reps={row['n_reps']} feat={F.shape[2]} "
          + " ".join(f"{m}={row['ceil_' + m]:.3f}(sd {np.std(ceil[m]):.3f})" for m in METRICS), flush=True)
    del F
    if do_tr and dataset in ("things_eeg2", "things_meg"):
        timeresolved(dataset, s)
    return row


def timeresolved(dataset, s):
    """Exploratory: kernels of the all-rep mean in 20 ms windows [t0, t0+20) ms, t0 = -100..780, native rate."""
    T, times, ch = trials_timeresolved(dataset, s)
    valid = ~np.isnan(T).any((2, 3))
    M = _masked_mean(T.reshape(T.shape[0], T.shape[1], -1), valid).reshape(T.shape[0], *T.shape[2:])
    out = os.path.join(CACHE, "timeresolved", dataset, f"sub-{s:02d}")
    for t0 in TR_WINDOWS:
        m = (times >= t0) & (times < t0 + 20)
        if m.sum() == 0:
            continue
        _save(os.path.join(out, f"win{t0}.npz"), kernels_from_features(M[..., m].reshape(len(M), -1)))


def update_csv(rows, dataset):
    path = os.path.join(ART, "brains.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[~((old.dataset == dataset) & old.subject.isin(new.subject))]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["dataset", "subject"]).reset_index(drop=True)
    new.to_csv(path, index=False, float_format="%.5f")
    return new


def run(dataset, do_tr=True):
    subs = subjects(dataset)
    print(f"{dataset}: subjects available {subs}", flush=True)
    rows = []
    for s in subs:
        rows.append(process_subject(dataset, s, do_tr))
        update_csv(rows, dataset)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=list(DATASETS) + ["all"])
    ap.add_argument("--subjects", type=int, nargs="*", help="restrict to these subject numbers")
    ap.add_argument("--no-timeresolved", action="store_true")
    a = ap.parse_args()
    for d in (DATASETS if a.dataset == "all" else [a.dataset]):
        if a.subjects:
            rows = [process_subject(d, s, not a.no_timeresolved) for s in a.subjects]
            update_csv(rows, d)
        else:
            run(d, not a.no_timeresolved)


if __name__ == "__main__":
    main()
