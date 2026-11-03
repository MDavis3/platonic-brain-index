"""Stimulus tables and image materialization for the THINGS test sets.

Builds artifacts/stimuli/{things_eeg2,things_meg,things_fmri}.csv with columns
idx,image,concept,path_rel. The row order (idx) is used by every kernel of that
dataset. Images are placed under ~/pbi_data/images/{dataset}/ and path_rel is
relative to that folder. Images are never committed (THINGS license: research only).

Sources
-------
things_eeg2 : ~/pbi_data/things_eeg2/test_images/<NNNNN_concept>/<file>.jpg, sorted by folder
              (this is the condition order of preprocessed_eeg_test.npy).
things_meg  : the 200 test images (trial_type == 'test') from OpenNeuro ds004212
              sourcedata/sample_attributes_P{1..4}.csv (identical set in all 4 participants),
              sorted by file name.
things_fmri : the 100 unique 'stimulus' values in sub-0X_StimulusMetadataTestset.csv, sorted by
              file name. They are a subset of the MEG test images.
MEG/fMRI image files are extracted from the THINGS database archive images_THINGS.zip on OSF
(project jum2f) with HTTP range requests (only the needed members are fetched). The archive
password is published openly by the THINGS authors in password_images.txt in the same project.

Concept normalization (concept_from_name):
  1. take the file stem (EEG2: the folder name with its leading "NNNNN_" index removed;
     MEG/fMRI: the image file name without extension and without the trailing exemplar
     suffix "_<digits><letter>", e.g. "_22s", "_14n", "_01b"),
  2. strip trailing digits (THINGS sense-disambiguation numbers, e.g. "bat1" -> "bat"),
  3. replace underscores with spaces and strip whitespace.

Usage: python -m pbi.stimuli [--datasets things_eeg2 things_meg things_fmri] [--no-images]
"""
from __future__ import annotations

import argparse
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PBI_DATA", Path.home() / "pbi_data")).expanduser()
OUT_DIR = REPO / "artifacts" / "stimuli"
IMG_ROOT = DATA / "images"

OPENNEURO = "https://s3.amazonaws.com/openneuro.org/ds004212/sourcedata/sample_attributes_P{p}.csv"
OSF_THINGS_ZIP = "https://osf.io/download/rdxy2/"      # images_THINGS.zip (jum2f)
OSF_PASSWORD_TXT = "https://osf.io/download/j6a3m/"    # password_images.txt (jum2f)

DATASETS = ("things_eeg2", "things_meg", "things_fmri")


def concept_from_name(name: str, eeg_folder: bool = False) -> str:
    s = Path(name).stem if not eeg_folder else name
    if eeg_folder:
        s = re.sub(r"^\d+_", "", s)
    else:
        s = re.sub(r"_\d+[a-z]?$", "", s)
    s = re.sub(r"\d+$", "", s)
    return s.replace("_", " ").strip()


# ----------------------------------------------------------------------------- lists
def eeg2_list() -> pd.DataFrame:
    root = DATA / "things_eeg2" / "test_images"
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        assert len(files) == 1, (folder, files)
        rows.append(dict(image=files[0].name, concept=concept_from_name(folder.name, eeg_folder=True),
                         src=str(files[0]), path_rel=f"{folder.name}/{files[0].name}"))
    return pd.DataFrame(rows)


def _meg_meta(p: int) -> Path:
    d = DATA / "things_meg" / "meta"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"sample_attributes_P{p}.csv"
    if not f.exists():
        urllib.request.urlretrieve(OPENNEURO.format(p=p), f)
    return f


def meg_list() -> pd.DataFrame:
    sets = []
    for p in (1, 2, 3, 4):
        d = pd.read_csv(_meg_meta(p))
        sets.append(set(d.loc[d.trial_type == "test", "image_path"].map(lambda s: s.split("/")[-1])))
    assert all(s == sets[0] for s in sets), "MEG test sets differ across participants"
    names = sorted(sets[0])
    assert len(names) == 200, len(names)
    return pd.DataFrame(dict(image=names, concept=[concept_from_name(n) for n in names]))


def fmri_list() -> pd.DataFrame:
    root = DATA / "things_fmri" / "testset" / "betas_csv_testset"
    sets = [set(pd.read_csv(f).stimulus) for f in sorted(root.glob("sub-0*_StimulusMetadataTestset.csv"))]
    assert sets and all(s == sets[0] for s in sets), "fMRI test sets differ across subjects"
    names = sorted(sets[0])
    assert len(names) == 100, len(names)
    return pd.DataFrame(dict(image=names, concept=[concept_from_name(n) for n in names]))


# ----------------------------------------------------------------------------- images
def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src)


def _things_password() -> bytes:
    txt = urllib.request.urlopen(OSF_PASSWORD_TXT, timeout=60).read().decode()
    m = re.search(r"Password for images_THINGS\.zip:\s*(\S+)", txt)
    assert m, "could not parse password_images.txt"
    return m.group(1).encode()


def fetch_things_images(names: list[str], dest: Path) -> list[str]:
    """Extract the named image files (by basename) from images_THINGS.zip into dest.
    Returns the names that could not be found."""
    dest.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (dest / n).exists()]
    if not todo:
        return []
    from remotezip import RemoteZip  # pip install remotezip

    pwd = _things_password()
    missing = []
    with RemoteZip(OSF_THINGS_ZIP) as z:
        by_base = {}
        for info in z.infolist():
            if not info.is_dir():
                by_base.setdefault(info.filename.rsplit("/", 1)[-1], info)
        for i, n in enumerate(todo):
            info = by_base.get(n)
            if info is None:
                missing.append(n)
                continue
            data = z.read(info, pwd=pwd)
            tmp = dest / (n + ".tmp")
            tmp.write_bytes(data)
            tmp.rename(dest / n)
            if (i + 1) % 25 == 0:
                print(f"  fetched {i + 1}/{len(todo)}", flush=True)
    return missing


def build(datasets=DATASETS, images: bool = True) -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    meg = meg_list() if ("things_meg" in datasets or "things_fmri" in datasets) else None
    meg_dir = IMG_ROOT / "things_meg"
    missing: set[str] = set()
    if images and meg is not None:
        need = list(meg.image)
        if "things_fmri" in datasets:
            need = sorted(set(need) | set(fmri_list().image))
        missing = set(fetch_things_images(need, meg_dir))
        if missing:
            print(f"WARNING: {len(missing)} images not found in THINGS archive: {sorted(missing)[:10]}")

    for ds in datasets:
        if ds == "things_eeg2":
            df = eeg2_list()
            if images:
                for src, rel in zip(df.src, df.path_rel):
                    _link(Path(src), IMG_ROOT / ds / rel)
            df = df.drop(columns="src")
        elif ds == "things_meg":
            df = meg.copy()
            df["path_rel"] = [n if (meg_dir / n).exists() else "" for n in df.image]
        elif ds == "things_fmri":
            df = fmri_list()
            rel = []
            for n in df.image:
                if (meg_dir / n).exists():
                    if images:
                        _link(meg_dir / n, IMG_ROOT / ds / n)
                    rel.append(n)
                else:
                    rel.append("")
            df["path_rel"] = rel
        else:
            raise ValueError(ds)
        df.insert(0, "idx", range(len(df)))
        df = df[["idx", "image", "concept", "path_rel"]]
        df.to_csv(OUT_DIR / f"{ds}.csv", index=False)
        out[ds] = df
    return out


def load(dataset: str) -> pd.DataFrame:
    """Read artifacts/stimuli/{dataset}.csv and add an absolute 'path' column."""
    df = pd.read_csv(OUT_DIR / f"{dataset}.csv", keep_default_na=False)
    df["path"] = [str(IMG_ROOT / dataset / r) if r else "" for r in df.path_rel]
    return df


def verify(datasets=DATASETS) -> None:
    from PIL import Image

    expected = dict(things_eeg2=200, things_meg=200, things_fmri=100)
    for ds in datasets:
        df = load(ds)
        bad = []
        for p in df.path:
            try:
                with Image.open(p) as im:
                    im.convert("RGB").load()
            except Exception as e:  # noqa: BLE001
                bad.append((p, repr(e)))
        print(f"{ds}: n={len(df)} (expected {expected[ds]}), unique images={df.image.nunique()}, "
              f"unique concepts={df.concept.nunique()}, unreadable={len(bad)}")
        for b in bad[:5]:
            print("   BAD", b)
        print(df[["idx", "image", "concept", "path_rel"]].iloc[[0, len(df) // 2, -1]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--no-images", action="store_true", help="only write CSVs (no download/linking)")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if not a.verify_only:
        build(a.datasets, images=not a.no_images)
    if not a.no_images:
        verify(a.datasets)


if __name__ == "__main__":
    main()
