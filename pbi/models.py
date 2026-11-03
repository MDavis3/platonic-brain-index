"""Vision model zoo: registry, feature extraction, model kernels.

Primary embedding = final pooled pre-logit embedding:
  timm        -> create_model(..., num_classes=0); forward() output
  open_clip   -> encode_image() output (projection, unnormalized)
  torchvision -> AlexNet with the last Linear removed (fc7 after ReLU, 4096-d)
Each model uses its own preprocessing (timm resolve_data_config / open_clip preprocess).

CLI:  python -m pbi.models --dataset all --models all [--force] [--layers]
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pbi.kernels import kernels_from_features, pack

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PBI_DATA", Path.home() / "pbi_data"))
CACHE = DATA / "cache" / "features"
DATASETS = ("things_eeg2", "things_meg", "things_fmri")
FP16_ABOVE = 300e6
LAYER_FRACS = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class Spec:
    model_id: str
    family: str
    objective: str
    lib: str            # timm | open_clip | tv | timm_random
    name: str           # timm name / open_clip arch
    tag: str = ""       # open_clip pretrained tag
    training_data: str = ""
    size_label: str = ""
    seed: int = 0
    approx_params: float = 0.0  # for ordering only (millions)
    extra: dict = field(default_factory=dict)


def _registry():
    R = []
    T = lambda *a, **k: R.append(Spec(*a, **k))
    # supervised (ImageNet)
    for d, tag, p in [(18, "a1_in1k", 12), (34, "a1_in1k", 22), (50, "a1_in1k", 24),
                      (101, "a1h_in1k", 43), (152, "a1h_in1k", 58)]:
        T(f"resnet{d}", "resnet", "supervised", "timm", f"resnet{d}.{tag}",
          training_data="ImageNet-1k", size_label=f"R{d}", approx_params=p)
    for s, lab, p in [("tiny", "Ti", 6), ("small", "S", 22), ("base", "B", 86), ("large", "L", 304)]:
        T(f"vit_{s}_p16_in21k", "vit_sup", "supervised", "timm",
          f"vit_{s}_patch16_224.augreg_in21k_ft_in1k", training_data="ImageNet-21k->1k",
          size_label=lab, approx_params=p)
    for s, lab, p in [("tiny", "T", 28), ("small", "S", 50), ("base", "B", 88), ("large", "L", 197)]:
        T(f"convnext_{s}", "convnext", "supervised", "timm", f"convnext_{s}.fb_in1k",
          training_data="ImageNet-1k", size_label=lab, approx_params=p)
    T("alexnet", "alexnet", "supervised", "tv", "alexnet", training_data="ImageNet-1k",
      size_label="AlexNet", approx_params=57)
    # self-supervised
    for s, lab, p in [("small", "S", 22), ("base", "B", 86)]:
        T(f"dino_vit_{s}_p16", "dino", "ssl", "timm", f"vit_{s}_patch16_224.dino",
          training_data="ImageNet-1k (no labels)", size_label=lab, approx_params=p)
    for s, lab, p in [("small", "S", 22), ("base", "B", 86), ("large", "L", 304), ("giant", "g", 1100)]:
        T(f"dinov2_vit_{s}_p14", "dinov2", "ssl", "timm", f"vit_{s}_patch14_dinov2.lvd142m",
          training_data="LVD-142M", size_label=lab, approx_params=p)
    for s, lab, ps, p in [("base", "B", 16, 86), ("large", "L", 16, 304), ("huge", "H", 14, 632)]:
        T(f"mae_vit_{s}", "mae", "ssl", "timm", f"vit_{s}_patch{ps}_224.mae",
          training_data="ImageNet-1k (no labels)", size_label=lab, approx_params=p)
    # CLIP image towers
    for arch, tag, fam, lab, p in [
        ("RN50", "openai", "clip_openai", "RN50", 38), ("ViT-B-32", "openai", "clip_openai", "B/32", 88),
        ("ViT-B-16", "openai", "clip_openai", "B/16", 86), ("ViT-L-14", "openai", "clip_openai", "L/14", 304),
        ("ViT-B-32", "laion2b_s34b_b79k", "clip_laion", "B/32", 88),
        ("ViT-B-16", "laion2b_s34b_b88k", "clip_laion", "B/16", 86),
        ("ViT-L-14", "laion2b_s32b_b82k", "clip_laion", "L/14", 304),
        ("ViT-H-14", "laion2b_s32b_b79k", "clip_laion", "H/14", 632),
        ("ViT-g-14", "laion2b_s34b_b88k", "clip_laion", "g/14", 1013)]:
        mid = f"clip_{arch.lower().replace('-', '_')}_{'openai' if tag == 'openai' else 'laion2b'}"
        T(mid, fam, "clip", "open_clip", arch, tag,
          training_data="WIT-400M" if tag == "openai" else "LAION-2B", size_label=lab, approx_params=p)
    for arch, lab, p in [("ViT-B-16-SigLIP", "B/16", 92), ("ViT-L-16-SigLIP-256", "L/16", 316),
                         ("ViT-SO400M-14-SigLIP", "So400m/14", 428)]:
        T("siglip_" + arch.lower().replace("-", "_").replace("_siglip", ""), "siglip", "siglip",
          "open_clip", arch, "webli", training_data="WebLI", size_label=lab, approx_params=p)
    # random-init controls
    for name, short, p in [("resnet50", "resnet50", 24), ("vit_base_patch16_224", "vit_base_p16", 86),
                           ("convnext_tiny", "convnext_tiny", 28)]:
        for s in (0, 1, 2):
            T(f"random_{short}_s{s}", f"random_{short}", "random", "timm_random", name,
              training_data="none", size_label=short, seed=s, approx_params=p)
    return {s.model_id: s for s in R}


REGISTRY = _registry()


# ---------------------------------------------------------------- loading

class Loaded:
    def __init__(self, model, transform, dtype, source, params, layer_modules=None):
        self.model, self.transform, self.dtype = model, transform, dtype
        self.source, self.params = source, params
        self.layer_modules = layer_modules or []


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _nparams(m):
    return int(sum(p.numel() for p in m.parameters()))


def _timm_layers(m):
    """Depth-fraction modules for the secondary layer sweep."""
    if hasattr(m, "blocks"):
        blocks = list(m.blocks)
    elif hasattr(m, "stages"):
        blocks = list(m.stages)
    elif hasattr(m, "layer4"):
        blocks = [m.layer1, m.layer2, m.layer3, m.layer4]
    elif hasattr(m, "trunk"):
        return _timm_layers(m.trunk)
    elif hasattr(m, "transformer") and hasattr(m.transformer, "resblocks"):
        blocks = list(m.transformer.resblocks)
    else:
        return []
    n = len(blocks)
    return [blocks[max(0, int(np.ceil(f * n)) - 1)] for f in LAYER_FRACS]


def load(spec: Spec) -> Loaded:
    import timm
    dev = device()
    if spec.lib in ("timm", "timm_random"):
        if spec.lib == "timm_random":
            torch.manual_seed(spec.seed)
            m = timm.create_model(spec.name, pretrained=False, num_classes=0)
            src = f"timm:{spec.name} pretrained=False torch.manual_seed({spec.seed})"
        else:
            m = timm.create_model(spec.name, pretrained=True, num_classes=0)
            src = f"timm:{spec.name}"
        cfg = timm.data.resolve_data_config({}, model=m)
        tf = timm.data.create_transform(**cfg)
        src += f" (input {cfg['input_size'][1]}px, {cfg['interpolation']}, crop {cfg['crop_pct']})"
        layers = _timm_layers(m)
    elif spec.lib == "open_clip":
        import open_clip
        full, _, tf = open_clip.create_model_and_transforms(spec.name, pretrained=spec.tag)
        m = full.visual
        del full
        src = f"open_clip:{spec.name}/{spec.tag} (encode_image = visual())"
        layers = _timm_layers(m)
    elif spec.lib == "tv":
        import torchvision
        w = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
        m = torchvision.models.alexnet(weights=w)
        m.classifier[6] = torch.nn.Identity()
        tf = w.transforms()
        src = "torchvision:alexnet IMAGENET1K_V1, classifier[6] removed (fc7+ReLU, 4096-d)"
        layers = [m.features[2], m.features[5], m.features[12], m.classifier[2], m.classifier[5]]
    else:
        raise ValueError(spec.lib)
    m.eval()
    params = _nparams(m)
    dtype = torch.float16 if params > FP16_ABOVE else torch.float32
    m = m.to(dev, dtype=dtype)
    return Loaded(m, tf, dtype, src, params, layers)


def free(L):
    if L is not None:
        L.model.to("cpu")
        del L.model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------- extraction

def _batch_size(params):
    return 32 if params < 150e6 else (16 if params < 700e6 else 8)


def _load_img(p, tf):
    with Image.open(p) as im:
        return tf(im.convert("RGB"))


def _pool(t):
    t = t[0] if isinstance(t, (tuple, list)) else t
    if t.ndim == 4:      # B C H W
        return t.float().mean((2, 3))
    if t.ndim == 3:      # B N D  (or N B D for open_clip resblocks w/o batch_first; handled by caller)
        return t.float().mean(1)
    return t.float()


@torch.no_grad()
def extract(L: Loaded, paths, layers=False, log_every=0):
    """Return (n, d) primary features, and optionally list of (n, d_l) layer features."""
    dev = device()
    bs = _batch_size(L.params)
    feats, lay = [], [[] for _ in L.layer_modules] if layers else None
    store = {}
    hooks = []
    if layers:
        for i, mod in enumerate(L.layer_modules):
            def h(_m, _in, out, i=i):
                store[i] = out
            hooks.append(mod.register_forward_hook(h))
    try:
        t0 = time.time()
        for b in range(0, len(paths), bs):
            x = torch.stack([_load_img(p, L.transform) for p in paths[b:b + bs]]).to(dev, dtype=L.dtype)
            y = L.model(x)
            y = y[0] if isinstance(y, (tuple, list)) else y
            feats.append(y.float().cpu().numpy())
            if layers:
                for i in range(len(L.layer_modules)):
                    o = store[i]
                    o = o[0] if isinstance(o, (tuple, list)) else o
                    if o.ndim == 3 and o.shape[0] != x.shape[0] and o.shape[1] == x.shape[0]:
                        o = o.transpose(0, 1)  # LND -> NLD
                    lay[i].append(_pool(o).cpu().numpy())
            if log_every and (b // bs) % log_every == 0:
                print(f"    {b + len(x)}/{len(paths)}  {time.time() - t0:.0f}s", flush=True)
    finally:
        for h in hooks:
            h.remove()
    F = np.concatenate(feats).astype(np.float32)
    if layers:
        return F, [np.concatenate(l).astype(np.float32) for l in lay]
    return F


def cached_features(spec, L_getter, paths, cache_name):
    """Per-(model, image path) feature cache: CACHE/model_id/cache_name.npz holds path->row."""
    f = CACHE / spec.model_id / f"{cache_name}.npz"
    have = {}
    if f.exists():
        d = np.load(f, allow_pickle=False)
        have = dict(zip(d["paths"].tolist(), d["feats"]))
    keys = [str(Path(p).resolve()) for p in paths]
    missing = [k for k in dict.fromkeys(keys) if k not in have]
    if missing:
        L = L_getter()
        F = extract(L, missing, log_every=20 if len(missing) > 1000 else 0)
        have.update(zip(missing, F))
        f.parent.mkdir(parents=True, exist_ok=True)
        allk = list(have)
        np.savez(f, paths=np.array(allk), feats=np.stack([have[k] for k in allk]))
    return np.stack([have[k] for k in keys])


# ---------------------------------------------------------------- datasets / io

def stimulus_paths(dataset):
    df = pd.read_csv(REPO / "artifacts" / "stimuli" / f"{dataset}.csv").sort_values("idx")
    assert (df["idx"].values == np.arange(len(df))).all()
    root = DATA / "images" / dataset
    paths = [root / p for p in df["path_rel"]]
    ok = all(p.exists() for p in paths)
    return paths, ok


def kernel_path(dataset, model_id):
    return REPO / "artifacts" / "kernels" / dataset / "model" / f"{model_id}.npz"


def layer_path(dataset, model_id, i):
    return REPO / "artifacts" / "layers" / dataset / model_id / f"layer{i}.npz"


META_FIELDS = ["model_id", "family", "modality", "objective", "training_data", "params", "emb_dim",
               "capability", "size_label", "source"]
META_CSV = REPO / "artifacts" / "models_vision.csv"
META_JSON = DATA / "cache" / "models_vision_meta.json"  # running state (not committed)


def load_meta():
    if META_JSON.exists():
        return json.loads(META_JSON.read_text())
    if META_CSV.exists():
        return {r["model_id"]: r for r in csv.DictReader(open(META_CSV))}
    return {}


def save_meta(meta):
    META_JSON.parent.mkdir(parents=True, exist_ok=True)
    META_JSON.write_text(json.dumps(meta, indent=1))
    order = list(REGISTRY)
    rows = sorted(meta.values(), key=lambda r: order.index(r["model_id"]) if r["model_id"] in order else 1e9)
    with open(META_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=META_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in META_FIELDS})


def log_blocker(msg):
    with open(REPO / "BLOCKERS.md", "a") as fh:
        fh.write(f"* {time.strftime('%Y-%m-%d')} — **Vision model failed:** {msg}\n")


def git_commit(msg):
    paths = ["pbi/models.py", "pbi/capability.py", "artifacts/models_vision.csv"]
    paths += [str(p.relative_to(REPO)) for p in (REPO / "artifacts" / "kernels").glob("*/model")]
    if (REPO / "artifacts" / "layers").exists():
        paths.append("artifacts/layers")
    paths = [p for p in paths if (REPO / p).exists()]
    for _ in range(10):
        r = subprocess.run(["git", "add", "--"] + paths, cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0:
            r = subprocess.run(["git", "commit", "-m", msg + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
                                "--", *paths], cwd=REPO, capture_output=True, text=True)
            if r.returncode == 0 or "nothing" in (r.stdout + r.stderr):
                return
        if "index.lock" not in r.stderr:
            print("git:", r.stdout, r.stderr)
            return
        time.sleep(3)


# ---------------------------------------------------------------- main

def run(model_ids, datasets, force=False, layers=False, commit_every=10):
    meta = load_meta()
    ds_paths = {}
    for ds in datasets:
        p, ok = stimulus_paths(ds)
        if ok:
            ds_paths[ds] = p
        else:
            print(f"[skip dataset] {ds}: images missing")
    done_since_commit = 0
    for mid in sorted(model_ids, key=lambda m: REGISTRY[m].approx_params):
        spec = REGISTRY[mid]
        todo = [ds for ds in ds_paths if force or not kernel_path(ds, mid).exists()]
        todo_layers = [ds for ds in ds_paths if layers and spec.lib != "tv" and
                       (force or not layer_path(ds, mid, len(LAYER_FRACS) - 1).exists())]
        if not todo and not todo_layers and mid in meta:
            continue
        print(f"[{mid}] datasets={todo} layers={todo_layers}", flush=True)
        t0 = time.time()
        L = None
        try:
            def getL():
                nonlocal L
                if L is None:
                    L = load(spec)
                return L
            emb_dim = None
            for ds in todo or ([next(iter(ds_paths))] if mid not in meta else []):
                X = cached_features(spec, getL, ds_paths[ds], ds)
                emb_dim = X.shape[1]
                assert np.isfinite(X).all(), f"non-finite features for {mid} on {ds}"
                kp = kernel_path(ds, mid)
                kp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(kp, **pack(kernels_from_features(X)))
            for ds in todo_layers:
                _, Ls = extract(getL(), ds_paths[ds], layers=True)
                for i, Xl in enumerate(Ls):
                    lp = layer_path(ds, mid, i)
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(lp, **pack(kernels_from_features(Xl)))
            if mid not in meta or force or emb_dim is not None:
                Lm = getL() if (mid not in meta or force) else None
                r = meta.get(mid, {})
                r.update(model_id=mid, family=spec.family, modality="vision", objective=spec.objective,
                         training_data=spec.training_data, size_label=spec.size_label)
                if Lm is not None:
                    r.update(params=Lm.params, source=Lm.source)
                if emb_dim is not None:
                    r["emb_dim"] = emb_dim
                r.setdefault("capability", "")
                meta[mid] = r
                save_meta(meta)
            print(f"[{mid}] ok {time.time() - t0:.0f}s", flush=True)
            done_since_commit += 1
        except Exception as e:  # noqa
            traceback.print_exc()
            log_blocker(f"`{mid}` ({spec.lib}:{spec.name} {spec.tag}) — {type(e).__name__}: {str(e)[:200]} — skipped.")
        finally:
            free(L)
            L = None
        if done_since_commit >= commit_every:
            git_commit(f"Vision model kernels (through {mid})")
            done_since_commit = 0
    if done_since_commit:
        git_commit("Vision model kernels")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--models", default="all", help="all | comma list | family:<fam> | objective:<obj>")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--layers", action="store_true", help="also run secondary layer sweep")
    ap.add_argument("--max-params", type=float, default=None, help="only models with approx params (M) <= this")
    a = ap.parse_args(argv)
    ds = list(DATASETS) if a.dataset == "all" else a.dataset.split(",")
    if a.models == "all":
        ms = list(REGISTRY)
    elif a.models.startswith("family:"):
        ms = [m for m, s in REGISTRY.items() if s.family == a.models[7:]]
    elif a.models.startswith("objective:"):
        ms = [m for m, s in REGISTRY.items() if s.objective == a.models[10:]]
    else:
        ms = a.models.split(",")
    if a.max_params is not None:
        ms = [m for m in ms if REGISTRY[m].approx_params <= a.max_params]
    run(ms, ds, force=a.force, layers=a.layers)


if __name__ == "__main__":
    main()
