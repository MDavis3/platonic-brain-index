"""Language-only and audio model kernels on the concept names of each dataset's stimuli.

Language (preregistered): sentence "A photo of a/an {concept}." -> mean over tokens of the final hidden
state (attention-masked; one sentence per forward pass, so no padding). CLIP text towers use open_clip
`encode_text` (projected EOT embedding = the tower's pooled output). SBERT models: mean pooling + L2
normalisation, i.e. the sentence-transformers pipeline of those checkpoints (exploratory).

Audio (exploratory): concept name spoken by macOS `say` (default voice) -> 16 kHz mono wav.
Whisper: encoder on the 30 s padded log-mel, mean over the encoder frames (50 Hz) covering the audio.
HuBERT: mean over all frames of the final hidden state.

Embeddings are computed once per unique concept (union over datasets) and cached under
~/pbi_data/cache/langaudio/; kernels go to artifacts/kernels/{dataset}/model/{model_id}.npz.

    python -m pbi.text_audio --dataset all [--models id1,id2] [--force]
"""
import argparse
import gc
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pbi.kernels import kernels_from_features, pack

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PBI_DATA", Path.home() / "pbi_data"))
CACHE = DATA / "cache" / "langaudio"
TTS = DATA / "cache" / "tts"
DATASETS = ("things_eeg2", "things_meg", "things_fmri")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# model_id, family, modality, objective, training_data, size_label, source
MODELS = [
    ("pythia-70m", "pythia", "language", "lm", "The Pile (300B tok, standard)", "70M", "EleutherAI/pythia-70m"),
    ("pythia-160m", "pythia", "language", "lm", "The Pile (300B tok, standard)", "160M", "EleutherAI/pythia-160m"),
    ("pythia-410m", "pythia", "language", "lm", "The Pile (300B tok, standard)", "410M", "EleutherAI/pythia-410m"),
    ("pythia-1b", "pythia", "language", "lm", "The Pile (300B tok, standard)", "1B", "EleutherAI/pythia-1b"),
    ("pythia-1.4b", "pythia", "language", "lm", "The Pile (300B tok, standard)", "1.4B", "EleutherAI/pythia-1.4b"),
    ("pythia-2.8b", "pythia", "language", "lm", "The Pile (300B tok, standard)", "2.8B", "EleutherAI/pythia-2.8b"),
    ("qwen2.5-0.5b", "qwen2.5", "language", "lm", "Qwen2.5 pretraining (18T tok)", "0.5B", "Qwen/Qwen2.5-0.5B"),
    ("qwen2.5-1.5b", "qwen2.5", "language", "lm", "Qwen2.5 pretraining (18T tok)", "1.5B", "Qwen/Qwen2.5-1.5B"),
    ("qwen2.5-3b", "qwen2.5", "language", "lm", "Qwen2.5 pretraining (18T tok)", "3B", "Qwen/Qwen2.5-3B"),
    ("cliptext-vit-b-32-openai", "clip_text", "language", "clip_text", "WIT-400M (OpenAI)", "B/32", "open_clip:ViT-B-32/openai"),
    ("cliptext-vit-l-14-openai", "clip_text", "language", "clip_text", "WIT-400M (OpenAI)", "L/14", "open_clip:ViT-L-14/openai"),
    ("cliptext-vit-h-14-laion2b", "clip_text", "language", "clip_text", "LAION-2B", "H/14", "open_clip:ViT-H-14/laion2b_s32b_b79k"),
    ("sbert-minilm-l6", "sbert", "language", "sbert", "1B sentence pairs (sentence-transformers)", "L6", "sentence-transformers/all-MiniLM-L6-v2"),
    ("sbert-mpnet-base", "sbert", "language", "sbert", "1B sentence pairs (sentence-transformers)", "base", "sentence-transformers/all-mpnet-base-v2"),
    ("whisper-tiny", "whisper", "audio", "asr", "680k h weakly supervised speech", "tiny", "openai/whisper-tiny"),
    ("whisper-base", "whisper", "audio", "asr", "680k h weakly supervised speech", "base", "openai/whisper-base"),
    ("whisper-small", "whisper", "audio", "asr", "680k h weakly supervised speech", "small", "openai/whisper-small"),
    ("whisper-medium", "whisper", "audio", "asr", "680k h weakly supervised speech", "medium", "openai/whisper-medium"),
    ("hubert-base-ls960", "hubert", "audio", "audio_ssl", "LibriSpeech 960 h", "base", "facebook/hubert-base-ls960"),
    ("hubert-large-ll60k", "hubert", "audio", "audio_ssl", "Libri-Light 60k h", "large", "facebook/hubert-large-ll60k"),
]
COLS = ["model_id", "family", "modality", "objective", "training_data", "params", "emb_dim",
        "capability", "size_label", "source"]


def sentence(concept):
    art = "an" if concept[:1].lower() in "aeiou" else "a"
    return f"A photo of {art} {concept}."


def load_stimuli():
    return {d: pd.read_csv(ROOT / "artifacts" / "stimuli" / f"{d}.csv") for d in DATASETS}


def nparams(m):
    return int(sum(p.numel() for p in m.parameters()))


def free():
    gc.collect()
    if DEV == "mps":
        torch.mps.empty_cache()


# ------------------------------------------------------------------ language

@torch.no_grad()
def embed_hf_lm(source, texts, sbert=False):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source, dtype=torch.float32)
    P = nparams(model)
    dtype = torch.float16 if P > 300e6 else torch.float32
    model = model.to(DEV, dtype=dtype).eval()
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt").to(DEV)
        h = model(**enc).last_hidden_state.float()[0]          # (tokens, d)
        m = enc["attention_mask"][0].float()[:, None]
        e = (h * m).sum(0) / m.sum()
        if sbert:
            e = torch.nn.functional.normalize(e, dim=0)
        out.append(e.cpu().numpy())
    X = np.stack(out)
    if not np.isfinite(X).all() and dtype == torch.float16:  # fp16 overflow -> retry in bf16
        model = model.to(dtype=torch.bfloat16)
        out = []
        for t in texts:
            enc = tok(t, return_tensors="pt").to(DEV)
            h = model(**enc).last_hidden_state.float()[0]
            m = enc["attention_mask"][0].float()[:, None]
            out.append(((h * m).sum(0) / m.sum()).cpu().numpy())
        X = np.stack(out)
        dtype = torch.bfloat16
    info = {"params": P, "dtype": str(dtype)}
    del model; free()
    return X, info


@torch.no_grad()
def embed_clip_text(source, texts):
    import open_clip
    arch, pre = source.split(":", 1)[1].split("/")
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pre)
    if hasattr(model, "visual"):
        del model.visual
    P = nparams(model) - (int(model.logit_scale.numel()) if hasattr(model, "logit_scale") else 0) \
        - (int(model.logit_bias.numel()) if getattr(model, "logit_bias", None) is not None else 0)
    dtype = torch.float16 if P > 300e6 else torch.float32
    model = model.to(DEV, dtype=dtype).eval()
    tokz = open_clip.get_tokenizer(arch)
    out = []
    for i in range(0, len(texts), 32):
        tk = tokz(texts[i:i + 32]).to(DEV)
        out.append(model.encode_text(tk).float().cpu().numpy())
    X = np.concatenate(out)
    del model; free()
    return X, {"params": P, "dtype": str(dtype)}


# ------------------------------------------------------------------ audio

def tts_wav(concept):
    TTS.mkdir(parents=True, exist_ok=True)
    stem = concept.replace(" ", "_").replace("/", "_")
    wav = TTS / f"{stem}.wav"
    if not wav.exists():
        aiff = TTS / f"{stem}.aiff"
        subprocess.run(["say", "-o", str(aiff), concept], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
        aiff.unlink()
    return wav


def read_wav(path):
    import warnings
    from scipy.io import wavfile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # afconvert FLLR chunk
        sr, a = wavfile.read(path)
    assert sr == 16000, (path, sr)
    a = a.astype(np.float32)
    if a.ndim > 1:
        a = a.mean(1)
    return a / 32768.0


@torch.no_grad()
def embed_whisper(source, waves):
    from transformers import WhisperFeatureExtractor, WhisperModel
    fe = WhisperFeatureExtractor.from_pretrained(source)
    full = WhisperModel.from_pretrained(source, dtype=torch.float32)
    enc = full.get_encoder()
    del full.decoder
    P = nparams(enc)
    dtype = torch.float16 if P > 300e6 else torch.float32
    enc = enc.to(DEV, dtype=dtype).eval()
    out = []
    for a in waves:
        feats = fe(a, sampling_rate=16000, return_tensors="pt").input_features.to(DEV, dtype=dtype)
        h = enc(feats).last_hidden_state.float()[0]           # (1500, d), 50 Hz
        nf = min(len(h), max(1, math.ceil(len(a) / 16000 * 50)))
        out.append(h[:nf].mean(0).cpu().numpy())
    del enc, full; free()
    return np.stack(out), {"params": P, "dtype": str(dtype)}


@torch.no_grad()
def embed_hubert(source, waves):
    from transformers import AutoFeatureExtractor, HubertModel
    fe = AutoFeatureExtractor.from_pretrained(source)
    model = HubertModel.from_pretrained(source, dtype=torch.float32)
    P = nparams(model)
    dtype = torch.float16 if P > 300e6 else torch.float32
    model = model.to(DEV, dtype=dtype).eval()
    out = []
    for a in waves:
        x = fe(a, sampling_rate=16000, return_tensors="pt").input_values.to(DEV, dtype=dtype)
        out.append(model(x).last_hidden_state.float()[0].mean(0).cpu().numpy())
    del model; free()
    return np.stack(out), {"params": P, "dtype": str(dtype)}


# ------------------------------------------------------------------ driver

def embed(row, concepts):
    mid, fam, modality, obj, _, _, src = row
    if modality == "language":
        texts = [sentence(c) for c in concepts]
        if fam == "clip_text":
            return embed_clip_text(src, texts)
        return embed_hf_lm(src, texts, sbert=(fam == "sbert"))
    waves = [read_wav(tts_wav(c)) for c in concepts]
    if fam == "whisper":
        return embed_whisper(src, waves)
    return embed_hubert(src, waves)


def log_blocker(msg):
    with open(ROOT / "BLOCKERS.md", "a") as f:
        f.write(f"* {time.strftime('%Y-%m-%d')} — {msg}\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--models", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    stim = load_stimuli()
    dsets = DATASETS if args.dataset == "all" else tuple(args.dataset.split(","))
    for d in dsets:
        dup = stim[d].concept[stim[d].concept.duplicated()].tolist()
        if dup:
            print(f"[{d}] duplicated concepts (identical embeddings): {dup}")
    concepts = sorted(set().union(*[set(stim[d].concept) for d in dsets]))
    cidx = {c: i for i, c in enumerate(concepts)}
    sel = [m for m in MODELS if not args.models or m[0] in args.models.split(",")]
    meta_path = ROOT / "artifacts" / "models_langaudio.csv"
    meta = pd.read_csv(meta_path).set_index("model_id") if meta_path.exists() else pd.DataFrame(columns=COLS[1:])
    CACHE.mkdir(parents=True, exist_ok=True)
    for row in sel:
        mid = row[0]
        outs = [ROOT / "artifacts" / "kernels" / d / "model" / f"{mid}.npz" for d in dsets]
        if not args.force and all(o.exists() for o in outs) and mid in meta.index:
            print(f"[skip] {mid}")
            continue
        cpath = CACHE / f"{mid}.npz"
        t0 = time.time()
        try:
            if cpath.exists() and not args.force:
                z = np.load(cpath, allow_pickle=True)
                cached = dict(zip(z["concepts"].tolist(), z["X"]))
                info = z["info"].item()
                missing = [c for c in concepts if c not in cached]
            else:
                cached, info, missing = {}, None, concepts
            if missing:
                Xm, info = embed(row, missing)
                cached.update(zip(missing, Xm))
                np.savez(cpath, concepts=np.array(list(cached)), X=np.stack(list(cached.values())),
                         info=np.array(info, dtype=object))
            X = np.stack([cached[c] for c in concepts])
            assert np.isfinite(X).all(), f"{mid}: non-finite embeddings"
        except Exception as e:
            traceback.print_exc()
            log_blocker(f"**{mid}** ({row[6]}) failed: {type(e).__name__}: {str(e)[:200]} — skipped; other models continue.")
            free()
            continue
        for d, o in zip(dsets, outs):
            o.parent.mkdir(parents=True, exist_ok=True)
            Xd = X[[cidx[c] for c in stim[d].concept]]
            np.savez_compressed(o, **pack(kernels_from_features(Xd)))
        meta.loc[mid] = dict(zip(COLS[1:], [row[1], row[2], row[3], row[4], info["params"], X.shape[1], np.nan,
                                            row[5], row[6]]))
        meta.reset_index(names="model_id")[COLS].to_csv(meta_path, index=False)
        meta = pd.read_csv(meta_path).set_index("model_id")
        print(f"[done] {mid}: X={X.shape} params={info['params']:,} {info['dtype']} ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
