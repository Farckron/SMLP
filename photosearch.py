#!/usr/bin/env python3
"""
photosearch.py ‑‑ CLIP + FAISS semantic image search (GPU‑first, 2025‑08‑30)

Steps:
  ▶  Index  : python photosearch.py index --root photos --index data/set.faiss --meta data/set.pkl
  ▶  Query  : python photosearch.py query --index data/set.faiss --meta data/set.pkl --text "golden hour" --k 12

Changes vs 2023 version
  • Uses CUDA automatically when available (set PS_DEVICE=cpu to override)
  • Model switched to ViT‑B‑32‑quickgelu → no activation mismatch warning
  • Deprecation fix: datetime.utcnow() → datetime.now(timezone.utc)
  • RotatingFileHandler writes to data/log.txt (5 MB, 3 backups)
"""
from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0. Constants & logging
# ---------------------------------------------------------------------------
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / "log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] photosearch: %(message)s",
    handlers=[logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=5_242_880, backupCount=3)],
)
log = logging.getLogger("photosearch")

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# 1. Heavy deps (torch / faiss / open_clip)
# ---------------------------------------------------------------------------
try:
    import torch
    import faiss
    import open_clip
except Exception as e:
    log.exception("Missing dependency – install requirements.txt first")
    sys.exit(1)

DEVICE = os.environ.get("PS_DEVICE", "auto")
if DEVICE == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info("running on device=%s", DEVICE)

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def _find_images(root: str) -> List[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in ALLOWED_EXTS)


def _safe_open(p: Path) -> Image.Image:
    """Open image as RGB, load data into memory, close fp."""
    with Image.open(p) as im:
        im.load()
        return im.convert("RGB")


# ---------------------------------------------------------------------------
# 3. CLIP / FAISS
# ---------------------------------------------------------------------------

def _load_clip() -> tuple:
    model_name = "ViT-B-32-quickgelu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model.eval().to(DEVICE)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def _build_index(dim: int) -> faiss.Index:
    """Cosine similarity via inner‑product on L2‑normalised vectors"""
    return faiss.IndexFlatIP(dim)


def _embed_images(files: List[Path], model, preprocess, batch: int = 32) -> Tuple[np.ndarray, List[Path]]:
    feats, ok_paths = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(files), batch), desc="embed", unit="batch"):
            batch_paths = files[i : i + batch]
            ims = []
            for p in batch_paths:
                try:
                    ims.append(preprocess(_safe_open(p)))
                except Exception:
                    continue
            if not ims:
                continue
            vec = model.encode_image(torch.stack(ims).to(DEVICE))
            vec = (vec / vec.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
            feats.append(vec)
            ok_paths.extend(batch_paths[: len(ims)])
    if not feats:
        return np.zeros((0, 512), dtype="float32"), []
    return np.concatenate(feats, 0), ok_paths


# ---------------------------------------------------------------------------
# 4. Commands
# ---------------------------------------------------------------------------

def cmd_index(args):
    model, preprocess, _ = _load_clip()
    img_files = _find_images(args.root)
    log.info("found %d images under %s", len(img_files), args.root)
    t0 = time.time()
    feats, paths_ok = _embed_images(img_files, model, preprocess, args.batch)
    dt = time.time() - t0
    if feats.size == 0:
        log.error("no embeddings produced – abort")
        return

    index = _build_index(feats.shape[1]); index.add(feats)
    Path(args.index).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, args.index)

    meta = dict(
        version="0.2.0",
        created=datetime.now(timezone.utc).isoformat(),
        model="ViT-B-32-quickgelu",
        dim=int(feats.shape[1]),
        root=str(Path(args.root).resolve()),
        files=[str(p) for p in paths_ok],
        count=len(paths_ok),
        build_seconds=round(dt, 3),
    )
    with open(args.meta, "wb") as f:
        pickle.dump(meta, f)
    log.info("indexed %d imgs in %.1fs → %s", len(paths_ok), dt, args.index)


def cmd_query(args):
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    index = faiss.read_index(args.index)
    model, _, tok = _load_clip()

    tokens = tok([args.text]).to(DEVICE)
    with torch.no_grad():
        qvec = model.encode_text(tokens)
    qvec = (qvec / qvec.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")

    k = min(args.k, len(meta["files"]))
    D, I = index.search(qvec, k)
    for r, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        print(f"{r:2d}. {score:.4f}  {meta['files'][idx]}")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser("PhotoSearch – CLIP+FAISS")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="index a folder of images")
    p_idx.add_argument("--root", required=True)
    p_idx.add_argument("--index", required=True)
    p_idx.add_argument("--meta", required=True)
    p_idx.add_argument("--batch", type=int, default=32)
    p_idx.set_defaults(func=cmd_index)

    p_q = sub.add_parser("query", help="query an existing index")
    p_q.add_argument("--index", required=True)
    p_q.add_argument("--meta", required=True)
    p_q.add_argument("--text", required=True)
    p_q.add_argument("--k", type=int, default=12)
    p_q.set_defaults(func=cmd_query)
    return p


if __name__ == "__main__":
    a = _build_parser().parse_args(); a.func(a)