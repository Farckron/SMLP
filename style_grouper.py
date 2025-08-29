#!/usr/bin/env python3
"""
style_grouper.py – cluster photos by style, summarise with GPT-4o, select best.

Key upgrades (2025-08-30)
  • GPU-first embeddings (OpenCLIP ViT-B-32-quickgelu)
  • --clusters -1 triggers automatic silhouette-based k selection
  • GPT summary now returns {summary, param_weights[]} (≤60 tokens)
  • BEST / OTHERS tree with preset.json + data.csv + CHOOSEN_* naming
  • Robust logging (console + UTF-8 rotating file) in data/log.txt
"""
from __future__ import annotations
import argparse
import json
import logging
import logging.handlers
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── logging setup ──────────────────────────────────────────────────────────
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / "log.txt"
# File handler (UTF-8, rotating) + console stream handler
file_handler = logging.handlers.RotatingFileHandler(
    LOG_PATH, maxBytes=5_242_880, backupCount=3, encoding="utf-8"
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
)
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)
log = logging.getLogger("style_grouper")

# ─── heavy deps ────────────────────────────────────────────────────────────
import torch
import open_clip
import faiss
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# ─── helpers ───────────────────────────────────────────────────────────────
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def _list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in ALLOWED_EXTS])


def _open_rgb(p: Path) -> Image.Image:
    with Image.open(p) as im:
        im.load()
        return im.convert("RGB")

# technical metrics
import numpy as np
from scipy.signal import convolve2d

def _arr_gray(im: Image.Image) -> np.ndarray:
    return np.array(im.convert("L"), dtype=np.float32)

def sharpness(arr: np.ndarray) -> float:
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
    return float(convolve2d(arr, k, mode="same", boundary="symm").var())

def brightness(arr: np.ndarray) -> float:
    return float(arr.mean() / 255.0)

# copy util
def copy_preserving(
    src_root: Path, src: Path, dst_root: Path, new_name: str | None = None
):
    rel = src.relative_to(src_root)
    out = (
        dst_root / rel
        if new_name is None
        else dst_root / rel.parent / new_name
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)

# ─── embeddings ────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def embed_images(paths: List[Path], batch: int = 32) -> np.ndarray:
    model, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    model.eval().to(DEVICE)
    vecs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="embed", unit="batch"):
            ims = []
            for p in paths[i : i + batch]:
                try:
                    ims.append(pre(_open_rgb(p)))
                except Exception:
                    continue
            if not ims:
                continue
            v = model.encode_image(torch.stack(ims).to(DEVICE))
            v = (v / v.norm(dim=-1, keepdim=True)).cpu().numpy().astype("float32")
            vecs.append(v)
    return np.concatenate(vecs) if vecs else np.zeros((0, 512), dtype="float32")

# ─── auto-k selection ───────────────────────────────────────────────────────
def auto_k(feats: np.ndarray, k_max: int = 30) -> int:
    """Estimate optimal k via silhouette score.

    ``range(2, k_max)`` previously skipped ``k_max`` and the default ceiling was
    fairly low which often under‑clustered datasets.  The updated version sweeps
    up to and including ``k_max`` (capped by the sample size) so more candidate
    cluster counts are considered.
    """
    if len(feats) < 2:
        return 1
    k_max = min(k_max, len(feats) - 1)
    if k_max < 2:
        return 1
    best_k, best_s = 2, -1.0
    for k in range(2, k_max + 1):
        km = MiniBatchKMeans(
            n_clusters=k, random_state=0, batch_size=256, n_init="auto"
        )
        lbls = km.fit_predict(feats)
        if len(set(lbls)) < 2:
            continue
        s = silhouette_score(feats, lbls)
        if s > best_s:
            best_k, best_s = k, s
    log.info("auto-k selected %d (silhouette=%.3f)", best_k, best_s)
    return best_k

# ─── GPT summarisation ─────────────────────────────────────────────────────
from dotenv import load_dotenv; load_dotenv()
from openai import OpenAI
_client = OpenAI()

def gpt_summary(
    names: List[str], stats: Dict[str, float]
) -> Dict[str, object]:
    """Return concise JSON with summary and ordered param_weights."""
    prompt = (
        "You are PhotoCuratorAI. Summarise the cluster in <60 tokens JSON, "
        "keys: summary + param_weights array by importance. No extra text.\n\n"
        f"KEYWORDS: {', '.join(names[:15])}\n"
        f"STATS: {json.dumps(stats, separators=(',',':'))}"
    )
    try:
        rsp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=60,
            temperature=0.4,
        )
        return json.loads(rsp.choices[0].message.content.strip())
    except Exception as e:
        log.warning("GPT summary failed: %s", e)
        return {"summary": "n/a", "param_weights": ["aesthetic", "sharpness"]}

# ─── main pipeline ─────────────────────────────────────────────────────────
def curate(args):
    root = Path(args.root).resolve()
    paths = _list_images(root)
    feats = embed_images(paths)

    # determine cluster count
    k = args.clusters if args.clusters != -1 else auto_k(feats)
    km = MiniBatchKMeans(
        n_clusters=k, random_state=0, batch_size=256, n_init="auto"
    )
    labels = km.fit_predict(feats)

    out = Path(args.out).resolve()
    (out / "best").mkdir(parents=True, exist_ok=True)
    others_root = out / "others"
    if args.save_rest:
        others_root.mkdir(exist_ok=True)

    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        c_paths = [paths[i] for i in idxs]
        c_feats = feats[idxs]
        br_vals, sh_vals = [], []
        for p in c_paths:
            arr = _arr_gray(_open_rgb(p))
            br_vals.append(brightness(arr))
            sh_vals.append(sharpness(arr))
        stats = {
            "count": len(c_paths),
            "bright": round(np.mean(br_vals), 3),
            "sharp": round(np.mean(sh_vals), 3)
        }
        smry = gpt_summary([p.stem for p in c_paths], stats)

        # score and pick best (centroid proximity + sharpness + brightness)
        centroid = km.cluster_centers_[cid]
        dist = np.linalg.norm(c_feats - centroid, axis=1)
        dist_norm = 1 - (dist - dist.min()) / (dist.ptp() + 1e-8)
        sh_arr = np.array(sh_vals)
        sh_norm = (sh_arr - sh_arr.min()) / (sh_arr.ptp() + 1e-8)
        br_arr = np.array(br_vals)
        scores = 0.5 * dist_norm + 0.3 * sh_norm + 0.2 * br_arr
        best_idx = int(np.argmax(scores))
        best_p = c_paths[best_idx]

        # setup cluster folder
        target = others_root if args.save_rest else out / "best"
        clus_root = target / f"cluster{cid}"
        clus_root.mkdir(parents=True, exist_ok=True)

        # copy chosen image
        copy_preserving(root, best_p, clus_root, new_name=f"CHOOSEN_{best_p.name}")
        # copy leftovers
        if args.save_rest:
            for i, p in enumerate(c_paths):
                if i == best_idx:
                    continue
                copy_preserving(root, p, clus_root)

        # write preset.json + data.csv
        preset = {"created": datetime.now(timezone.utc).isoformat(), **smry, "weights": stats}
        (clus_root / "preset.json").write_text(
            json.dumps(preset, indent=2, ensure_ascii=False)
        )
        with open(clus_root / "data.csv", "w", encoding="utf-8") as f:
            f.write("path,score\n")
            for p, s in sorted(zip(c_paths, scores), key=lambda x: -x[1]):
                f.write(f"{p.relative_to(root)},{s:.4f}\n")

        log.info("cluster %d -> %s (n=%d)", cid, clus_root, len(c_paths))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clusters", type=int, default=-1)
    ap.add_argument(
        "--save-rest",
        type=lambda x: x.lower().startswith("t"),
        default=True
    )
    args = ap.parse_args()
    curate(args)
