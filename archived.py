#!/usr/bin/env python3
"""
archived.py – legacy routines extracted from terminal.py (kept for power-users).

Usage:
    python archived.py duplicates "dest/folder" --hamming 8
    python archived.py inst preset.json "dest/folder" --count 12
"""
from __future__ import annotations

import argparse, json, os, pickle, hashlib, shutil, math
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATA_DIR = Path(os.environ.get("PS_DATA", "data"))
REG_PATH = DATA_DIR / "registry.json"
DEVICE = os.environ.get("PS_DEVICE", "cpu")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

try:
    from PIL import Image
    import imagehash
except Exception:
    Image = None
    imagehash = None

import numpy as np

# registry helpers

def load_reg() -> Dict:
    if REG_PATH.exists():
        with open(REG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def get_current(reg: Dict):
    return reg.get("_current")

def read_meta(meta_path: str):
    try:
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def list_images_under(root: str) -> List[str]:
    files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = str(Path(dirpath) / fn)
            if Path(full).suffix.lower() in ALLOWED_EXTS:
                files.append(full)
    files.sort()
    return files

def files_for_entry(entry: dict) -> Tuple[str, List[str]]:
    meta = read_meta(entry.get("meta", ""))
    root = entry.get("root")
    if meta and meta.get("files"):
        return root, [f for f in meta["files"] if os.path.isfile(f)]
    return root, list_images_under(root)

def sha1_of(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def img_dims(path: str) -> Tuple[int, int]:
    if Image is None:
        return 0, 0
    try:
        with Image.open(path) as im:
            im.load()
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0

def phash_hex(path: str):
    if Image is None or imagehash is None:
        return None
    try:
        with Image.open(path) as im:
            h = imagehash.phash(im)
            return str(h)
    except Exception:
        return None

def hamming(a_hex: str, b_hex: str) -> int:
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")

def best_rep(paths: List[str]) -> str:
    scored: List[Tuple[Tuple[int, int, int], str]] = []
    for p in paths:
        w, h = img_dims(p)
        size = os.path.getsize(p) if os.path.exists(p) else 0
        scored.append(((w * h, size, -len(p)), p))
    scored.sort(reverse=True)
    return scored[0][1]

def copy_preserving(root: str, src: str, dest_root: str):
    rel = os.path.relpath(src, root)
    out_path = Path(dest_root) / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)

# --- helpers for /inst

def to_gray_np(im: Image.Image) -> np.ndarray:
    arr = np.array(im.convert("L"), dtype=np.float32)
    return arr

def laplacian_var(arr: np.ndarray) -> float:
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from scipy.signal import convolve2d as conv2
    try:
        resp = conv2(arr, k, mode="same", boundary="symm")
        return float(resp.var())
    except Exception:
        pad = np.pad(arr, 1, mode='edge')
        resp = (
            pad[1:-1,2:] + pad[1:-1,:-2] + pad[2:,1:-1] + pad[:-2,1:-1] - 4*pad[1:-1,1:-1]
        )
        return float(resp.var())

def brightness_and_clip(arr: np.ndarray) -> Tuple[float, float]:
    mean = float(arr.mean() / 255.0)
    low_clip = float((arr <= 3).mean())
    high_clip = float((arr >= 252).mean())
    clip = low_clip + high_clip
    return mean, clip

def load_preset(preset_path: str) -> Dict[str, Any]:
    with open(preset_path, "r", encoding="utf-8") as f:
        preset = json.load(f)
    preset.setdefault("version", "1")
    preset.setdefault("weights", {})
    w = preset["weights"]
    w.setdefault("aesthetic", 0.5)
    w.setdefault("sharpness", 0.2)
    w.setdefault("exposure", 0.2)
    w.setdefault("resolution", 0.1)
    preset.setdefault("prompts", [{"text": "a sharp, well-composed Instagram photo in natural light", "weight": 1.0}])
    c = preset.setdefault("constraints", {})
    c.setdefault("min_width", 1080)
    c.setdefault("min_height", 1080)
    c.setdefault("aspect_min", 0.8)
    c.setdefault("aspect_max", 1.91)
    c.setdefault("target_pixels", 1080*1350)
    return preset

def normalize(arr: List[float]) -> List[float]:
    if not arr:
        return []
    a = np.array(arr, dtype=np.float32)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [0.0 for _ in arr]
    return [float((x - lo) / (hi - lo)) for x in a]

# duplicates command

def run_duplicates(dest: str, hamming_th: int = 8) -> None:
    if Image is None or imagehash is None:
        print("[error] Missing deps. Install: pip install pillow imagehash")
        return
    reg = load_reg()
    cur = get_current(reg)
    if not cur:
        print("[error] no active photoset. use /use <name> or /index first")
        return
    entry = reg.get(cur)
    if not entry:
        print(f"[error] current photoset '{cur}' not found")
        return
    root, files = files_for_entry(entry)
    if not files:
        print("[error] no images found in current photoset")
        return
    print(f"[info] scanning {len(files)} files for duplicates (exact + pHash<= {hamming_th})")

    by_sha: Dict[str, List[str]] = {}
    for p in files:
        try:
            h = sha1_of(p)
        except Exception:
            continue
        by_sha.setdefault(h, []).append(p)
    dupes_exact = set()
    for group in by_sha.values():
        keep_rep = best_rep(group)
        for g in group:
            if g != keep_rep:
                dupes_exact.add(g)
    candidates = [p for p in files if p not in dupes_exact]
    phashes: Dict[str, str] = {}
    for p in candidates:
        hh = phash_hex(p)
        if hh:
            phashes[p] = hh
    buckets: Dict[str, List[str]] = {}
    for p, hh in phashes.items():
        pref = hh[:4]
        buckets.setdefault(pref, []).append(p)
    keep = set()
    dupes = set(dupes_exact)
    for group in buckets.values():
        group = sorted(group)
        for i, a in enumerate(group):
            if a in dupes or a in keep:
                continue
            a_hash = phashes[a]
            cluster = [a]
            for b in group[i+1:]:
                if b in dupes or b in keep:
                    continue
                d = hamming(a_hash, phashes[b])
                if d <= hamming_th:
                    cluster.append(b)
            rep = best_rep(cluster)
            keep.add(rep)
            for p in cluster:
                if p != rep:
                    dupes.add(p)
    for p in candidates:
        if p not in keep and p not in dupes:
            keep.add(p)
    kept = sorted(list(keep))
    total = len(files)
    deleted = total - len(kept)
    Path(dest).mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(p, root)
    deleted_list = sorted(rel(p) for p in dupes)
    kept_list = sorted(rel(p) for p in kept)
    for p in kept:
        try:
            copy_preserving(root, p, dest)
        except Exception:
            pass
    try:
        with open(Path(dest)/"deleted.txt", "w", encoding="utf-8") as f:
            for r in deleted_list:
                f.write(r + "\n")
        with open(Path(dest)/"kept.txt", "w", encoding="utf-8") as f:
            for r in kept_list:
                f.write(r + "\n")
        with open(Path(dest)/"manifest.csv", "w", encoding="utf-8") as f:
            f.write("status,path\n")
            for r in kept_list:
                f.write(f"keep,{r}\n")
            for r in deleted_list:
                f.write(f"delete,{r}\n")
    except Exception:
        pass
    print("[deleted]")
    for r in deleted_list:
        print(r)
    print(f"[summary] total={total} kept={len(kept)} deleted={deleted} → {dest}")

# instagram picker

def run_instagram_picker(preset_path: str, dest: str, count: int = 12) -> None:
    if Image is None:
        print("[error] Missing deps. Install: pip install pillow")
        return
    reg = load_reg()
    cur = get_current(reg)
    if not cur:
        print("[error] no active photoset. use /use <name> or /index first")
        return
    entry = reg.get(cur)
    if not entry:
        print(f"[error] current photoset '{cur}' not found")
        return
    root, files = files_for_entry(entry)
    if not files:
        print("[error] no images found in current photoset")
        return
    try:
        preset = load_preset(preset_path)
    except Exception as e:
        print(f"[error] failed to read preset: {e}")
        return
    try:
        import torch
        import open_clip
    except Exception as e:
        print(f"[error] missing deps for aesthetics: {e}. Install: pip install torch open-clip-torch")
        return
    device = DEVICE
    if device == "cuda" and (not torch.cuda.is_available()):
        print("[warn] CUDA requested but not available. Using CPU.")
        device = "cpu"
    model_name = preset.get("model", "ViT-B-32-quickgelu")
    pretrained = preset.get("pretrained", "openai")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval().to(device)
    prompts = preset.get("prompts", [])
    texts = [p.get("text", "") for p in prompts if p.get("weight", 0) != 0]
    tweights = [float(p.get("weight", 1.0)) for p in prompts if p.get("weight", 0) != 0]
    if not texts:
        texts = ["a high quality instagram photo"]
        tweights = [1.0]
    with torch.no_grad():
        toks = tokenizer(texts).to(device)
        tfeat = model.encode_text(toks)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    tfeat = tfeat.float().cpu()
    import torch
    tweights = torch.tensor(tweights, dtype=torch.float32).view(-1, 1)
    cons = preset.get("constraints", {})
    min_w, min_h = int(cons.get("min_width", 0)), int(cons.get("min_height", 0))
    ar_min, ar_max = float(cons.get("aspect_min", 0)), float(cons.get("aspect_max", 10))
    target_pixels = float(cons.get("target_pixels", 1080*1350))
    paths: List[str] = []
    aest_raw: List[float] = []
    sharp_raw: List[float] = []
    expo_raw: List[float] = []
    res_raw: List[float] = []
    dims: List[Tuple[int,int]] = []
    batch_imgs = []
    batch_paths = []
    def flush_batch():
        nonlocal batch_imgs, batch_paths
        if not batch_imgs:
            return []
        with torch.no_grad():
            imgs = torch.stack(batch_imgs).to(device)
            if imgs.dtype != torch.float32:
                imgs = imgs.float()
            ivec = model.encode_image(imgs)
            ivec = ivec / ivec.norm(dim=-1, keepdim=True)
            ivec = ivec.float().cpu()
        sim = (ivec @ tfeat.T)
        wsum = (sim * tweights.T).sum(dim=1) / max(float(tweights.sum().item()), 1e-6)
        scores = wsum.tolist()
        out = list(zip(batch_paths, scores))
        batch_imgs = []
        batch_paths = []
        return out
    for p in files:
        try:
            with Image.open(p) as im:
                im.load()
                w, h = im.width, im.height
                ar = (w / h) if h else 1
                if w < min_w or h < min_h or ar < ar_min or ar > ar_max:
                    continue
                arr = to_gray_np(im)
                b_mean, clip = brightness_and_clip(arr)
                expo_score = max(0.0, 1.0 - abs(b_mean - 0.55)*2.0) * (1.0 - min(clip*5.0, 1.0))
                sharp_score = laplacian_var(arr)
                res_score = min(1.0, (w*h) / target_pixels)
                batch_imgs.append(preprocess(im))
                batch_paths.append(p)
                paths.append(p)
                sharp_raw.append(sharp_score)
                expo_raw.append(expo_score)
                res_raw.append(res_score)
                dims.append((w, h))
        except Exception:
            continue
        if len(batch_imgs) >= 16:
            for bp, s in flush_batch():
                try:
                    idx = paths.index(bp)
                    aest_raw.insert(idx, s)
                except ValueError:
                    pass
    for bp, s in flush_batch():
        try:
            idx = paths.index(bp)
            aest_raw.insert(idx, s)
        except ValueError:
            pass
    if not paths:
        print("[error] no images matched constraints")
        return
    if len(aest_raw) != len(paths):
        avg = float(np.mean(aest_raw)) if aest_raw else 0.0
        while len(aest_raw) < len(paths):
            aest_raw.append(avg)
        if len(aest_raw) > len(paths):
            aest_raw = aest_raw[:len(paths)]
    aest = normalize(aest_raw)
    sharp = normalize(sharp_raw)
    expo = [float(min(max(x, 0.0), 1.0)) for x in expo_raw]
    res = [float(min(max(x, 0.0), 1.0)) for x in res_raw]
    wts = preset.get("weights", {})
    wa = float(wts.get("aesthetic", 0.5))
    ws = float(wts.get("sharpness", 0.2))
    we = float(wts.get("exposure", 0.2))
    wr = float(wts.get("resolution", 0.1))
    totals = [wa*a + ws*s + we*e + wr*r for a,s,e,r in zip(aest, sharp, expo, res)]
    ranked = sorted(list(zip(paths, totals, aest, sharp, expo, res, dims)), key=lambda x: x[1], reverse=True)
    sel = ranked[:min(count, len(ranked))]
    Path(dest).mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(p, entry.get("root"))
    with open(Path(dest)/"selected.csv", "w", encoding="utf-8") as f:
        f.write("path,score,aesthetic,sharpness,exposure,resolution,width,height\n")
        for p, sc, a, s, e, r, (w,h) in sel:
            f.write(f"{rel(p)},{sc:.6f},{a:.4f},{s:.4f},{e:.4f},{r:.4f},{w},{h}\n")
    print("[selected]")
    for p, sc, *_ in sel:
        print(f"{rel(p)}  score={sc:.4f}")
    print(f"[summary] picked={len(sel)} of {len(ranked)} → {dest}")
    for p, *_ in sel:
        try:
            copy_preserving(entry.get("root"), p, dest)
        except Exception:
            pass

def main(argv=None) -> None:
    parser = argparse.ArgumentParser("Archived commands")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_dup = sub.add_parser("duplicates")
    p_dup.add_argument("dest")
    p_dup.add_argument("--hamming", type=int, default=8)
    p_inst = sub.add_parser("inst")
    p_inst.add_argument("preset")
    p_inst.add_argument("dest")
    p_inst.add_argument("--count", type=int, default=12)
    args = parser.parse_args(argv)
    if args.cmd == "duplicates":
        run_duplicates(args.dest, args.hamming)
    elif args.cmd == "inst":
        run_instagram_picker(args.preset, args.dest, args.count)

if __name__ == "__main__":
    main()
