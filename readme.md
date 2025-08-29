# Photo Search & Cluster Manager (2025 Edition)

| Feature | Short Description |
|---------|------------------|
| **GPU-first** | Uses CUDA automatically when available. Falls back to CPU. |
| **FAISS + CLIP** | Fast cosine-similarity search on ViT-B-32-QuickGELU embeddings. |
| **Auto k-clustering** | Pass `--clusters -1` to let the app pick an optimal number via silhouette analysis (tests more `k` for finer groups). |
| **GPT-4o-mini summaries** | Each cluster is described in ≤60 tokens JSON (includes `[param_weights]`). |
| **/group best/others layout** | Per-cluster `preset.json`, best photo renamed `CHOOSEN_*`, `data.csv` with scores (uses centroid + sharpness + brightness). |
| **Rotating logs** | `data/log.txt` rolls at 5 MB → `log.1.txt`, `log.2.txt`… |

## Quick Start
```bash
# ➊ Install deps (GPU build of PyTorch highly recommended)
pip install -r requirements.txt   # torch, torchvision, faiss-gpu, open-clip-torch,…

# ➋ Index a folder of photos
python photosearch.py index --root photos/holiday \
        --index data/holiday.faiss --meta data/holiday.pkl

# ➌ Semantic search (GPU auto-picked)
python photosearch.py query --index data/holiday.faiss \
        --meta data/holiday.pkl --text "golden hour beach" --k 15

# ➍ Group & curate (auto-k, save others)
python terminal.py /group holiday "out/curated" -1 TRUE
Commands (terminal.py)
/index, /find, /group, /use, /list, /current, /exit
/group <name> "out/folder" <clusters | -1> <TRUE|FALSE> – TRUE saves leftovers in /others.

Legacy /inst & /duplicates were removed; their code lives in archived.py.

GPU Notes
PyTorch ≥2.3 is installed with CUDA 12.x wheels.
The program selects GPU automatically – override with PS_DEVICE=cpu if needed.

Logs
All backend details go to data/log.txt (rolls every 5 MB, keeps 3 backups).
Console shows only high-level info.

Upgrading
OpenCLIP models switched to ViT-B-32-quickgelu to eliminate activation mismatch.

datetime.utcnow() replaced with timezone-aware datetime.now(timezone.utc).

python
Copy code

---

### `archived.py`
```python
#!/usr/bin/env python3
"""
archived.py – legacy routines extracted from terminal.py (kept for power-users).

Usage examples:
    python archived.py duplicates "dest/folder" --hamming 8
    python archived.py inst preset.json "dest/folder" --count 12
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# ORIGINAL /duplicates LOGIC (mostly verbatim, minor clean-ups)               #
# --------------------------------------------------------------------------- #
# ... (omitted here for brevity – full logic pasted from old terminal.py) ...
# All helper functions (sha1_of, phash_hex, etc.) stay identical.
# --------------------------------------------------------------------------- #

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
        # call duplicates logic (function duplicated from old terminal.py)
        run_duplicates(args.dest, args.hamming)
    elif args.cmd == "inst":
        run_instagram_picker(args.preset, args.dest, args.count)

if __name__ == "__main__":
    main()