# Photo Search & Cluster Manager

Photo Search & Cluster Manager is a GPU-accelerated toolkit for indexing, searching, and grouping photos with CLIP embeddings and FAISS.

## Features
- **GPU-first:** Uses CUDA when available, falls back to CPU.
- **FAISS + CLIP:** Fast cosine similarity search on ViT-B-32-QuickGELU embeddings.
- **Auto k-clustering:** Use `--clusters -1` to pick an optimal number via silhouette analysis.
- **GPT-4o-mini summaries:** Each cluster is described in ≤60-token JSON including `[param_weights]`.
- **/group best/others layout:** Per-cluster `preset.json`, best photo renamed `CHOOSEN_*`, CSV with scores.
- **Rotating logs:** `data/log.txt` rolls at 5 MB retaining three backups.

## Quick Start
```bash
# Install dependencies (GPU build of PyTorch recommended)
pip install -r requirements.txt

# Index a folder of photos
python photosearch.py index --root photos/holiday \
        --index data/holiday.faiss --meta data/holiday.pkl

# Semantic search
python photosearch.py query --index data/holiday.faiss \
        --meta data/holiday.pkl --text "golden hour beach" --k 15

# Group & curate (auto-k, save leftovers)
python terminal.py /group holiday "out/curated" -1 TRUE
```

`terminal.py` also provides `/index`, `/find`, `/group`, `/use`, `/list`, `/current`, and `/exit`. Legacy `/inst` and `/duplicates` commands live in `archived.py`.

## GPU Notes
- PyTorch ≥2.3 wheels with CUDA 12.x are supported.
- Device is selected automatically; override with `PS_DEVICE=cpu`.

## Logs
Backend details are written to `data/log.txt` with rotation at 5 MB (up to three backups). The console shows high-level output only.

## Upgrading
- OpenCLIP models switched to `ViT-B-32-quickgelu`.
- `datetime.utcnow()` replaced with timezone-aware `datetime.now(timezone.utc)`.
