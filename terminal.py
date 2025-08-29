#!/usr/bin/env python3
"""
terminal.py – thin CLI wrapper for photosearch + legacy commands

Commands:
  /index "folder" [name]
  /find  "prompt" [k]
  /group name "out/folder" <clusters|-1> [TRUE|FALSE]
  /duplicates "dest" [hamming<=8]
  /inst "preset.json" "dest" [count]
  /use   name
  /list
  /current
  /exit

Legacy heavy routines (/duplicates and /inst) are executed via archived.py.
"""

from __future__ import annotations
import json, logging, logging.handlers, os, shlex, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

# logging
DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
LOG_PATH = DATA_DIR / "log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] terminal: %(message)s",
    handlers=[logging.handlers.RotatingFileHandler(LOG_PATH, maxBytes=5_242_880, backupCount=3)],
)
log = logging.getLogger("terminal")

# paths
ROOT = Path(__file__).parent
PHOTSEARCH = str(ROOT / "photosearch.py")
ARCHIVED = str(ROOT / "archived.py")
PYBIN = sys.executable or "python"
REG_PATH = DATA_DIR / "registry.json"

HELP = (
    "Commands:\n"
    "  /index \"folder\" [name]\n"
    "  /find  \"prompt\" [k]\n"
    "  /group name \"out/folder\" <clusters|-1> [TRUE|FALSE]\n"
    "  /duplicates \"dest\" [hamming<=8]\n"
    "  /inst \"preset.json\" \"dest\" [count]\n"
    "  /use name\n  /list | /current\n  /exit\n"
)

# registry helpers
def _ensure_reg() -> Dict[str, dict]:
    if REG_PATH.exists():
        with open(REG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _save_reg(reg: Dict):
    DATA_DIR.mkdir(exist_ok=True)
    with open(REG_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)

def _norm(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "default"

def _current(reg: Dict):
    return reg.get("_current")

# exec helper
def _run(cmd: list[str]):
    log.info("exec: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT))

# commands
def cmd_index(reg: Dict, rest: str):
    parts = shlex.split(rest)
    if not parts:
        return print("usage: /index \"folder\" [name]")
    root = parts[0]
    name = _norm(parts[1]) if len(parts) > 1 else _norm(Path(root).name)
    idx = str(DATA_DIR / f"{name}.faiss")
    meta = str(DATA_DIR / f"{name}.pkl")
    _run([PYBIN, PHOTSEARCH, "index", "--root", root, "--index", idx, "--meta", meta])
    reg[name] = {"index": idx, "meta": meta, "root": root, "created": datetime.now(timezone.utc).isoformat()}
    reg["_current"] = name
    _save_reg(reg)
    print(f"[ok] indexed as {name}")

def cmd_find(reg: Dict, rest: str):
    parts = shlex.split(rest)
    k = 12
    if parts and parts[-1].isdigit():
        k = int(parts.pop())
    prompt = " ".join(parts)
    cur = _current(reg)
    if not cur:
        return print("no current set – /use name")
    ent = reg[cur]
    _run([PYBIN, PHOTSEARCH, "query",
          "--index", ent["index"],
          "--meta", ent["meta"],
          "--text", prompt,
          "--k", str(k)])

def cmd_group(reg: Dict, rest: str):
    parts = shlex.split(rest)
    if len(parts) < 3:
        return print("usage: /group name \"out\" <k|-1> [TRUE|FALSE]")
    name, out_dir, k_str = parts[:3]
    save = len(parts) > 3 and parts[3].lower().startswith("t")
    ent = reg.get(name)
    if not ent:
        return print("unknown photoset")
    _run([PYBIN, str(ROOT / "style_grouper.py"),
          "--root", ent["root"],
          "--out", out_dir,
          "--clusters", k_str,
          "--save-rest", str(save)])

def cmd_duplicates(reg: Dict, rest: str):
    parts = shlex.split(rest)
    if not parts:
        return print("usage: /duplicates \"dest\" [hamming]")
    dest = parts[0]
    ham = parts[1] if len(parts) > 1 else "8"
    cur = _current(reg)
    if not cur:
        return print("no current set – /use name")
    _run([PYBIN, ARCHIVED, "duplicates", dest, "--hamming", ham])

def cmd_inst(reg: Dict, rest: str):
    parts = shlex.split(rest)
    if len(parts) < 2:
        return print("usage: /inst \"preset.json\" \"dest\" [count]")
    preset, dest = parts[0], parts[1]
    count = parts[2] if len(parts) > 2 else "12"
    cur = _current(reg)
    if not cur:
        return print("no current set – /use name")
    _run([PYBIN, ARCHIVED, "inst", preset, dest, "--count", count])

# interactive loop
def main():
    reg = _ensure_reg()
    print(HELP)
    while True:
        try:
            line = input(" > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/index"):
            cmd_index(reg, line[6:].strip()); continue
        if line.startswith("/find"):
            cmd_find(reg, line[5:].strip()); continue
        if line.startswith("/group"):
            cmd_group(reg, line[6:].strip()); continue
        if line.startswith("/duplicates"):
            cmd_duplicates(reg, line[11:].strip()); continue
        if line.startswith("/inst"):
            cmd_inst(reg, line[5:].strip()); continue
        if line.startswith("/use"):
            name = _norm(line[4:].strip())
            if name in reg:
                reg["_current"] = name; _save_reg(reg); print("current=", name)
            else:
                print("unknown name")
            continue
        if line == "/list":
            for n in reg:
                mark = "*" if n == _current(reg) else " "
                print(mark, n)
            continue
        if line == "/current":
            print(_current(reg)); continue
        if line in {"/exit", "/quit"}:
            break
        print("unknown command\n" + HELP)

if __name__ == "__main__":
    main()
