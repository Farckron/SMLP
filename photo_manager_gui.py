#!/usr/bin/env python3
"""
Photo Manager GUI — integrates photosearch + terminal features
Version: 0.5.2  (20 Aug 2025)

Run:
    python photo_manager_gui.py
"""
from __future__ import annotations

import sys
import json
import shlex
import subprocess
from pathlib import Path
from typing import List

from PyQt5 import QtCore, QtGui, QtWidgets  # type: ignore

# --------------------------------------------------------------------- paths
ROOT      = Path(__file__).resolve().parent
PYBIN     = sys.executable or "python"
TERMINAL  = ROOT / "terminal.py"          # CLI backend
DATA_DIR  = ROOT / "data"

# --------------------------------------------------------------------- style
SEA_BG    = "#0f3d5e"     # deep blue
TXT       = "#e8f4fa"     # light text

# ---------------------------------------------------------------------------#
class StreamWorker(QtCore.QThread):
    """
    Run a subprocess in its own QThread and stream stdout → Qt signal.
    """

    line_emitted  = QtCore.pyqtSignal(str)
    finished_clean = QtCore.pyqtSignal()

    def __init__(self, cmd: List[str]):
        # … inside __init__():
        dup_frame.setVisible(False)   # legacy feature archived
        inst_frame.setVisible(False)  # legacy feature archived
        super().__init__()
        self.cmd = cmd

    def run(self):  # type: ignore[override]
        proc = subprocess.Popen(
            self.cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,                 # line-buffered
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.line_emitted.emit(line.rstrip("\n"))
        proc.wait()
        self.line_emitted.emit(f"[done] exit code {proc.returncode}")
        self.finished_clean.emit()


# ---------------------------------------------------------------------------#
class MainWindow(QtWidgets.QWidget):
    """
    Main PyQt5 window housing all widgets and actions.
    """

    def __init__(self):
        super().__init__()

        self.workers: List[StreamWorker] = []     # keep references

        # ------------- basic window -------------
        self.setWindowTitle("Photo Manager GUI")
        self.setMinimumSize(880, 620)
        self.setStyleSheet(
            f"background:{SEA_BG}; color:{TXT}; font-family:Segoe UI; font-size:10pt;"
        )

        # ------------- registry & presets -------------
        self.registry = self.load_registry()
        self.presets  = self.scan_presets()

        # ------------- main layout -------------
        layout = QtWidgets.QHBoxLayout(self)
        left   = QtWidgets.QVBoxLayout()
        right  = QtWidgets.QVBoxLayout()
        layout.addLayout(left,  0)
        layout.addLayout(right, 1)

        # ================= LEFT PANE =================
        # ---- Photoset selector
        top_bar  = QtWidgets.QHBoxLayout()
        self.cb_sets = QtWidgets.QComboBox()
        self.cb_sets.setStyleSheet("min-width:180px;")
        self.refresh_sets()
        self.cb_sets.currentIndexChanged.connect(self.change_set)
        top_bar.addWidget(QtWidgets.QLabel("Photoset:"))
        top_bar.addWidget(self.cb_sets, 1)
        left.addLayout(top_bar)

        # ---- Index frame
        idx_frame           = self.frame("Index Photoset")
        self.le_idx_root    = QtWidgets.QLineEdit()
        btn_browse_idx      = QtWidgets.QPushButton("Browse…")
        btn_browse_idx.clicked.connect(lambda: self.choose_dir(self.le_idx_root))
        h_idx1              = self.hbox([self.le_idx_root, btn_browse_idx])
        self.le_idx_name    = QtWidgets.QLineEdit()
        self.le_idx_name.setPlaceholderText("name (optional)")
        btn_idx             = QtWidgets.QPushButton("Index")
        btn_idx.clicked.connect(self.action_index)
        idx_frame.body.addLayout(h_idx1)
        idx_frame.body.addWidget(self.le_idx_name)
        idx_frame.body.addWidget(btn_idx)
        left.addWidget(idx_frame)

        # ---- Search frame
        search_frame        = self.frame("Search")
        self.le_query       = QtWidgets.QLineEdit()
        spin_k              = QtWidgets.QSpinBox()
        spin_k.setRange(1, 100)
        spin_k.setValue(10)
        btn_search          = QtWidgets.QPushButton("Find")
        btn_search.clicked.connect(lambda: self.action_search(spin_k.value()))
        search_frame.body.addWidget(self.le_query)
        search_frame.body.addLayout(self.hbox([QtWidgets.QLabel("Top-k:"), spin_k, btn_search]))
        left.addWidget(search_frame)

        # ---- Duplicate frame
        dup_frame           = self.frame("Remove Duplicates")
        self.le_dup_dest    = QtWidgets.QLineEdit()
        btn_dup_dest        = QtWidgets.QPushButton("Browse…")
        btn_dup_dest.clicked.connect(lambda: self.choose_dir(self.le_dup_dest, save=True))
        spin_h              = QtWidgets.QSpinBox()
        spin_h.setRange(0, 64)
        spin_h.setValue(8)
        btn_dupes           = QtWidgets.QPushButton("Run")
        btn_dupes.clicked.connect(lambda: self.action_dupes(spin_h.value()))
        dup_frame.body.addLayout(self.hbox([self.le_dup_dest, btn_dup_dest]))
        dup_frame.body.addLayout(self.hbox([QtWidgets.QLabel("Hamming ≤"), spin_h, btn_dupes]))
        left.addWidget(dup_frame)

        # ---- Instagram picker frame
        inst_frame          = self.frame("Instagram Picker")
        self.cb_preset      = QtWidgets.QComboBox()
        self.cb_preset.addItems(self.presets)
        btn_reload_presets  = QtWidgets.QPushButton("⟳")
        btn_reload_presets.clicked.connect(self.reload_presets)
        inst_frame.body.addLayout(self.hbox([self.cb_preset, btn_reload_presets]))
        self.le_inst_dest   = QtWidgets.QLineEdit()
        btn_inst_dest       = QtWidgets.QPushButton("Browse…")
        btn_inst_dest.clicked.connect(lambda: self.choose_dir(self.le_inst_dest, save=True))
        spin_count          = QtWidgets.QSpinBox()
        spin_count.setRange(1, 100)
        spin_count.setValue(12)
        btn_pick            = QtWidgets.QPushButton("Pick")
        btn_pick.clicked.connect(lambda: self.action_inst(spin_count.value()))
        inst_frame.body.addLayout(self.hbox([self.le_inst_dest, btn_inst_dest]))
        inst_frame.body.addLayout(self.hbox([QtWidgets.QLabel("Count:"), spin_count, btn_pick]))
        left.addWidget(inst_frame)
        left.addStretch()

        # ================= RIGHT PANE =================
        right.addWidget(QtWidgets.QLabel("<b>Output Log</b>"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("background:#062235; color:#e8f4fa;")
        right.addWidget(self.log, 1)

    # ------------------------------------------------------------------ helpers
    def frame(self, title: str):
        f         = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.StyledPanel)
        v         = QtWidgets.QVBoxLayout(f)
        lab       = QtWidgets.QLabel(f"<b>{title}</b>")
        lab.setStyleSheet(f"color:{TXT}")
        v.addWidget(lab)
        body      = QtWidgets.QVBoxLayout()
        v.addLayout(body)
        f.body    = body      # type: ignore
        return f

    def hbox(self, widgets):
        h = QtWidgets.QHBoxLayout()
        for w in widgets:
            h.addWidget(w, 1 if isinstance(w, (QtWidgets.QLineEdit, QtWidgets.QComboBox)) else 0)
        return h

    def choose_dir(self, lineedit: QtWidgets.QLineEdit, save: bool = False):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            lineedit.setText(path)

    # ---------------------------------------------------------------- registry
    def load_registry(self):
        try:
            with open(DATA_DIR / "registry.json", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def refresh_sets(self):
        names = sorted(k for k in self.registry if not k.startswith("_"))
        current = self.registry.get("_current")
        self.cb_sets.blockSignals(True)
        self.cb_sets.clear()
        self.cb_sets.addItems(names)
        if current in names:
            self.cb_sets.setCurrentText(current)
        self.cb_sets.blockSignals(False)

    def scan_presets(self):
        return sorted(
            str(p)
            for p in DATA_DIR.glob("*.json")
            if not p.name.lower().startswith("registry")
        )

    def reload_presets(self):
        self.presets = self.scan_presets()
        self.cb_preset.clear()
        self.cb_preset.addItems(self.presets)

    # -------------------------------------------------------------- backend glue
    def run_backend(self, cmd: List[str]):
        """
        Spawn TERMINAL.py with -u (unbuffered) so prints flush line-by-line.
        """
        full_cmd = [PYBIN, "-u"] + cmd
        self.log.appendPlainText("\n> " + " ".join(shlex.quote(c) for c in full_cmd))

        worker = StreamWorker(full_cmd)
        worker.line_emitted.connect(self.log.appendPlainText)
        worker.finished_clean.connect(lambda w=worker: self.workers.remove(w))
        self.workers.append(worker)
        worker.start()

    # -------------------------------------------------------------- actions
    def action_index(self):
        root = self.le_idx_root.text().strip()
        if not root:
            self.log.appendPlainText("[error] choose folder")
            return
        name = self.le_idx_name.text().strip()
        cmd  = [str(TERMINAL), "/index", root] + ([name] if name else [])
        self.run_backend(cmd)

    def action_search(self, k: int):
        q = self.le_query.text().strip()
        if not q:
            self.log.appendPlainText("[error] query required")
            return
        self.run_backend([str(TERMINAL), "/find", q, str(k)])

    def action_dupes(self, hamming: int):
        dest = self.le_dup_dest.text().strip()
        if not dest:
            self.log.appendPlainText("[error] destination required")
            return
        self.run_backend([str(TERMINAL), "/duplicates", dest, str(hamming)])

    def action_inst(self, count: int):
        preset = self.cb_preset.currentText().strip()
        dest   = self.le_inst_dest.text().strip()
        if not preset or not dest:
            self.log.appendPlainText("[error] preset and dest required")
            return
        self.run_backend([str(TERMINAL), "/inst", preset, dest, str(count)])

    # -------------------------------------------------------------- set switch
    def change_set(self):
        name = self.cb_sets.currentText().strip()
        if name:
            self.run_backend([str(TERMINAL), "/use", name])


# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
