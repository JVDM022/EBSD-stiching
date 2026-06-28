#!/usr/bin/env python3
"""Small local UI for Nathan's HCP EBSD stitching notebook."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import traceback
import types
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent
PIPELINE_NOTEBOOK = ROOT / "nathan's HCP test.ipynb"


class QueueWriter:
    def __init__(self, log_queue: queue.Queue[str]) -> None:
        self.log_queue = log_queue

    def write(self, text: str) -> int:
        if text:
            self.log_queue.put(text)
        return len(text)

    def flush(self) -> None:
        pass


def load_pipeline() -> dict:
    if not PIPELINE_NOTEBOOK.exists():
        raise FileNotFoundError(f"Missing notebook backend: {PIPELINE_NOTEBOOK}")

    notebook = json.loads(PIPELINE_NOTEBOOK.read_text(encoding="utf-8"))
    module_name = "__nathan_hcp_pipeline__"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    env: dict = module.__dict__
    for index, cell in enumerate(notebook["cells"]):
        if index >= 25:
            break
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.lstrip().startswith(("!", "%")):
            continue
        exec(compile(source, f"{PIPELINE_NOTEBOOK.name}:cell{index}", "exec"), env)
    return env


def parse_float(name: str, value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


class NathanHcpApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Nathan HCP Stitching")
        self.geometry("980x720")
        self.minsize(860, 620)
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.output_dir = ROOT / "nathan_hcp_4x4_stitch_output"

        self.vars = {
            "tiles": tk.StringVar(value=str(ROOT / "nathan_hcp_4x4")),
            "output": tk.StringVar(value=str(self.output_dir)),
            "rows": tk.StringVar(value="4"),
            "cols": tk.StringVar(value="4"),
            "overlap": tk.StringVar(value="0.20"),
            "search": tk.StringVar(value="4"),
            "orientation_samples": tk.StringVar(value="120"),
            "min_score": tk.StringVar(value="0.60"),
            "max_mis": tk.StringVar(value="10.0"),
            "min_valid": tk.StringVar(value="0.50"),
        }

        self._build_ui()
        self.after(100, self._drain_log)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        title = ttk.Label(self, text="Nathan HCP Stitching", font=("Helvetica", 20, "bold"))
        title.grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))

        form = ttk.Frame(self, padding=(18, 8, 18, 8))
        form.grid(row=1, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(4, weight=1)

        self._path_row(form, 0, "Tile folder", "tiles", self._browse_tiles)
        self._path_row(form, 1, "Output folder", "output", self._browse_output)

        fields = [
            ("Rows", "rows"),
            ("Cols", "cols"),
            ("Overlap fraction", "overlap"),
            ("Search radius", "search"),
            ("Orientation samples", "orientation_samples"),
            ("Min accept score", "min_score"),
            ("Max HCP misorientation", "max_mis"),
            ("Min valid fraction", "min_valid"),
        ]
        for offset, (label, key) in enumerate(fields):
            row = 2 + offset // 2
            col = 0 if offset % 2 == 0 else 3
            ttk.Label(form, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=6)
            ttk.Entry(form, textvariable=self.vars[key], width=18).grid(
                row=row, column=col + 1, sticky="ew", padx=(0, 18), pady=6
            )

        actions = ttk.Frame(self, padding=(18, 0, 18, 8))
        actions.grid(row=2, column=0, sticky="new")
        self.run_button = ttk.Button(actions, text="Run stitching", command=self._start_run)
        self.run_button.grid(row=0, column=0, padx=(0, 8), pady=8)
        ttk.Button(actions, text="Open output folder", command=self._open_output).grid(
            row=0, column=1, padx=(0, 8), pady=8
        )
        ttk.Button(actions, text="Open stitched preview", command=self._open_preview).grid(
            row=0, column=2, padx=(0, 8), pady=8
        )
        self.status = ttk.Label(actions, text="Ready")
        self.status.grid(row=0, column=3, sticky="w", padx=(12, 0))

        log_frame = ttk.Frame(self, padding=(18, 4, 18, 18))
        log_frame.grid(row=3, column=0, sticky="nsew")
        self.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", height=22)
        scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _path_row(self, parent: ttk.Frame, row: int, label: str, key: str, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Entry(parent, textvariable=self.vars[key]).grid(
            row=row, column=1, columnspan=4, sticky="ew", padx=(0, 8), pady=6
        )
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=5, sticky="e", pady=6)

    def _browse_tiles(self) -> None:
        path = filedialog.askdirectory(initialdir=ROOT)
        if path:
            self.vars["tiles"].set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(initialdir=ROOT)
        if path:
            self.vars["output"].set(path)

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            rows = parse_int("Rows", self.vars["rows"].get())
            cols = parse_int("Cols", self.vars["cols"].get())
            args = SimpleNamespace(
                tiles=self.vars["tiles"].get(),
                output=self.vars["output"].get(),
                overlap_fraction=parse_float("Overlap fraction", self.vars["overlap"].get()),
                search_radius=parse_int("Search radius", self.vars["search"].get()),
                orientation_samples=parse_int("Orientation samples", self.vars["orientation_samples"].get()),
                max_tiles=rows * cols,
                max_pairs=0,
                use_ml_training=True,
                self_supervised_train_fraction=0.70,
                min_accept_score=parse_float("Min accept score", self.vars["min_score"].get()),
                max_accept_misorientation=parse_float("Max HCP misorientation", self.vars["max_mis"].get()),
                min_valid_fraction=parse_float("Min valid fraction", self.vars["min_valid"].get()),
                parent_ang=None,
                parent_tilt_degrees=None,
            )
        except Exception as exc:
            messagebox.showerror("Invalid setting", str(exc))
            return

        tile_dir = Path(args.tiles)
        if not tile_dir.exists():
            messagebox.showerror("Missing tile folder", f"Tile folder does not exist:\n{tile_dir}")
            return

        self.output_dir = Path(args.output)
        self.log.delete("1.0", "end")
        self._append_log("Starting HCP stitch run...\n")
        self.run_button.configure(state="disabled")
        self.status.configure(text="Running")
        self.worker = threading.Thread(target=self._run_pipeline, args=(args,), daemon=True)
        self.worker.start()

    def _run_pipeline(self, args: SimpleNamespace) -> None:
        try:
            env = load_pipeline()
            writer = QueueWriter(self.log_queue)
            with redirect_stdout(writer):
                env["run"](args)
            self._write_no_parent_summary(args)
            self.log_queue.put("\nDone. Use the buttons above to open the output folder or stitched preview.\n")
            self.log_queue.put("__STATUS__:Done")
        except Exception:
            self.log_queue.put("\nERROR\n")
            self.log_queue.put(traceback.format_exc())
            self.log_queue.put("__STATUS__:Failed")

    def _write_no_parent_summary(self, args: SimpleNamespace) -> None:
        import pandas as pd

        out = Path(args.output)
        selected_path = out / "selected_pair_shifts.csv"
        origins_path = out / "tile_origins.csv"
        if not selected_path.exists() or not origins_path.exists():
            return

        selected = pd.read_csv(selected_path)
        origins = pd.read_csv(origins_path)
        summary = {
            "tiles_folder": str(Path(args.tiles).resolve()),
            "output_dir": str(out.resolve()),
            "parent_ang": None,
            "n_selected_pairs": int(len(selected)),
            "n_accepted_seams": int(selected["accepted"].sum()) if "accepted" in selected else None,
            "n_tiles_placed": int(len(origins)),
            "median_final_score": float(selected["final_score"].median()) if "final_score" in selected else None,
            "median_hcp_misorientation_deg": (
                float(selected["median_misorientation_deg"].median())
                if "median_misorientation_deg" in selected
                else None
            ),
            "note": "No parent ANG was supplied; these are internal seam/layout diagnostics.",
        }
        validation_dir = out / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        (validation_dir / "no_parent_app_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        self.log_queue.put("\nNo-parent summary\n")
        self.log_queue.put(json.dumps(summary, indent=2) + "\n")

    def _drain_log(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if item.startswith("__STATUS__:"):
                    status = item.split(":", 1)[1]
                    self.status.configure(text=status)
                    self.run_button.configure(state="normal")
                else:
                    self._append_log(item)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _append_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def _open_output(self) -> None:
        path = Path(self.vars["output"].get())
        if path.exists():
            subprocess.run(["open", str(path)], check=False)
        else:
            messagebox.showinfo("Output folder", f"Output folder does not exist yet:\n{path}")

    def _open_preview(self) -> None:
        preview = Path(self.vars["output"].get()) / "stitched_ipf_preview.png"
        if not preview.exists():
            preview = Path(self.vars["output"].get()) / "stitched_iq.png"
        if preview.exists():
            subprocess.run(["open", str(preview)], check=False)
        else:
            messagebox.showinfo("Preview", "No stitched preview exists yet. Run stitching first.")


if __name__ == "__main__":
    NathanHcpApp().mainloop()
