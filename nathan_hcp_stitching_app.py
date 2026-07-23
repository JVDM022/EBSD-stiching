#!/usr/bin/env python3
"""Browser UI for Nathan's HCP EBSD stitching notebook.

This intentionally uses only Python's standard-library web server for the UI,
because some local Python installs do not include Tkinter.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import traceback
import types
import webbrowser
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
PIPELINE_NOTEBOOK = ROOT / "nathan's HCP test.ipynb"
DEFAULT_OUTPUT = ROOT / "nathan_hcp_4x4_stitch_output"

STATE_LOCK = threading.Lock()
STATE = {
    "running": False,
    "status": "Ready",
    "log": "",
    "output": str(DEFAULT_OUTPUT),
}


class LogWriter:
    def write(self, text: str) -> int:
        if text:
            append_log(text)
        return len(text)

    def flush(self) -> None:
        pass


def append_log(text: str) -> None:
    with STATE_LOCK:
        STATE["log"] += text


def set_state(**kwargs) -> None:
    with STATE_LOCK:
        STATE.update(kwargs)


def get_state() -> dict:
    with STATE_LOCK:
        return dict(STATE)


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


def build_args(form: dict[str, str]) -> SimpleNamespace:
    rows = parse_int("Rows", form.get("rows", "4"))
    cols = parse_int("Cols", form.get("cols", "4"))
    return SimpleNamespace(
        tiles=form.get("tiles", str(ROOT / "nathan_hcp_4x4")),
        output=form.get("output", str(DEFAULT_OUTPUT)),
        overlap_fraction=parse_float("Overlap fraction", form.get("overlap", "0.20")),
        search_radius=parse_int("Search radius", form.get("search", "4")),
        orientation_samples=parse_int("Orientation samples", form.get("orientation_samples", "120")),
        max_tiles=rows * cols,
        max_pairs=0,
        use_ml_training=True,
        self_supervised_train_fraction=0.70,
        min_accept_score=parse_float("Min accept score", form.get("min_score", "0.60")),
        max_accept_misorientation=parse_float("Max HCP misorientation", form.get("max_mis", "10.0")),
        min_valid_fraction=parse_float("Min valid fraction", form.get("min_valid", "0.50")),
        parent_ang=None,
        parent_tilt_degrees=None,
    )


def write_no_parent_summary(args: SimpleNamespace) -> None:
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
    append_log("\nNo-parent summary\n")
    append_log(json.dumps(summary, indent=2) + "\n")


def run_pipeline(args: SimpleNamespace) -> None:
    try:
        set_state(running=True, status="Running", output=str(args.output))
        append_log("Starting HCP stitch run...\n")
        env = load_pipeline()
        with redirect_stdout(LogWriter()):
            env["run"](args)
        write_no_parent_summary(args)
        append_log("\nDone. Use the buttons above to open the output folder or stitched preview.\n")
        set_state(running=False, status="Done")
    except Exception:
        append_log("\nERROR\n")
        append_log(traceback.format_exc())
        set_state(running=False, status="Failed")


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Nathan HCP Stitching</title>
  <style>
    :root {
      color-scheme: light;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f6f8;
      color: #1f2933;
    }
    body { margin: 0; }
    main { max-width: 1120px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 18px; font-size: 28px; }
    .panel {
      background: #fff;
      border: 1px solid #d7dde5;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 16px;
    }
    .grid {
      display: grid;
      grid-template-columns: 170px minmax(0, 1fr) 150px minmax(0, 1fr);
      gap: 12px 14px;
      align-items: center;
    }
    label { font-size: 14px; color: #334155; }
    input {
      box-sizing: border-box;
      width: 100%;
      padding: 9px 10px;
      border: 1px solid #c7d0dc;
      border-radius: 6px;
      font-size: 14px;
      background: #fff;
    }
    .wide-label { grid-column: 1; }
    .wide-input { grid-column: 2 / 5; }
    .actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    button, a.button {
      border: 1px solid #1f6feb;
      background: #1f6feb;
      color: white;
      padding: 9px 13px;
      border-radius: 6px;
      font-size: 14px;
      cursor: pointer;
      text-decoration: none;
    }
    button.secondary, a.secondary { background: #fff; color: #1f6feb; }
    button:disabled { opacity: 0.5; cursor: default; }
    #status { margin-left: 8px; font-weight: 600; }
    pre {
      height: 360px;
      overflow: auto;
      background: #111827;
      color: #d1fae5;
      border-radius: 8px;
      padding: 14px;
      white-space: pre-wrap;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .hint { color: #5b6777; font-size: 13px; margin-top: 10px; }
    @media (max-width: 760px) {
      main { padding: 16px; }
      .grid { grid-template-columns: 1fr; }
      .wide-label, .wide-input { grid-column: auto; }
    }
  </style>
</head>
<body>
<main>
  <h1>Nathan HCP Stitching</h1>
  <form id="runForm" class="panel">
    <div class="grid">
      <label class="wide-label" for="tiles">Tile folder</label>
      <input class="wide-input" id="tiles" name="tiles" value="__TILES__">
      <label class="wide-label" for="output">Output folder</label>
      <input class="wide-input" id="output" name="output" value="__OUTPUT__">
      <label for="rows">Rows</label><input id="rows" name="rows" value="4">
      <label for="cols">Cols</label><input id="cols" name="cols" value="4">
      <label for="overlap">Overlap fraction</label><input id="overlap" name="overlap" value="0.20">
      <label for="search">Search radius</label><input id="search" name="search" value="4">
      <label for="orientation_samples">Orientation samples</label><input id="orientation_samples" name="orientation_samples" value="120">
      <label for="min_score">Min accept score</label><input id="min_score" name="min_score" value="0.60">
      <label for="max_mis">Max HCP misorientation</label><input id="max_mis" name="max_mis" value="10.0">
      <label for="min_valid">Min valid fraction</label><input id="min_valid" name="min_valid" value="0.50">
    </div>
    <p class="hint">This app expects 16 EBSD .ang tiles named like tile_r0_c0_clean.ang, or a tile_manifest.csv with tile_name,row,col.</p>
  </form>
  <section class="panel actions">
    <button id="runButton" type="button">Run stitching</button>
    <a class="button secondary" href="/open-output" target="_blank">Open output folder</a>
    <a class="button secondary" href="/open-preview" target="_blank">Open stitched preview</a>
    <span id="status">Ready</span>
  </section>
  <section class="panel">
    <pre id="log"></pre>
  </section>
</main>
<script>
const runButton = document.getElementById('runButton');
const statusText = document.getElementById('status');
const log = document.getElementById('log');

function formPayload() {
  const data = new FormData(document.getElementById('runForm'));
  return Object.fromEntries(data.entries());
}

async function poll() {
  try {
    const res = await fetch('/status');
    const data = await res.json();
    statusText.textContent = data.status;
    runButton.disabled = data.running;
    log.textContent = data.log || '';
    log.scrollTop = log.scrollHeight;
  } catch (err) {
    statusText.textContent = 'Disconnected';
  }
}

runButton.addEventListener('click', async () => {
  runButton.disabled = true;
  const res = await fetch('/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(formPayload())
  });
  const data = await res.json();
  if (!data.ok) {
    alert(data.error || 'Unable to start run');
  }
  poll();
});

setInterval(poll, 1000);
poll();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_html(
                HTML.replace("__TILES__", html_escape(str(ROOT / "nathan_hcp_4x4")))
                .replace("__OUTPUT__", html_escape(str(DEFAULT_OUTPUT)))
            )
        elif parsed.path == "/status":
            self.send_json(get_state())
        elif parsed.path == "/open-output":
            self.open_path(Path(get_state()["output"]))
        elif parsed.path == "/open-preview":
            out = Path(get_state()["output"])
            preview = out / "stitched_ipf_preview.png"
            if not preview.exists():
                preview = out / "stitched_iq.png"
            self.open_path(preview)
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/run":
            self.send_error(404)
            return

        if get_state()["running"]:
            self.send_json({"ok": False, "error": "A stitching run is already active."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            content_type = self.headers.get("Content-Type", "")
            if "application/json" in content_type:
                form = json.loads(body or "{}")
            else:
                form = {k: v[-1] for k, v in parse_qs(body).items()}
            args = build_args({k: str(v) for k, v in form.items()})
            tile_dir = Path(args.tiles)
            if not tile_dir.exists():
                raise FileNotFoundError(f"Tile folder does not exist: {tile_dir}")
            set_state(running=True, status="Starting", log="", output=str(args.output))
            thread = threading.Thread(target=run_pipeline, args=(args,), daemon=True)
            thread.start()
            self.send_json({"ok": True})
        except Exception as exc:
            set_state(running=False, status="Failed")
            self.send_json({"ok": False, "error": str(exc)})

    def open_path(self, path: Path) -> None:
        if path.exists():
            subprocess.run(["open", str(path)], check=False)
            self.send_html("<p>Opened. You can close this tab.</p>")
        else:
            self.send_html(f"<p>Path does not exist yet:<br>{html_escape(str(path))}</p>")

    def send_json(self, data: dict) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


def html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = server.server_address
    url = f"http://{host}:{port}/"
    print(f"Nathan HCP Stitching App is running at {url}")
    print("Keep this terminal window open while using the app.")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
