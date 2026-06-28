#!/usr/bin/env python3
"""Render maps from a stitched EDAX ANG file.

This is the Python translation of the MATLAB/MTEX verification script that did:
    ebsd = EBSD.load(angFile, 'convertEuler2SpatialReferenceFrame', 'setting 2')
    plot(ebsd, ebsd.orientations)

It reloads the exported ANG file, reconstructs the scan grid, and writes IQ, CI,
and cubic IPF-Z orientation maps from the file on disk. The orientation colors
are MTEX-like but not byte-identical to MTEX's ipfHSVKey interpolation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parent / ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANG_COLUMNS = [
    "phi1",
    "Phi",
    "phi2",
    "x",
    "y",
    "IQ",
    "CI",
    "phase",
    "SEM_signal",
    "fit",
]


def read_ang_numeric(path: Path) -> pd.DataFrame:
    numeric_rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                numeric_rows.append([float(part) for part in parts])
            except ValueError:
                continue
    if not numeric_rows:
        raise ValueError(f"No numeric ANG rows found in {path}")

    ncols = max(len(row) for row in numeric_rows)
    arr = np.full((len(numeric_rows), ncols), np.nan, dtype=float)
    for row_index, row in enumerate(numeric_rows):
        arr[row_index, : len(row)] = row

    columns = ANG_COLUMNS[:ncols] + [f"extra_{i}" for i in range(max(0, ncols - len(ANG_COLUMNS)))]
    df = pd.DataFrame(arr, columns=columns[:ncols])
    for col, default in [("IQ", 1.0), ("CI", 1.0), ("phase", 1)]:
        if col not in df:
            df[col] = default

    indexed = df["phase"].fillna(-1).ge(0)
    finite_eulers = df.loc[indexed, ["phi1", "Phi", "phi2"]].to_numpy(dtype=float)
    finite_eulers = finite_eulers[np.isfinite(finite_eulers)]
    if finite_eulers.size and np.nanpercentile(finite_eulers, 99.9) > 2 * np.pi + 1e-3:
        df.loc[indexed, ["phi1", "Phi", "phi2"]] = np.deg2rad(df.loc[indexed, ["phi1", "Phi", "phi2"]])
    return df


def gridify_ang(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    ys = np.sort(df["y"].dropna().unique())
    row_counts = df.groupby("y").size()
    width = int(row_counts.max())
    shape = (len(ys), width)
    arrays = {
        "phi1": np.full(shape, np.nan, dtype=float),
        "Phi": np.full(shape, np.nan, dtype=float),
        "phi2": np.full(shape, np.nan, dtype=float),
        "iq": np.full(shape, np.nan, dtype=float),
        "ci": np.full(shape, np.nan, dtype=float),
        "phase": np.full(shape, -1, dtype=int),
    }
    for yi, (_, group) in enumerate(df.sort_values(["y", "x"]).groupby("y", sort=True)):
        group = group.sort_values("x").head(width)
        n = len(group)
        arrays["phi1"][yi, :n] = group["phi1"].to_numpy()
        arrays["Phi"][yi, :n] = group["Phi"].to_numpy()
        arrays["phi2"][yi, :n] = group["phi2"].to_numpy()
        arrays["iq"][yi, :n] = group["IQ"].to_numpy()
        arrays["ci"][yi, :n] = group["CI"].to_numpy()
        arrays["phase"][yi, :n] = group["phase"].fillna(-1).to_numpy(dtype=int)
    arrays["valid"] = (
        (arrays["phase"] >= 0)
        & np.isfinite(arrays["phi1"])
        & np.isfinite(arrays["Phi"])
        & np.isfinite(arrays["phi2"])
        & np.isfinite(arrays["iq"])
    )
    return arrays


def euler_to_matrix(phi1: np.ndarray, Phi: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    out = np.empty(phi1.shape + (3, 3), dtype=float)
    out[..., 0, 0] = c1 * c2 - s1 * s2 * c
    out[..., 0, 1] = s1 * c2 + c1 * s2 * c
    out[..., 0, 2] = s2 * s
    out[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    out[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    out[..., 1, 2] = c2 * s
    out[..., 2, 0] = s1 * s
    out[..., 2, 1] = -c1 * s
    out[..., 2, 2] = c
    return out


def edax_reference_frame_correction(setting: int = 2) -> np.ndarray:
    corrections = {
        1: np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
        2: np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
        3: np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        4: np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
    }
    if setting not in corrections:
        raise ValueError(f"Unsupported EDAX reference-frame setting: {setting}")
    return corrections[setting]


def cubic_ipf_z_rgb(arrays: Dict[str, np.ndarray], edax_setting: int = 2) -> np.ndarray:
    rotations = euler_to_matrix(arrays["phi1"], arrays["Phi"], arrays["phi2"])
    correction = edax_reference_frame_correction(edax_setting)
    rotations = np.einsum("ij,...jk->...ik", correction, rotations)
    direction = np.abs(rotations[..., :, 2])
    fundamental = np.sort(direction, axis=2)[..., ::-1]
    h, k, l = np.moveaxis(fundamental, -1, 0)
    weights = np.stack((h - k, k - l, l), axis=-1)
    rgb = np.sqrt(np.clip(weights, 0.0, None))
    denom = np.max(rgb, axis=2, keepdims=True)
    rgb = rgb / np.where(denom > 1e-12, denom, 1.0)
    rgb[~arrays["valid"]] = np.nan
    return np.clip(rgb, 0.0, 1.0)


def save_image(path: Path, arr: np.ndarray, cmap: str = "gray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    if arr.ndim == 3:
        ax.imshow(np.nan_to_num(arr, nan=1.0), origin="upper")
    else:
        ax.imshow(arr, cmap=cmap, origin="upper")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_ang(ang_path: Path, output_dir: Path, prefix: str = "stitched_ang_python", edax_setting: int = 2) -> dict:
    df = read_ang_numeric(ang_path)
    arrays = gridify_ang(df)
    ipf_rgb = cubic_ipf_z_rgb(arrays, edax_setting=edax_setting)
    files = {
        "orientation": f"{prefix}_orientations.png",
        "iq": f"{prefix}_iq.png",
        "ci": f"{prefix}_ci.png",
    }
    save_image(output_dir / files["orientation"], ipf_rgb)
    save_image(output_dir / files["iq"], arrays["iq"], cmap="gray")
    save_image(output_dir / files["ci"], arrays["ci"], cmap="viridis")
    summary = {
        "source_ang": str(ang_path),
        "grid_shape": [int(arrays["iq"].shape[0]), int(arrays["iq"].shape[1])],
        "numeric_rows": int(len(df)),
        "indexed_points": int(arrays["valid"].sum()),
        "reference_frame": f"MTEX convertEuler2SpatialReferenceFrame setting {edax_setting}",
        "files": files,
    }
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render orientation, IQ, and CI maps from a stitched ANG file.")
    parser.add_argument("ang", nargs="?", default="stitched_map.ang", help="ANG file to render")
    parser.add_argument("--output", default=None, help="Output directory; defaults to the ANG file directory")
    parser.add_argument("--prefix", default="stitched_ang_python", help="Output filename prefix")
    parser.add_argument("--edax-setting", type=int, default=2, choices=[1, 2, 3, 4], help="EDAX reference-frame setting to match MTEX conversion")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    ang_path = Path(args.ang)
    if not ang_path.is_absolute():
        ang_path = script_dir / ang_path
    output_dir = Path(args.output) if args.output else ang_path.parent
    summary = render_ang(ang_path, output_dir, args.prefix, edax_setting=args.edax_setting)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
