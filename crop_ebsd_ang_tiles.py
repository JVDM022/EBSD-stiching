#!/usr/bin/env python3
"""
Crop one EBSD .ang map into overlapping tiles, with optional y-tilt correction
and optional artificial coordinate distortions.

This script does not require MTEX. It preserves all numeric columns and only
modifies the configured X/Y coordinate columns when correction or distortion is
applied.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config section
# ---------------------------------------------------------------------------

ANG_ROOT = Path("EBSD stiching dev dataset")
INPUT_ANG = ANG_ROOT / "5%coldrolled_30min_975C_65mag2.ang"
OUTPUT_DIR = "Cropped"

APPLY_TILT_CORRECTION = True
TILT_DEGREES = 70.0

TILE_WIDTH = 200.0        # same units as .ang X coordinate
TILE_HEIGHT = 200.0       # same units as .ang Y coordinate
OVERLAP_FRACTION = 0.20

X_COL = 3
Y_COL = 4

DISTORT_EVERY_OTHER_TILE = True
SAVE_UNDISTORTED_TILES = True
RANDOM_SEED = 42

# Artificial distortion settings.
#
# Kikuchipy's detector PC fitting uses projective transforms for map grids
# whose parallel lines may become non-parallel.  Use the same transform family
# here to synthesize a rectangle-to-trapezoid coordinate distortion.
DISTORTION_MODEL = "projective_trapezoid"  # "projective_trapezoid" or "affine_drift"
TRAPEZOID_TOP_WIDTH_SCALE = 0.92
TRAPEZOID_BOTTOM_WIDTH_SCALE = 1.08
TRAPEZOID_TOP_SHIFT_X = 4.0
TRAPEZOID_BOTTOM_SHIFT_X = -4.0
TRAPEZOID_TOP_SHIFT_Y = 0.0
TRAPEZOID_BOTTOM_SHIFT_Y = 0.0

# Legacy affine/drift distortion settings, used when DISTORTION_MODEL is
# "affine_drift".
TRANSLATION_X = 5.0
TRANSLATION_Y = -3.0
ROTATION_DEG = 1.5
SHEAR_X = 0.02
SHEAR_Y = 0.00
SCALE_X = 1.00
SCALE_Y = 1.00
ADD_NONLINEAR_DRIFT = True
DRIFT_AMPLITUDE = 2.0

MIN_POINTS_PER_TILE = 10


MANIFEST_FIELDS = [
    "tile_name",
    "row",
    "col",
    "xmin",
    "xmax",
    "ymin",
    "ymax",
    "n_points",
    "tilt_corrected",
    "distorted",
    "distortion_model",
    "trapezoid_top_width_scale",
    "trapezoid_bottom_width_scale",
    "trapezoid_top_shift_x",
    "trapezoid_bottom_shift_x",
    "trapezoid_top_shift_y",
    "trapezoid_bottom_shift_y",
    "translation_x",
    "translation_y",
    "rotation_deg",
    "shear_x",
    "shear_y",
    "scale_x",
    "scale_y",
    "nonlinear_drift",
]


def _is_numeric_row(line: str) -> bool:
    """Return True when a line can be parsed as a whitespace-delimited row."""
    stripped = line.strip()
    if not stripped:
        return False

    parts = stripped.split()
    if not parts:
        return False

    try:
        [float(part) for part in parts]
    except ValueError:
        return False
    return True


def _configure_scientific_package_cache() -> None:
    """Keep optional scientific package cache/config writes inside the project."""
    cache_root = Path(".cache")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))


def _read_ang_text(path: Path) -> Tuple[List[str], List[str]]:
    """Read headers and numeric rows, tolerating non-comment preamble lines."""
    header_lines: List[str] = []
    numeric_lines: List[str] = []
    found_numeric_data = False
    expected_cols = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("#") and not found_numeric_data:
                header_lines.append(line)
                continue

            if not line.strip():
                continue

            if _is_numeric_row(line):
                found_numeric_data = True
                cols = len(line.split())
                if expected_cols is None:
                    expected_cols = cols
                elif cols != expected_cols:
                    raise ValueError(
                        f"Inconsistent numeric column count at line {line_number}: "
                        f"expected {expected_cols}, found {cols}"
                    )
                numeric_lines.append(line)
            elif not found_numeric_data:
                # Common .ang files use # headers, but tolerate stray preamble
                # lines before the first numeric row by keeping them as headers.
                header_lines.append("# " + line if not line.startswith("#") else line)
            else:
                raise ValueError(
                    f"Non-numeric row found after numeric data starts at line {line_number}"
                )

    if not numeric_lines:
        raise ValueError(f"No numeric data rows were found in: {path}")

    return header_lines, numeric_lines


def _numeric_lines_to_array(numeric_lines: Sequence[str]) -> np.ndarray:
    data = np.loadtxt(numeric_lines, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _read_ang_with_orix(path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Read an .ang file with orix when available.

    kikuchipy depends on orix, while orix owns the ANG crystal-map reader. We
    still return the raw numeric table because this script deliberately writes
    cropped ANG-like files with every original numeric column preserved.
    """
    _configure_scientific_package_cache()

    try:
        import orix.io
    except Exception as exc:
        raise ImportError("orix is not available for ANG loading") from exc

    header_lines, numeric_lines = _read_ang_text(path)
    cache_root = Path(".cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    temp_path = None

    # Let orix validate vendor/header/column conventions. Its returned
    # CrystalMap is not used for writing because that would normalize the file.
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".ang",
            prefix="orix_",
            dir=cache_root,
            delete=False,
            encoding="utf-8",
        ) as handle:
            temp_path = Path(handle.name)
            handle.writelines(header_lines)
            handle.writelines(numeric_lines)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            orix.io.load(temp_path, autogen_names=False)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    data = _numeric_lines_to_array(numeric_lines)

    return header_lines, data


def read_ang(path: str | os.PathLike[str]) -> Tuple[List[str], np.ndarray]:
    """
    Read an .ang file.

    Prefer the orix ANG reader, which is installed with kikuchipy and handles
    vendor-specific ANG conventions. Fall back to the local numeric parser for
    unusual files or environments where orix is unavailable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input .ang file does not exist: {path}")

    try:
        header_lines, data = _read_ang_with_orix(path)
        print("Loaded ANG with orix reader")
        return header_lines, data
    except Exception as exc:
        print(f"orix ANG reader unavailable or failed ({exc}); using fallback parser")

    header_lines, numeric_lines = _read_ang_text(path)
    return header_lines, _numeric_lines_to_array(numeric_lines)


def write_ang(
    path: str | os.PathLike[str],
    header_lines: Sequence[str],
    data_array: np.ndarray,
) -> None:
    """Write an .ang-like text file with preserved header and all columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for line in header_lines:
            handle.write(line if line.endswith("\n") else line + "\n")

        for row in data_array:
            handle.write(" ".join(f"{value:.10g}" for value in row) + "\n")


def apply_y_tilt_correction(
    data: np.ndarray,
    y_col: int,
    tilt_degrees: float,
) -> np.ndarray:
    """
    Stretch Y by 1/cos(tilt) to undo the projected EBSD map shortening.

    Only the Y coordinate column is replaced; all other columns are preserved.
    """
    corrected = data.copy()
    factor = 1.0 / math.cos(math.radians(tilt_degrees))
    corrected[:, y_col] = corrected[:, y_col] * factor
    print(f"Applied y-tilt correction: factor = {factor:.8g}")
    return corrected


def crop_tile(
    data: np.ndarray,
    x_col: int,
    y_col: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> np.ndarray:
    """Return points whose corrected coordinates fall inside one tile."""
    x = data[:, x_col]
    y = data[:, y_col]
    mask = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    return data[mask].copy()


def distort_coordinates(
    data: np.ndarray,
    x_col: int,
    y_col: int,
    params: Dict[str, float | bool | str],
) -> np.ndarray:
    """
    Apply a small synthetic coordinate distortion to one tile.

    The default model is a projective rectangle-to-trapezoid warp. This is the
    inverse of the usual trapezoid correction: correction maps a measured
    trapezoid back to a rectangle, while this function maps the clean rectangle
    into a measured trapezoid. The legacy affine/drift model remains available
    for older experiments.
    """
    if params.get("distortion_model") == "projective_trapezoid":
        return _distort_coordinates_projective_trapezoid(data, x_col, y_col, params)
    return _distort_coordinates_affine_drift(data, x_col, y_col, params)


def _distort_coordinates_affine_drift(
    data: np.ndarray,
    x_col: int,
    y_col: int,
    params: Dict[str, float | bool | str],
) -> np.ndarray:
    """Apply the previous centered affine plus sinusoidal drift model."""
    distorted = data.copy()
    x = distorted[:, x_col].copy()
    y = distorted[:, y_col].copy()

    x_centroid = float(np.mean(x))
    y_centroid = float(np.mean(y))
    x_centered = x - x_centroid
    y_centered = y - y_centroid

    scale_x = float(params["scale_x"])
    scale_y = float(params["scale_y"])
    shear_x = float(params["shear_x"])
    shear_y = float(params["shear_y"])
    translation_x = float(params["translation_x"])
    translation_y = float(params["translation_y"])

    # Affine part: scale, shear, and translation in centered coordinates.
    x_affine = scale_x * x_centered + shear_x * y_centered + translation_x
    y_affine = shear_y * x_centered + scale_y * y_centered + translation_y

    # Rotation is applied after the affine transform.
    theta = math.radians(float(params["rotation_deg"]))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_rot = cos_theta * x_affine - sin_theta * y_affine
    y_rot = sin_theta * x_affine + cos_theta * y_affine

    x_new = x_rot + x_centroid
    y_new = y_rot + y_centroid

    if bool(params["add_nonlinear_drift"]):
        xmin = float(params["xmin"])
        xmax = float(params["xmax"])
        ymin = float(params["ymin"])
        ymax = float(params["ymax"])
        width = max(xmax - xmin, np.finfo(float).eps)
        height = max(ymax - ymin, np.finfo(float).eps)
        amplitude = float(params["drift_amplitude"])

        # Sinusoidal drift mimics slow scan distortion across the tile.
        x_new += amplitude * np.sin(2.0 * np.pi * (y - ymin) / height)
        y_new += 0.5 * amplitude * np.sin(2.0 * np.pi * (x - xmin) / width)

    distorted[:, x_col] = x_new
    distorted[:, y_col] = y_new
    return distorted


def _homography_from_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 projective transform H with dst ~= H @ src."""
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("Projective trapezoid distortion requires four 2D corners")

    try:
        from kikuchipy.detectors._fit_projection_center import (
            get_projective_transform_matrix,
        )
    except Exception:
        get_projective_transform_matrix = None

    if get_projective_transform_matrix is not None:
        # Kikuchipy returns the matrix transposed for row-vector dot products.
        # This script applies homographies in the conventional column-vector
        # form, so transpose it back here.
        return get_projective_transform_matrix(src, dst).T

    rows = []
    rhs = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rhs.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        rhs.append(v)

    h = np.linalg.solve(np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float))
    return np.array(
        [
            [h[0], h[1], h[2]],
            [h[3], h[4], h[5]],
            [h[6], h[7], 1.0],
        ],
        dtype=float,
    )


def _apply_homography(x: np.ndarray, y: np.ndarray, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    denom = matrix[2, 0] * x + matrix[2, 1] * y + matrix[2, 2]
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("Projective trapezoid transform produced points at infinity")
    x_new = (matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]) / denom
    y_new = (matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]) / denom
    return x_new, y_new


def _scaled_edge_corners(
    xmin: float,
    xmax: float,
    y: float,
    width_scale: float,
    shift_x: float,
    shift_y: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    center = 0.5 * (xmin + xmax)
    half_width = 0.5 * (xmax - xmin) * width_scale
    return (
        (center - half_width + shift_x, y + shift_y),
        (center + half_width + shift_x, y + shift_y),
    )


def _distort_coordinates_projective_trapezoid(
    data: np.ndarray,
    x_col: int,
    y_col: int,
    params: Dict[str, float | bool | str],
) -> np.ndarray:
    """Map a clean rectangular tile into a projective trapezoid."""
    distorted = data.copy()
    x = distorted[:, x_col].copy()
    y = distorted[:, y_col].copy()

    xmin = float(params["xmin"])
    xmax = float(params["xmax"])
    ymin = float(params["ymin"])
    ymax = float(params["ymax"])
    top_scale = float(params["trapezoid_top_width_scale"])
    bottom_scale = float(params["trapezoid_bottom_width_scale"])

    if top_scale <= 0.0 or bottom_scale <= 0.0:
        raise ValueError("Trapezoid width scales must be positive")

    bottom_left, bottom_right = _scaled_edge_corners(
        xmin,
        xmax,
        ymin,
        bottom_scale,
        float(params["trapezoid_bottom_shift_x"]),
        float(params["trapezoid_bottom_shift_y"]),
    )
    top_left, top_right = _scaled_edge_corners(
        xmin,
        xmax,
        ymax,
        top_scale,
        float(params["trapezoid_top_shift_x"]),
        float(params["trapezoid_top_shift_y"]),
    )

    src = np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=float,
    )
    dst = np.array(
        [
            bottom_left,
            bottom_right,
            top_right,
            top_left,
        ],
        dtype=float,
    )
    matrix = _homography_from_points(src, dst)
    x_new, y_new = _apply_homography(x, y, matrix)

    if bool(params["add_nonlinear_drift"]):
        width = max(xmax - xmin, np.finfo(float).eps)
        height = max(ymax - ymin, np.finfo(float).eps)
        amplitude = float(params["drift_amplitude"])
        x_new += amplitude * np.sin(2.0 * np.pi * (y - ymin) / height)
        y_new += 0.5 * amplitude * np.sin(2.0 * np.pi * (x - xmin) / width)

    distorted[:, x_col] = x_new
    distorted[:, y_col] = y_new
    return distorted


def _distortion_params(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Dict[str, float | bool | str]:
    return {
        "distortion_model": DISTORTION_MODEL,
        "trapezoid_top_width_scale": TRAPEZOID_TOP_WIDTH_SCALE,
        "trapezoid_bottom_width_scale": TRAPEZOID_BOTTOM_WIDTH_SCALE,
        "trapezoid_top_shift_x": TRAPEZOID_TOP_SHIFT_X,
        "trapezoid_bottom_shift_x": TRAPEZOID_BOTTOM_SHIFT_X,
        "trapezoid_top_shift_y": TRAPEZOID_TOP_SHIFT_Y,
        "trapezoid_bottom_shift_y": TRAPEZOID_BOTTOM_SHIFT_Y,
        "translation_x": TRANSLATION_X,
        "translation_y": TRANSLATION_Y,
        "rotation_deg": ROTATION_DEG,
        "shear_x": SHEAR_X,
        "shear_y": SHEAR_Y,
        "scale_x": SCALE_X,
        "scale_y": SCALE_Y,
        "add_nonlinear_drift": ADD_NONLINEAR_DRIFT,
        "drift_amplitude": DRIFT_AMPLITUDE,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
    }


def _manifest_row(
    tile_name: str,
    row: int,
    col: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    n_points: int,
    tilt_corrected: bool,
    distorted: bool,
) -> Dict[str, float | int | str | bool]:
    return {
        "tile_name": tile_name,
        "row": row,
        "col": col,
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "n_points": n_points,
        "tilt_corrected": tilt_corrected,
        "distorted": distorted,
        "distortion_model": DISTORTION_MODEL if distorted else "none",
        "trapezoid_top_width_scale": TRAPEZOID_TOP_WIDTH_SCALE if distorted else 1.0,
        "trapezoid_bottom_width_scale": TRAPEZOID_BOTTOM_WIDTH_SCALE if distorted else 1.0,
        "trapezoid_top_shift_x": TRAPEZOID_TOP_SHIFT_X if distorted else 0.0,
        "trapezoid_bottom_shift_x": TRAPEZOID_BOTTOM_SHIFT_X if distorted else 0.0,
        "trapezoid_top_shift_y": TRAPEZOID_TOP_SHIFT_Y if distorted else 0.0,
        "trapezoid_bottom_shift_y": TRAPEZOID_BOTTOM_SHIFT_Y if distorted else 0.0,
        "translation_x": TRANSLATION_X if distorted else 0.0,
        "translation_y": TRANSLATION_Y if distorted else 0.0,
        "rotation_deg": ROTATION_DEG if distorted else 0.0,
        "shear_x": SHEAR_X if distorted else 0.0,
        "shear_y": SHEAR_Y if distorted else 0.0,
        "scale_x": SCALE_X if distorted else 1.0,
        "scale_y": SCALE_Y if distorted else 1.0,
        "nonlinear_drift": ADD_NONLINEAR_DRIFT if distorted else False,
    }


def _save_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _save_tile_layout_plot(
    path: Path,
    data: np.ndarray,
    x_col: int,
    y_col: int,
    tile_bounds: Sequence[Tuple[int, int, float, float, float, float]],
) -> None:
    cache_dir = path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib is not installed; skipping tile_layout.png")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(data[:, x_col], data[:, y_col], s=0.5, alpha=0.25, color="black")

    for row, col, xmin, xmax, ymin, ymax in tile_bounds:
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            linewidth=0.8,
            edgecolor="tab:red" if (row + col) % 2 else "tab:blue",
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"r{row}c{col}", fontsize=6, color="tab:gray")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("EBSD tile layout")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_example_plot(
    path: Path,
    clean_tile: np.ndarray | None,
    distorted_tile: np.ndarray | None,
    x_col: int,
    y_col: int,
) -> None:
    if clean_tile is None or distorted_tile is None:
        return

    cache_dir = path.parent / ".matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping example_clean_vs_distorted.png")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False, sharey=False)
    axes[0].scatter(clean_tile[:, x_col], clean_tile[:, y_col], s=2, color="black")
    axes[0].set_title("Clean tile")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].scatter(
        distorted_tile[:, x_col],
        distorted_tile[:, y_col],
        s=2,
        color="tab:red",
    )
    axes[1].set_title("Distorted tile")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_tiles(
    data: np.ndarray,
    header_lines: Sequence[str],
    output_dir: str | os.PathLike[str] = OUTPUT_DIR,
    x_col: int = X_COL,
    y_col: int = Y_COL,
    tilt_corrected: bool = False,
) -> Tuple[int, Path]:
    """Crop, save tiles, write manifest, and save optional visualizations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    x = data[:, x_col]
    y = data[:, y_col]
    xmin_all = float(np.min(x))
    xmax_all = float(np.max(x))
    ymin_all = float(np.min(y))
    ymax_all = float(np.max(y))

    step_x = TILE_WIDTH * (1.0 - OVERLAP_FRACTION)
    step_y = TILE_HEIGHT * (1.0 - OVERLAP_FRACTION)

    manifest_rows: List[Dict[str, object]] = []
    tile_bounds: List[Tuple[int, int, float, float, float, float]] = []
    example_clean = None
    example_distorted = None

    row = 0
    y0 = ymin_all
    while y0 <= ymax_all:
        col = 0
        x0 = xmin_all
        while x0 <= xmax_all:
            xmin = x0
            xmax = x0 + TILE_WIDTH
            ymin = y0
            ymax = y0 + TILE_HEIGHT

            tile = crop_tile(data, x_col, y_col, xmin, xmax, ymin, ymax)
            if tile.shape[0] >= MIN_POINTS_PER_TILE:
                tile_bounds.append((row, col, xmin, xmax, ymin, ymax))

                if SAVE_UNDISTORTED_TILES:
                    clean_name = f"tile_r{row}_c{col}_clean.ang"
                    write_ang(output_path / clean_name, header_lines, tile)
                    manifest_rows.append(
                        _manifest_row(
                            clean_name,
                            row,
                            col,
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            tile.shape[0],
                            tilt_corrected,
                            False,
                        )
                    )

                should_distort = (
                    (row + col) % 2 == 1 if DISTORT_EVERY_OTHER_TILE else True
                )
                if should_distort:
                    params = _distortion_params(xmin, xmax, ymin, ymax)
                    distorted = distort_coordinates(tile, x_col, y_col, params)
                    distorted_name = f"tile_r{row}_c{col}_distorted.ang"
                    write_ang(output_path / distorted_name, header_lines, distorted)
                    manifest_rows.append(
                        _manifest_row(
                            distorted_name,
                            row,
                            col,
                            xmin,
                            xmax,
                            ymin,
                            ymax,
                            distorted.shape[0],
                            tilt_corrected,
                            True,
                        )
                    )
                    if example_clean is None:
                        example_clean = tile
                        example_distorted = distorted

            col += 1
            x0 += step_x

        row += 1
        y0 += step_y

    manifest_path = output_path / "tile_manifest.csv"
    _save_manifest(manifest_path, manifest_rows)
    _save_tile_layout_plot(
        output_path / "tile_layout.png",
        data,
        x_col,
        y_col,
        tile_bounds,
    )
    _save_example_plot(
        output_path / "example_clean_vs_distorted.png",
        example_clean,
        example_distorted,
        x_col,
        y_col,
    )

    return len(manifest_rows), manifest_path


def _validate_config(data: np.ndarray) -> None:
    if not (0.0 <= OVERLAP_FRACTION <= 0.9):
        raise ValueError("OVERLAP_FRACTION must be between 0 and 0.9")
    if TILE_WIDTH <= 0 or TILE_HEIGHT <= 0:
        raise ValueError("TILE_WIDTH and TILE_HEIGHT must be greater than 0")
    if DISTORTION_MODEL not in {"projective_trapezoid", "affine_drift"}:
        raise ValueError(
            "DISTORTION_MODEL must be 'projective_trapezoid' or 'affine_drift'"
        )
    if TRAPEZOID_TOP_WIDTH_SCALE <= 0 or TRAPEZOID_BOTTOM_WIDTH_SCALE <= 0:
        raise ValueError("Trapezoid width scales must be greater than 0")
    if X_COL < 0 or Y_COL < 0:
        raise ValueError("X_COL and Y_COL must be zero-based, non-negative indexes")
    if data.shape[1] <= max(X_COL, Y_COL):
        raise ValueError(
            f"Data has {data.shape[1]} columns, but X_COL={X_COL} and Y_COL={Y_COL}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop an EBSD .ang file into overlapping clean/distorted tiles."
    )
    parser.add_argument(
        "input_ang",
        nargs="?",
        default=INPUT_ANG,
        help="Path to input .ang file. Defaults to INPUT_ANG in the config section.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for cropped tiles. Defaults to OUTPUT_DIR in the config section.",
    )
    return parser.parse_args()


def main() -> None:
    np.random.seed(RANDOM_SEED)
    args = _parse_args()

    header_lines, data = read_ang(args.input_ang)
    _validate_config(data)

    print(f"Loaded {data.shape[0]} points and {data.shape[1]} columns")

    working_data = data
    if APPLY_TILT_CORRECTION:
        working_data = apply_y_tilt_correction(working_data, Y_COL, TILT_DEGREES)

    x_range = (
        float(np.min(working_data[:, X_COL])),
        float(np.max(working_data[:, X_COL])),
    )
    y_range = (
        float(np.min(working_data[:, Y_COL])),
        float(np.max(working_data[:, Y_COL])),
    )
    print(f"Corrected coordinate range: X {x_range}, Y {y_range}")

    tiles_saved, manifest_path = make_tiles(
        working_data,
        header_lines,
        output_dir=args.output_dir,
        x_col=X_COL,
        y_col=Y_COL,
        tilt_corrected=APPLY_TILT_CORRECTION,
    )

    print(f"Tiles saved: {tiles_saved}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
