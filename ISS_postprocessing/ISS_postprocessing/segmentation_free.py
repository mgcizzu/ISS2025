"""Adapters for segmentation-free transcript analysis.

SAINSC is intentionally kept outside the cell-instance segmentation dispatcher.
It produces kernel-density maps and local-maxima cell proxies, not a label mask
whose positive integers identify individual cells.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

from .segmentation_wrappers import (
    TableLike,
    _as_command,
    _run_command,
    normalize_transcript_table,
)


PathLike = str | os.PathLike
Command = str | PathLike | Sequence[str | PathLike]


def _validate_coordinate_settings(
    coordinate_unit: str,
    pixel_size_um: float,
    grid_size_um: float,
) -> str:
    unit = str(coordinate_unit).lower()
    if unit not in {"pixel", "um"}:
        raise ValueError("coordinate_unit must be 'pixel' or 'um'")
    if float(pixel_size_um) <= 0:
        raise ValueError("pixel_size_um must be > 0")
    if float(grid_size_um) <= 0:
        raise ValueError("grid_size_um must be > 0")
    return unit


def _sainsc_output_dir(
    *,
    region: str,
    input_dir: PathLike | None,
    output_dir: PathLike | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    if input_dir is None:
        raise ValueError("Pass input_dir or an explicit output_dir")
    return (
        Path(input_dir)
        / region
        / "postprocessing"
        / "segmentation_free"
        / "sainsc"
    )


def _sainsc_paths(output_dir: Path, region: str) -> dict[str, Path]:
    stem = f"{region}_sainsc"
    return {
        "output_dir": output_dir,
        "transcripts": output_dir / f"{stem}_transcripts.csv",
        "config": output_dir / f"{stem}_config.json",
        "local_maxima": output_dir / f"{stem}_local_maxima.h5ad",
        "maps": output_dir / f"{stem}_maps.npz",
        "analysis": output_dir / f"{stem}_analysis.pickle",
        "metadata": output_dir / f"{stem}_metadata.json",
        "signatures": output_dir / f"{stem}_signatures.csv",
    }


def _prepare_signatures(
    signatures: PathLike | pd.DataFrame | None,
    destination: Path,
) -> Path | None:
    if signatures is None:
        return None
    if isinstance(signatures, pd.DataFrame):
        if signatures.empty:
            raise ValueError("signatures cannot be empty")
        if signatures.index.hasnans:
            raise ValueError("signature gene names cannot contain missing values")
        signatures.to_csv(destination)
        return destination
    path = Path(signatures).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SAINSC signature table not found: {path}")
    return path


def prepare_sainsc_analysis(
    transcripts: TableLike,
    *,
    region: str,
    input_dir: PathLike | None = None,
    output_dir: PathLike | None = None,
    gene_col: str = "target",
    x_col: str = "xc",
    y_col: str = "yc",
    quality_col: str | None = None,
    min_quality: float | None = None,
    coordinate_unit: str = "pixel",
    pixel_size_um: float,
    grid_size_um: float = 0.5,
    kernel_bandwidth_um: float = 4.0,
    kernel_truncate: float = 2.0,
    background_min_norm: float | None = None,
    background_min_cosine: float | None = None,
    background_min_assignment: float | None = None,
    min_peak_distance_um: float = 4.0,
    min_peak_area_um2: float = 0.0,
    signatures: PathLike | pd.DataFrame | None = None,
    signatures_log_transformed: bool = False,
    n_threads: int | None = None,
    save_maps: bool = False,
    save_analysis: bool = True,
) -> dict[str, Path]:
    """Prepare normalized ISS input and a reproducible SAINSC runner config.

    The decoded coordinates are retained in their source unit in the prepared
    CSV.  The external runner uses ``coordinate_unit``, ``pixel_size_um`` and
    ``grid_size_um`` to create a physically calibrated SAINSC grid.
    """
    unit = _validate_coordinate_settings(
        coordinate_unit,
        pixel_size_um,
        grid_size_um,
    )
    if float(kernel_bandwidth_um) <= 0:
        raise ValueError("kernel_bandwidth_um must be > 0")
    if float(kernel_truncate) <= 0:
        raise ValueError("kernel_truncate must be > 0")
    if float(min_peak_distance_um) <= 0:
        raise ValueError("min_peak_distance_um must be > 0")
    if float(min_peak_area_um2) < 0:
        raise ValueError("min_peak_area_um2 must be >= 0")
    if n_threads is not None and int(n_threads) <= 0:
        raise ValueError("n_threads must be > 0 when provided")
    if background_min_norm is None and any(
        value is not None
        for value in (background_min_cosine, background_min_assignment)
    ):
        raise ValueError(
            "background_min_norm is required when cosine or assignment "
            "background thresholds are used"
        )
    if signatures is None and any(
        value is not None
        for value in (background_min_cosine, background_min_assignment)
    ):
        raise ValueError(
            "cosine and assignment thresholds require a signature table"
        )

    destination = _sainsc_output_dir(
        region=region,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    destination.mkdir(parents=True, exist_ok=True)
    paths = _sainsc_paths(destination, region)

    table = normalize_transcript_table(
        transcripts,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        quality_col=quality_col,
        min_quality=min_quality,
    )
    if (table[["x", "y"]] < 0).any().any():
        raise ValueError("SAINSC requires non-negative transcript coordinates")
    table.to_csv(paths["transcripts"], index=False)

    signature_path = _prepare_signatures(signatures, paths["signatures"])
    grid_size_source = (
        float(grid_size_um) / float(pixel_size_um)
        if unit == "pixel"
        else float(grid_size_um)
    )
    source_resolution_nm = (
        float(pixel_size_um) * 1000.0 if unit == "pixel" else 1000.0
    )
    grid_origin_index = [
        math.floor(float(table[axis].min()) / grid_size_source)
        for axis in ("x", "y")
    ]
    min_peak_distance_grid = max(
        1,
        int(round(float(min_peak_distance_um) / float(grid_size_um))),
    )
    min_peak_area_grid = max(
        0,
        int(round(float(min_peak_area_um2) / float(grid_size_um) ** 2)),
    )

    config = {
        "region": str(region),
        "transcripts": str(paths["transcripts"].resolve()),
        "signatures": None if signature_path is None else str(signature_path),
        "outputs": {
            "local_maxima": str(paths["local_maxima"].resolve()),
            "maps": str(paths["maps"].resolve()),
            "analysis": str(paths["analysis"].resolve()),
            "metadata": str(paths["metadata"].resolve()),
        },
        "coordinates": {
            "source_unit": unit,
            "pixel_size_um": float(pixel_size_um),
            "grid_size_um": float(grid_size_um),
            "grid_size_source": grid_size_source,
            "source_resolution_nm": source_resolution_nm,
            # SAINSC shifts binned coordinates so the smallest index is zero.
            # Retaining this origin lets the runner restore whole-slide positions.
            "grid_origin_index": grid_origin_index,
        },
        "analysis": {
            "kernel_bandwidth_um": float(kernel_bandwidth_um),
            "kernel_truncate": float(kernel_truncate),
            "background_min_norm": background_min_norm,
            "background_min_cosine": background_min_cosine,
            "background_min_assignment": background_min_assignment,
            "min_peak_distance_grid": min_peak_distance_grid,
            "min_peak_area_grid": min_peak_area_grid,
            "signatures_log_transformed": bool(signatures_log_transformed),
            "n_threads": None if n_threads is None else int(n_threads),
            "save_maps": bool(save_maps),
            "save_analysis": bool(save_analysis),
        },
        "input_summary": {
            "n_transcripts": int(len(table)),
            "n_genes": int(table["gene"].nunique()),
        },
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths


def run_sainsc_analysis(
    transcripts: TableLike,
    *,
    region: str,
    pixel_size_um: float,
    input_dir: PathLike | None = None,
    output_dir: PathLike | None = None,
    sainsc_python: Command = sys.executable,
    overwrite: bool = False,
    **kwargs,
) -> dict[str, Path]:
    """Run SAINSC in an explicit environment and return its output paths.

    ``sainsc_python`` may be a Python executable or a command prefix such as
    ``["conda", "run", "-n", "sainsc", "python"]``.
    """
    destination = _sainsc_output_dir(
        region=region,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    paths = _sainsc_paths(destination, region)
    if paths["local_maxima"].is_file() and not overwrite:
        print(f"[INFO] Reusing existing SAINSC output: {paths['local_maxima']}")
        return paths

    paths = prepare_sainsc_analysis(
        transcripts,
        region=region,
        input_dir=input_dir,
        output_dir=output_dir,
        pixel_size_um=pixel_size_um,
        **kwargs,
    )
    runner = Path(__file__).with_name("_sainsc_runner.py")
    command = _as_command(sainsc_python) + [str(runner), str(paths["config"])]
    _run_command(command)

    required = [paths["local_maxima"], paths["metadata"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "SAINSC completed without producing required output(s): "
            + ", ".join(missing)
        )
    return paths


__all__ = ["prepare_sainsc_analysis", "run_sainsc_analysis"]
