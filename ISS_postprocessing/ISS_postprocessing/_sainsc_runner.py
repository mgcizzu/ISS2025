#!/usr/bin/env python3
"""Execute one prepared SAINSC analysis inside the SAINSC environment."""

from __future__ import annotations

import argparse
import json
import pickle
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd


def _package_version() -> str:
    try:
        return version("sainsc")
    except PackageNotFoundError:
        return "unknown"


def _load_signatures(path: str | None, available_genes: set[str]) -> pd.DataFrame | None:
    if path is None:
        return None
    signatures = pd.read_csv(path, index_col=0)
    if signatures.empty:
        raise ValueError("The SAINSC signature table is empty")
    signatures.index = signatures.index.astype(str)
    signatures = signatures.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(signatures.to_numpy(dtype=float)).all():
        raise ValueError("SAINSC signatures must contain finite values")
    if (signatures.to_numpy(dtype=float) < 0).any():
        raise ValueError("SAINSC signatures must be non-negative")
    missing = sorted(set(signatures.index) - available_genes)
    if missing:
        raise ValueError(
            "Signature genes are absent from the transcript table: "
            + ", ".join(missing[:20])
        )
    return signatures


def _add_physical_coordinates(adata, coordinate_config: dict) -> None:
    grid = np.asarray(adata.obsm["spatial"], dtype=float).copy()
    grid_size_um = float(coordinate_config["grid_size_um"])
    grid_size_source = float(coordinate_config["grid_size_source"])
    pixel_size_um = float(coordinate_config["pixel_size_um"])
    origin = np.asarray(coordinate_config["grid_origin_index"], dtype=float)

    spatial_source = (grid + origin) * grid_size_source
    if coordinate_config["source_unit"] == "pixel":
        spatial_px = spatial_source
        spatial_um = spatial_px * pixel_size_um
    else:
        spatial_um = spatial_source
        spatial_px = spatial_um / pixel_size_um
    adata.obsm["sainsc_grid"] = grid
    adata.obsm["spatial_um"] = spatial_um
    # The rest of the ISS postprocessing package uses image-pixel coordinates.
    adata.obsm["spatial"] = spatial_px


def run(config_path: str | Path) -> None:
    try:
        from sainsc import LazyKDE
    except ImportError as exc:
        raise RuntimeError(
            "SAINSC is not installed in this environment. Install sainsc==0.4.0 "
            "in Python 3.12+ or use the supplied environment file."
        ) from exc

    config_path = Path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    table = pd.read_csv(config["transcripts"])
    table["gene"] = table["gene"].astype(str)

    coords = config["coordinates"]
    settings = config["analysis"]
    outputs = {name: Path(path) for name, path in config["outputs"].items()}
    for path in outputs.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    analysis = LazyKDE.from_dataframe(
        table,
        resolution=float(coords["source_resolution_nm"]),
        binsize=float(coords["grid_size_source"]),
        n_threads=settings["n_threads"],
    )
    analysis.gaussian_kernel(
        float(settings["kernel_bandwidth_um"]),
        unit="um",
        truncate=float(settings["kernel_truncate"]),
    )
    analysis.calculate_total_mRNA()
    analysis.calculate_total_mRNA_KDE()

    signatures = _load_signatures(config.get("signatures"), set(analysis.genes))
    if signatures is not None:
        analysis.assign_celltype(
            signatures,
            log=bool(settings["signatures_log_transformed"]),
        )

    if settings["background_min_norm"] is not None:
        analysis.filter_background(
            float(settings["background_min_norm"]),
            None
            if settings["background_min_cosine"] is None
            else float(settings["background_min_cosine"]),
            None
            if settings["background_min_assignment"] is None
            else float(settings["background_min_assignment"]),
        )

    analysis.find_local_maxima(
        int(settings["min_peak_distance_grid"]),
        min_area=int(settings["min_peak_area_grid"]),
    )
    if analysis.local_maxima is None or len(analysis.local_maxima[0]) == 0:
        raise RuntimeError(
            "SAINSC found no local maxima. Review the grid, kernel, background, "
            "and peak-distance settings."
        )

    adata = analysis.load_local_maxima()
    _add_physical_coordinates(adata, coords)
    adata.uns["sainsc"] = {
        "version": _package_version(),
        "region": config["region"],
        "coordinate_unit": coords["source_unit"],
        "pixel_size_um": float(coords["pixel_size_um"]),
        "grid_size_um": float(coords["grid_size_um"]),
        "kernel_bandwidth_um": float(settings["kernel_bandwidth_um"]),
        "min_peak_distance_um": float(settings["min_peak_distance_grid"])
        * float(coords["grid_size_um"]),
    }
    adata.write_h5ad(outputs["local_maxima"])

    map_keys: list[str] = []
    if settings["save_maps"]:
        map_payload = {}
        for name in (
            "total_mRNA_KDE",
            "background",
            "celltype_map",
            "cosine_similarity",
            "assignment_score",
        ):
            value = getattr(analysis, name)
            if value is not None:
                map_payload[name] = np.asarray(value)
                map_keys.append(name)
        if analysis.celltypes is not None:
            map_payload["celltypes"] = np.asarray(analysis.celltypes, dtype=str)
            map_keys.append("celltypes")
        np.savez_compressed(outputs["maps"], **map_payload)

    if settings["save_analysis"]:
        with outputs["analysis"].open("wb") as handle:
            pickle.dump(analysis, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "method": "sainsc",
        "sainsc_version": _package_version(),
        "region": config["region"],
        "n_transcripts": int(len(table)),
        "n_genes": int(table["gene"].nunique()),
        "n_local_maxima": int(adata.n_obs),
        "celltypes": [] if analysis.celltypes is None else list(analysis.celltypes),
        "saved_map_keys": map_keys,
        "config": config,
    }
    outputs["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Prepared SAINSC JSON configuration")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
