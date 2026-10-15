"""Adapters for transcript-based spatial-domain analysis.

SFUMATO clusters spatial bins and quantifies assignment uncertainty.  Its
clusters are semantic tissue domains, not individual-cell segmentation labels.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .segmentation_wrappers import (
    TableLike,
    _as_command,
    _run_command,
    normalize_transcript_table,
)


PathLike = str | os.PathLike
Command = str | PathLike | Sequence[str | PathLike]

SFUMATO_TESTED_REVISION = "93806f5c374f88bf4de0bf6e13df4bb7c85df4a4"


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    if not result:
        raise ValueError("run_name must contain at least one letter or number")
    return result


def _sfumato_output_dir(
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
        / "spatial_domains"
        / "sfumato"
    )


def _sfumato_paths(output_dir: Path, region: str) -> dict[str, Path]:
    work = output_dir / "work"
    return {
        "output_dir": output_dir,
        "work_dir": work,
        "transcripts": work / f"{region}_sfumato_transcripts.csv",
        "config": work / f"{region}_sfumato_config.json",
        "cache_dir": output_dir / "preprocess_cache",
        "native_results": output_dir / "native_results",
        "manifest": output_dir / f"{region}_sfumato_manifest.json",
    }


def prepare_sfumato_analysis(
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
    run_name: str | None = None,
    k_list: Sequence[int] = (10, 15, 20),
    bin_width_um: float = 10.0,
    smoothing_factor: int = 5,
    min_transcripts_per_bin: int = 4,
    svd_components: int | None = None,
    bgm_oversample: float = 1.5,
    seed: int = 8,
    save_points2regions: bool = True,
    save_transcript_probabilities: bool = False,
    bgm_weight_prior: float = 1.0,
    color_map: str = "gist_ncar",
) -> dict[str, Path]:
    """Prepare an ISS transcript table and a complete SFUMATO JSON config.

    Coordinates written for SFUMATO are always micrometres, making
    ``bin_width_um`` comparable across microscopes and datasets.
    """
    unit = str(coordinate_unit).lower()
    if unit not in {"pixel", "um"}:
        raise ValueError("coordinate_unit must be 'pixel' or 'um'")
    if float(pixel_size_um) <= 0:
        raise ValueError("pixel_size_um must be > 0")
    if float(bin_width_um) <= 0:
        raise ValueError("bin_width_um must be > 0")
    if not float(bin_width_um).is_integer():
        raise ValueError(
            "SFUMATO currently casts bin_width to an integer; pass a whole number "
            "of micrometres to avoid silent truncation"
        )
    if int(smoothing_factor) <= 0:
        raise ValueError("smoothing_factor must be > 0")
    if int(min_transcripts_per_bin) <= 0:
        raise ValueError("min_transcripts_per_bin must be > 0")
    if float(bgm_oversample) < 1:
        raise ValueError("bgm_oversample must be >= 1")
    if float(bgm_weight_prior) <= 0:
        raise ValueError("bgm_weight_prior must be > 0")

    clusters = sorted({int(value) for value in k_list})
    if not clusters or any(value < 2 for value in clusters):
        raise ValueError("k_list must contain cluster counts >= 2")

    destination = _sfumato_output_dir(
        region=region,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    paths = _sfumato_paths(destination, region)
    paths["work_dir"].mkdir(parents=True, exist_ok=True)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)
    paths["native_results"].mkdir(parents=True, exist_ok=True)

    table = normalize_transcript_table(
        transcripts,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        quality_col=quality_col,
        min_quality=min_quality,
    )
    if (table[["x", "y"]] < 0).any().any():
        raise ValueError("SFUMATO requires non-negative transcript coordinates")

    n_genes = int(table["gene"].nunique())
    if n_genes < 2:
        raise ValueError("SFUMATO requires at least two detected genes")
    components = (
        min(10, n_genes - 1)
        if svd_components is None
        else int(svd_components)
    )
    if components <= 0 or components >= n_genes:
        raise ValueError(
            f"svd_components must be between 1 and n_genes - 1 ({n_genes - 1})"
        )

    coordinate_scale = float(pixel_size_um) if unit == "pixel" else 1.0
    sfumato_table = pd.DataFrame(
        {
            "x": table["x"].to_numpy(dtype=float) * coordinate_scale,
            "y": table["y"].to_numpy(dtype=float) * coordinate_scale,
            "target_name": table["gene"].astype(str).to_numpy(),
            "num_id": np.arange(len(table), dtype=np.int64),
        }
    )
    sfumato_table.to_csv(paths["transcripts"], index=False)

    resolved_run_name = _safe_name(run_name or f"{region}_sfumato")
    config = {
        "run_name": resolved_run_name,
        "input_csv": str(paths["transcripts"].resolve()),
        "columns": {
            "x": "x",
            "y": "y",
            "gene": "target_name",
            "num_id": "num_id",
        },
        "preprocess": {
            "k_list": clusters,
            "bgm_oversample": float(bgm_oversample),
            # SFUMATO's native name is min_genes_per_bin, but its implementation
            # thresholds transcript/gene-event density rather than unique genes.
            "min_genes_per_bin": int(min_transcripts_per_bin),
            "bin_width": int(bin_width_um),
            "factor": int(smoothing_factor),
            "seed": int(seed),
            "comp": components,
            "save_p2r": bool(save_points2regions),
            "alpha": 1.0,
            "cache_dir": str(paths["cache_dir"].resolve()),
        },
        "bgm": {
            "outroot": str(paths["native_results"].resolve()),
            "save_transcript_proba": bool(save_transcript_probabilities),
            "merge_method": "complete",
            "merge_metric": "cosine",
            "bgm_weight_prior": float(bgm_weight_prior),
        },
        "color": {
            "colormap": str(color_map),
            "colormap_start": 0.08,
            "colormap_end": 0.92,
            "dendrogram_branch_linewidth": 3.0,
        },
        "umap": {"run": False},
        "iss_metadata": {
            "region": str(region),
            "source_coordinate_unit": unit,
            "pixel_size_um": float(pixel_size_um),
            "sfumato_coordinate_unit": "um",
            "n_transcripts": int(len(table)),
            "n_genes": n_genes,
        },
    }
    paths["config"].write_text(json.dumps(config, indent=2), encoding="utf-8")
    return paths


def _sfumato_revision(sfumato_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(sfumato_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def standardize_sfumato_outputs(
    *,
    output_dir: PathLike,
    region: str,
    pixel_size_um: float,
) -> dict[str, Path]:
    """Create Squidpy-friendly bin AnnData files from native SFUMATO CSVs."""
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "Standardizing SFUMATO output requires anndata"
        ) from exc

    destination = Path(output_dir)
    paths = _sfumato_paths(destination, region)
    bin_files = sorted(paths["native_results"].glob("K*/SVD*/*_binlevel_FULL.csv"))
    if not bin_files:
        raise RuntimeError(
            f"SFUMATO did not produce bin-level output under {paths['native_results']}"
        )

    standardized: dict[str, Path] = {}
    for bin_file in bin_files:
        table = pd.read_csv(bin_file)
        required = {"x", "y", "bin_id", "cluster", "p1", "p2", "compl_p1"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"SFUMATO output {bin_file} is missing columns: {missing}")

        k_match = re.fullmatch(r"K(\d+)", bin_file.parent.parent.name)
        if k_match is None:
            raise ValueError(f"Cannot determine K from SFUMATO path: {bin_file}")
        k = int(k_match.group(1))
        spatial_um = table[["x", "y"]].to_numpy(dtype=float)
        obs = table.drop(columns=["x", "y"]).copy()
        obs.index = pd.Index(
            [f"bin-{int(value)}" for value in obs["bin_id"]],
            name="sfumato_bin",
        )
        adata = ad.AnnData(obs=obs)
        adata.obsm["spatial_um"] = spatial_um
        adata.obsm["spatial"] = spatial_um / float(pixel_size_um)
        adata.uns["sfumato"] = {
            "semantic_unit": "spatial_domain",
            "coordinate_unit": "um",
            "pixel_size_um": float(pixel_size_um),
            "requested_k": k,
            "native_bin_csv": str(bin_file.resolve()),
        }
        output = destination / f"{region}_sfumato_K{k}_bins.h5ad"
        adata.write_h5ad(output)
        standardized[f"bins_K{k}"] = output
    return standardized


def run_sfumato_analysis(
    transcripts: TableLike,
    *,
    region: str,
    pixel_size_um: float,
    sfumato_root: PathLike,
    input_dir: PathLike | None = None,
    output_dir: PathLike | None = None,
    sfumato_python: Command = sys.executable,
    tested_revision: str | None = SFUMATO_TESTED_REVISION,
    strict_revision: bool = False,
    overwrite: bool = False,
    **kwargs,
) -> dict[str, Path]:
    """Run SFUMATO and standardize its bin-level results.

    A CUDA-capable PyTorch environment is required by SFUMATO's BGM stage.
    ``sfumato_python`` may be an executable or a command prefix such as
    ``["conda", "run", "-n", "sfumato", "python"]``.
    """
    destination = _sfumato_output_dir(
        region=region,
        input_dir=input_dir,
        output_dir=output_dir,
    )
    paths = _sfumato_paths(destination, region)
    if paths["manifest"].is_file() and not overwrite:
        print(f"[INFO] Reusing existing SFUMATO output: {paths['manifest']}")
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        for name, value in manifest.get("standardized_outputs", {}).items():
            paths[name] = Path(value)
        return paths

    root = Path(sfumato_root).resolve()
    runner = root / "run_sfumato.py"
    if not runner.is_file():
        raise FileNotFoundError(f"SFUMATO runner not found: {runner}")

    revision = _sfumato_revision(root)
    if tested_revision and revision and revision != tested_revision:
        message = (
            f"SFUMATO revision {revision} differs from tested revision "
            f"{tested_revision}"
        )
        if strict_revision:
            raise RuntimeError(message)
        print(f"[WARNING] {message}")

    paths = prepare_sfumato_analysis(
        transcripts,
        region=region,
        input_dir=input_dir,
        output_dir=output_dir,
        pixel_size_um=pixel_size_um,
        **kwargs,
    )
    command = _as_command(sfumato_python) + [
        str(runner),
        "--config",
        str(paths["config"]),
    ]
    if overwrite:
        command.append("--force-preprocess")
    _run_command(command, cwd=root)

    standardized = standardize_sfumato_outputs(
        output_dir=destination,
        region=region,
        pixel_size_um=pixel_size_um,
    )
    paths.update(standardized)
    native_bin_files = sorted(
        paths["native_results"].glob("K*/SVD*/*_binlevel_FULL.csv")
    )
    manifest = {
        "method": "sfumato",
        "semantic_unit": "spatial_domain",
        "region": str(region),
        "tested_revision": tested_revision,
        "actual_revision": revision,
        "config": str(paths["config"].resolve()),
        "native_bin_outputs": [str(path.resolve()) for path in native_bin_files],
        "standardized_outputs": {
            name: str(path.resolve()) for name, path in standardized.items()
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths


__all__ = [
    "SFUMATO_TESTED_REVISION",
    "prepare_sfumato_analysis",
    "run_sfumato_analysis",
    "standardize_sfumato_outputs",
]
