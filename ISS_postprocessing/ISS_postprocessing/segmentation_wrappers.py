"""Adapters for transcript-aware segmentation programs.

The wrappers in this module deliberately keep Segger, Proseg, and BIDCell in
their own environments.  Their dependency stacks are not mutually compatible,
and importing them from :mod:`ISS_postprocessing.segmentation` would make the
light-weight mask utilities unusable unless every method was installed.

All public wrappers accept an ISS decoded transcript table (a path or a pandas
``DataFrame``), use pixel coordinates by default, and write a two-dimensional
integer label image with background label 0 using ``scipy.sparse.save_npz``.
The returned value matches the existing Cellpose and StarDist helpers:
``(dense_mask, coo_mask)``.
"""

from __future__ import annotations

import gzip
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, load_npz, save_npz
from skimage.draw import polygon as draw_polygon
from skimage.segmentation import expand_labels
from skimage.transform import resize
from tifffile import imread


PathLike = str | os.PathLike
TableLike = PathLike | pd.DataFrame
ArrayLike = PathLike | np.ndarray
Command = str | PathLike | Sequence[str | PathLike]


def _as_command(command: Command) -> list[str]:
    """Normalize an executable or an explicit command prefix.

    A string is treated as one executable path, never as a shell command.  Use
    a sequence for environment runners, for example
    ``["conda", "run", "-n", "segger", "segger"]``.
    """
    if isinstance(command, (str, os.PathLike)):
        return [str(command)]
    result = [str(part) for part in command]
    if not result:
        raise ValueError("Command prefix cannot be empty")
    return result


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    printable = shlex.join([str(part) for part in command])
    print(f"[INFO] Running: {printable}")
    try:
        subprocess.run(
            [str(part) for part in command],
            cwd=None if cwd is None else str(cwd),
            env=None if env is None else dict(env),
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Executable not found while running: {printable}. "
            "Install the method or pass an explicit command prefix."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"External segmentation command failed with exit code "
            f"{exc.returncode}: {printable}"
        ) from exc


def load_transcript_table(transcripts: TableLike) -> pd.DataFrame:
    """Load a transcript table from CSV/CSV.GZ/TSV/Parquet or copy a DataFrame."""
    if isinstance(transcripts, pd.DataFrame):
        return transcripts.copy()

    path = Path(transcripts)
    if not path.is_file():
        raise FileNotFoundError(f"Transcript table not found: {path}")

    lower = path.name.lower()
    if lower.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading Parquet transcript tables requires pyarrow or fastparquet"
            ) from exc
    if lower.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t")
    if lower.endswith((".csv", ".csv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path)
    raise ValueError(
        f"Unsupported transcript table format for {path}; use CSV, TSV, or Parquet"
    )


def normalize_transcript_table(
    transcripts: TableLike,
    *,
    gene_col: str = "target",
    x_col: str = "xc",
    y_col: str = "yc",
    cell_id_col: str | None = None,
    quality_col: str | None = None,
    min_quality: float | None = None,
) -> pd.DataFrame:
    """Return a validated table with ``gene``, ``x``, ``y`` and optional IDs.

    Coordinates are retained in input pixel units.  Each wrapper performs the
    conversion to the method's physical coordinate system with
    ``pixel_size_um``.
    """
    table = load_transcript_table(transcripts)
    required = {gene_col, x_col, y_col}
    if cell_id_col is not None:
        required.add(cell_id_col)
    if quality_col is not None:
        required.add(quality_col)
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Missing transcript columns: {missing}")

    columns = [gene_col, x_col, y_col]
    if cell_id_col is not None:
        columns.append(cell_id_col)
    if quality_col is not None:
        columns.append(quality_col)
    table = table.loc[:, columns].copy()

    table[x_col] = pd.to_numeric(table[x_col], errors="coerce")
    table[y_col] = pd.to_numeric(table[y_col], errors="coerce")
    if quality_col is not None:
        table[quality_col] = pd.to_numeric(table[quality_col], errors="coerce")
        if min_quality is not None:
            table = table.loc[table[quality_col] >= min_quality]

    before = len(table)
    table = table.dropna(subset=[gene_col, x_col, y_col]).copy()
    if len(table) != before:
        print(f"[INFO] Dropped {before - len(table)} transcripts with missing values")
    if table.empty:
        raise ValueError("No valid transcripts remain after normalization")
    if not np.isfinite(table[[x_col, y_col]].to_numpy(dtype=float)).all():
        raise ValueError("Transcript coordinates must be finite")

    renamed = {gene_col: "gene", x_col: "x", y_col: "y"}
    if cell_id_col is not None:
        renamed[cell_id_col] = "cell_id"
    if quality_col is not None:
        renamed[quality_col] = "quality"
    table = table.rename(columns=renamed)
    table["gene"] = table["gene"].astype(str)
    return table.reset_index(drop=True)


def _load_2d_image(image: ArrayLike) -> np.ndarray:
    array = imread(str(image)) if isinstance(image, (str, os.PathLike)) else np.asarray(image)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {array.shape}")
    return array


def _load_label_mask(mask: ArrayLike) -> np.ndarray:
    if isinstance(mask, (str, os.PathLike)):
        path = Path(mask)
        if not path.is_file():
            raise FileNotFoundError(f"Initial mask not found: {path}")
        if path.suffix.lower() == ".npz":
            array = load_npz(path).toarray()
        elif path.suffix.lower() == ".npy":
            array = np.load(path)
        else:
            array = imread(str(path))
    else:
        array = np.asarray(mask)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D label mask, got shape {array.shape}")
    if not np.isfinite(array).all() or np.any(array < 0):
        raise ValueError("Label masks must contain finite, non-negative values")
    if not np.all(array == np.floor(array)):
        raise ValueError("Label masks must contain integer values")
    return array.astype(np.uint32, copy=False)


def _validate_shape(mask_shape: Sequence[int] | None) -> tuple[int, int] | None:
    if mask_shape is None:
        return None
    if len(mask_shape) != 2:
        raise ValueError("mask_shape must be (height, width)")
    shape = (int(mask_shape[0]), int(mask_shape[1]))
    if shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"mask_shape must be positive, got {shape}")
    return shape


def _resolve_shape(
    *,
    image: ArrayLike | None,
    initial_mask: ArrayLike | None,
    mask_shape: Sequence[int] | None,
) -> tuple[int, int]:
    requested = _validate_shape(mask_shape)
    candidates: list[tuple[str, tuple[int, int]]] = []
    if requested is not None:
        candidates.append(("mask_shape", requested))
    if image is not None:
        candidates.append(("image", _load_2d_image(image).shape))
    if initial_mask is not None:
        candidates.append(("initial_mask", _load_label_mask(initial_mask).shape))
    if not candidates:
        raise ValueError("Provide image, initial_mask, or mask_shape to define the output canvas")
    shape = candidates[0][1]
    mismatches = [(name, value) for name, value in candidates[1:] if value != shape]
    if mismatches:
        detail = ", ".join(f"{name}={value}" for name, value in candidates)
        raise ValueError(f"Image/mask shapes do not agree: {detail}")
    return shape


def _cellpose_seed_mask(
    image: ArrayLike,
    *,
    diameter: float | None = None,
    gpu: bool = True,
    expansion_distance: int = 20,
) -> np.ndarray:
    """Create an initial label prior from DAPI without importing Cellpose globally."""
    image_array = _load_2d_image(image)
    try:
        from cellpose import models
    except ImportError as exc:
        raise ImportError(
            "An initial mask was not supplied. Install cellpose to derive one "
            "from image, or pass initial_mask explicitly."
        ) from exc

    try:
        model = models.CellposeModel(gpu=gpu, pretrained_model="nuclei")
    except TypeError:
        model = models.CellposeModel(gpu=gpu, model_type="nuclei")
    result = model.eval(image_array, diameter=diameter)
    nuclei = np.asarray(result[0], dtype=np.uint32)
    if expansion_distance > 0:
        nuclei = expand_labels(nuclei, distance=expansion_distance).astype(np.uint32)
    if nuclei.max(initial=0) == 0:
        raise RuntimeError("Cellpose produced an empty initial mask")
    return nuclei


def _get_seed_mask(
    *,
    initial_mask: ArrayLike | None,
    image: ArrayLike | None,
    diameter: float | None,
    gpu: bool,
    expansion_distance: int,
) -> np.ndarray | None:
    if initial_mask is not None:
        return _load_label_mask(initial_mask)
    if image is not None:
        print("[INFO] No initial mask supplied; deriving a Cellpose prior from the image")
        return _cellpose_seed_mask(
            image,
            diameter=diameter,
            gpu=gpu,
            expansion_distance=expansion_distance,
        )
    return None


def _resolve_output_path(
    method: str,
    *,
    region: str,
    input_dir: PathLike | None,
    output_dir_prefix: PathLike | None,
    output_path: PathLike | None,
    input_image_type: str,
) -> Path:
    if input_image_type not in {"stitched", "retiled"}:
        raise ValueError("input_image_type must be 'stitched' or 'retiled'")
    if output_path is not None:
        result = Path(output_path)
    else:
        if output_dir_prefix is not None:
            root = Path(output_dir_prefix)
        elif input_dir is not None:
            root = Path(input_dir)
        else:
            raise ValueError("Provide output_path, output_dir_prefix, or input_dir")
        result = (
            root
            / region
            / "postprocessing"
            / "segmentation"
            / f"{region}_{method}_{input_image_type}_expanded.npz"
        )
    if result.suffix.lower() != ".npz":
        raise ValueError(f"Segmentation output must end in .npz: {result}")
    result.parent.mkdir(parents=True, exist_ok=True)
    return result


def save_segmentation_mask(
    labels: np.ndarray,
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> tuple[np.ndarray, coo_matrix]:
    """Validate and save a dense label image in the pipeline's sparse format."""
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Segmentation output already exists: {output_path}. Pass overwrite=True to replace it."
        )
    labels = _load_label_mask(np.asarray(labels))
    sparse = coo_matrix(labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_path, sparse, compressed=True)
    print(
        f"[INFO] Saved {int(labels.max(initial=0))} labels with shape "
        f"{labels.shape} to: {output_path}"
    )
    return labels, sparse


def _open_geojson(path: Path) -> dict:
    # Proseg 3.2 compresses polygon output even when the requested filename has
    # a plain ``.geojson`` suffix, so inspect the gzip magic bytes as well as
    # the filename.
    with path.open("rb") as raw:
        is_gzip = raw.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip or path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _geometry_parts(geometry: Mapping) -> Iterable[list]:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if kind == "Polygon":
        yield coordinates
    elif kind == "MultiPolygon":
        yield from coordinates


def polygons_to_label_mask(
    features: Iterable[Mapping],
    *,
    mask_shape: Sequence[int],
    coordinate_scale: float = 1.0,
    cell_id_property: str = "cell",
) -> np.ndarray:
    """Rasterize GeoJSON-like polygon features into a positive integer mask.

    ``coordinate_scale`` is the number of source-coordinate units per output
    pixel.  For polygons in microns and an output mask in image pixels, pass the
    image's microns-per-pixel value.
    """
    shape = _validate_shape(mask_shape)
    if coordinate_scale <= 0:
        raise ValueError("coordinate_scale must be > 0")

    feature_list = list(features)
    feature_list.sort(
        key=lambda feature: str(feature.get("properties", {}).get(cell_id_property, ""))
    )
    labels = np.zeros(shape, dtype=np.uint32)
    cell_to_label: dict[str, int] = {}
    polygons_drawn = 0

    for feature in feature_list:
        geometry = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        raw_cell_id = props.get(cell_id_property, feature.get("id"))
        key = str(raw_cell_id) if raw_cell_id is not None else f"feature-{len(cell_to_label)}"
        label = cell_to_label.setdefault(key, len(cell_to_label) + 1)

        for rings in _geometry_parts(geometry):
            if not rings:
                continue
            exterior = np.asarray(rings[0], dtype=float)
            if exterior.ndim != 2 or exterior.shape[0] < 3 or exterior.shape[1] < 2:
                continue
            x = exterior[:, 0] / coordinate_scale
            y = exterior[:, 1] / coordinate_scale
            rr, cc = draw_polygon(y, x, shape=shape)
            empty = labels[rr, cc] == 0
            labels[rr[empty], cc[empty]] = label
            polygons_drawn += 1

            # Remove holes only from the cell currently being drawn.
            for hole in rings[1:]:
                hole = np.asarray(hole, dtype=float)
                if hole.ndim != 2 or hole.shape[0] < 3 or hole.shape[1] < 2:
                    continue
                hrr, hcc = draw_polygon(
                    hole[:, 1] / coordinate_scale,
                    hole[:, 0] / coordinate_scale,
                    shape=shape,
                )
                own = labels[hrr, hcc] == label
                labels[hrr[own], hcc[own]] = 0

    if polygons_drawn == 0 or labels.max(initial=0) == 0:
        raise ValueError("No Polygon or MultiPolygon features were found")
    return labels


def geojson_to_sparse_mask(
    geojson_path: PathLike,
    output_path: PathLike,
    *,
    mask_shape: Sequence[int],
    coordinate_scale: float = 1.0,
    cell_id_property: str = "cell",
    overwrite: bool = False,
) -> tuple[np.ndarray, coo_matrix]:
    """Convert polygon GeoJSON (optionally gzipped) to the pipeline's mask format."""
    geojson_path = Path(geojson_path)
    if not geojson_path.is_file():
        raise FileNotFoundError(f"Polygon output not found: {geojson_path}")
    document = _open_geojson(geojson_path)
    labels = polygons_to_label_mask(
        document.get("features", []),
        mask_shape=mask_shape,
        coordinate_scale=coordinate_scale,
        cell_id_property=cell_id_property,
    )
    return save_segmentation_mask(labels, output_path, overwrite=overwrite)


def proseg_segmentation(
    transcripts: TableLike,
    image: ArrayLike | None = None,
    *,
    region: str,
    input_dir: PathLike | None = None,
    output_dir_prefix: PathLike | None = None,
    output_path: PathLike | None = None,
    initial_mask: ArrayLike | None = None,
    mask_shape: Sequence[int] | None = None,
    gene_col: str = "target",
    x_col: str = "xc",
    y_col: str = "yc",
    cell_id_col: str | None = None,
    cell_id_unassigned: str = "0",
    quality_col: str | None = None,
    min_quality: float | None = None,
    pixel_size_um: float = 1.0,
    proseg_command: Command = "proseg",
    nthreads: int | None = None,
    seed_diameter: float | None = None,
    seed_gpu: bool = True,
    seed_expansion_distance: int = 20,
    work_dir: PathLike | None = None,
    extra_args: Sequence[str] | None = None,
    input_image_type: str = "stitched",
    overwrite: bool = False,
) -> tuple[np.ndarray, coo_matrix]:
    """Run Proseg and convert its consensus polygons to a sparse label mask.

    Proseg needs a prior segmentation.  Supply ``cell_id_col`` when the table
    already contains prior assignments, ``initial_mask`` for an existing label
    image, or ``image`` to create a Cellpose prior automatically.
    """
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0")
    if seed_expansion_distance < 0:
        raise ValueError("seed_expansion_distance must be >= 0")
    final_path = _resolve_output_path(
        "proseg",
        region=region,
        input_dir=input_dir,
        output_dir_prefix=output_dir_prefix,
        output_path=output_path,
        input_image_type=input_image_type,
    )
    if final_path.exists() and not overwrite:
        raise FileExistsError(f"Segmentation output already exists: {final_path}")

    shape = _resolve_shape(image=image, initial_mask=initial_mask, mask_shape=mask_shape)
    table = normalize_transcript_table(
        transcripts,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        cell_id_col=cell_id_col,
        quality_col=quality_col,
        min_quality=min_quality,
    )
    seed = _get_seed_mask(
        initial_mask=initial_mask,
        image=image,
        diameter=seed_diameter,
        gpu=seed_gpu,
        expansion_distance=seed_expansion_distance,
    )
    if seed is None and cell_id_col is None:
        raise ValueError(
            "Proseg requires a prior: pass cell_id_col, initial_mask, or image"
        )
    if seed is not None and seed.shape != shape:
        raise ValueError(f"Initial mask shape {seed.shape} does not match output shape {shape}")

    work = Path(work_dir) if work_dir is not None else final_path.parent / f"{region}_proseg_work"
    work.mkdir(parents=True, exist_ok=True)
    transcript_path = work / "transcripts.csv.gz"
    proseg_table = table[["gene", "x", "y"]].copy()
    proseg_table["z"] = 0.0
    if "cell_id" in table:
        proseg_table["cell_id"] = table["cell_id"].fillna(cell_id_unassigned).astype(str)
    else:
        proseg_table["cell_id"] = cell_id_unassigned
    proseg_table.to_csv(transcript_path, index=False, compression="gzip")

    polygon_name = "cell-polygons.geojson"
    command = _as_command(proseg_command) + [
        "--gene-column", "gene",
        "--x-column", "x",
        "--y-column", "y",
        "--z-column", "z",
        "--ignore-z-coord",
        "--cell-id-column", "cell_id",
        "--cell-id-unassigned", str(cell_id_unassigned),
        "--coordinate-scale", str(pixel_size_um),
        "--output-path", str(work),
        "--output-spatialdata", "proseg-output.zarr",
        "--output-cell-polygons", polygon_name,
        "--overwrite",
    ]
    if nthreads is not None:
        if int(nthreads) <= 0:
            raise ValueError("nthreads must be positive")
        command += ["--nthreads", str(int(nthreads))]
    if seed is not None:
        seed_path = work / "initial-mask.npy"
        np.save(seed_path, seed.astype(np.uint32, copy=False))
        command += [
            "--cellpose-masks", str(seed_path),
            "--cellpose-scale", str(pixel_size_um),
        ]
    if extra_args:
        command += [str(arg) for arg in extra_args]
    command.append(str(transcript_path))
    _run_command(command, cwd=work)

    return geojson_to_sparse_mask(
        work / polygon_name,
        final_path,
        mask_shape=shape,
        coordinate_scale=pixel_size_um,
        cell_id_property="cell",
        overwrite=overwrite,
    )


def segger_segmentation(
    transcripts: TableLike,
    image: ArrayLike | None = None,
    *,
    region: str,
    input_dir: PathLike | None = None,
    output_dir_prefix: PathLike | None = None,
    output_path: PathLike | None = None,
    initial_mask: ArrayLike | None = None,
    mask_shape: Sequence[int] | None = None,
    gene_col: str = "target",
    x_col: str = "xc",
    y_col: str = "yc",
    quality_col: str | None = None,
    min_quality: float | None = None,
    pixel_size_um: float = 1.0,
    segger_command: Command = "segger",
    segger_python: Command = sys.executable,
    seed_diameter: float | None = None,
    seed_gpu: bool = True,
    seed_expansion_distance: int = 20,
    work_dir: PathLike | None = None,
    extra_args: Sequence[str] | None = None,
    boundary_method: str = "delaunay",
    input_image_type: str = "stitched",
    overwrite: bool = False,
) -> tuple[np.ndarray, coo_matrix]:
    """Run Segger in its own environment and save its polygons as ``.npz``.

    The current Segger release requires standardized transcripts plus seed
    boundaries.  The wrapper creates a minimal Xenium-compatible bundle from an
    initial mask.  If only ``image`` is supplied, Cellpose supplies that prior.

    ``segger_command`` and ``segger_python`` may be explicit command prefixes,
    which makes a separate environment usable from a notebook, for example
    ``["conda", "run", "-n", "segger", "segger"]`` and
    ``["conda", "run", "-n", "segger", "python"]``.
    """
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be > 0")
    if seed_expansion_distance < 0:
        raise ValueError("seed_expansion_distance must be >= 0")
    if boundary_method not in {"delaunay", "convex_hull"}:
        raise ValueError("boundary_method must be 'delaunay' or 'convex_hull'")
    final_path = _resolve_output_path(
        "segger",
        region=region,
        input_dir=input_dir,
        output_dir_prefix=output_dir_prefix,
        output_path=output_path,
        input_image_type=input_image_type,
    )
    if final_path.exists() and not overwrite:
        raise FileExistsError(f"Segmentation output already exists: {final_path}")

    shape = _resolve_shape(image=image, initial_mask=initial_mask, mask_shape=mask_shape)
    table = normalize_transcript_table(
        transcripts,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        quality_col=quality_col,
        min_quality=min_quality,
    )
    seed = _get_seed_mask(
        initial_mask=initial_mask,
        image=image,
        diameter=seed_diameter,
        gpu=seed_gpu,
        expansion_distance=0,
    )
    if seed is None:
        raise ValueError("Segger requires initial_mask or image to construct seed boundaries")
    if seed.shape != shape:
        raise ValueError(f"Initial mask shape {seed.shape} does not match output shape {shape}")

    work = Path(work_dir) if work_dir is not None else final_path.parent / f"{region}_segger_work"
    input_bundle = work / "input"
    segger_output = work / "output"
    export_output = work / "export"
    for directory in (work, input_bundle, segger_output, export_output):
        directory.mkdir(parents=True, exist_ok=True)
    normalized_csv = work / "transcripts.csv"
    table[["gene", "x", "y"]].to_csv(normalized_csv, index=False)
    seed_path = work / "initial-mask.npy"
    np.save(seed_path, seed.astype(np.uint32, copy=False))

    bridge = Path(__file__).with_name("_segger_bridge.py")
    prepare_command = _as_command(segger_python) + [
        str(bridge),
        "prepare",
        "--transcripts", str(normalized_csv),
        "--mask", str(seed_path),
        "--output-directory", str(input_bundle),
        "--pixel-size-um", str(pixel_size_um),
        "--cell-expansion-distance", str(int(seed_expansion_distance)),
    ]
    _run_command(prepare_command, cwd=work)

    segment_command = _as_command(segger_command) + [
        "segment",
        "-i", str(input_bundle),
        "-o", str(segger_output),
    ]
    if extra_args:
        segment_command += [str(arg) for arg in extra_args]
    _run_command(segment_command, cwd=work)

    segmentation_path = segger_output / "segger_segmentation.parquet"
    if not segmentation_path.is_file():
        raise FileNotFoundError(f"Segger did not create: {segmentation_path}")
    export_command = _as_command(segger_command) + [
        "export", "boundaries",
        "-s", str(segmentation_path),
        "-i", str(input_bundle),
        "-o", str(export_output),
        "--method", boundary_method,
    ]
    _run_command(export_command, cwd=work)

    boundary_path = export_output / "cell_boundaries.parquet"
    rasterize_command = _as_command(segger_python) + [
        str(bridge),
        "rasterize",
        "--boundaries", str(boundary_path),
        "--output", str(final_path),
        "--height", str(shape[0]),
        "--width", str(shape[1]),
        "--pixel-size-um", str(pixel_size_um),
    ]
    _run_command(rasterize_command, cwd=work)
    sparse = load_npz(final_path).tocoo()
    labels = sparse.toarray().astype(np.uint32)
    print(f"[INFO] Saved Segger mask to: {final_path}")
    return labels, sparse


def prepare_bidcell_reference(
    reference_adata,
    *,
    cell_type_col: str,
    spatial_genes: Iterable[str],
    output_dir: PathLike,
    layer: str | None = None,
    use_raw: bool = False,
    reference_name: str | None = None,
    positive_quantile: float = 0.9,
    negative_quantile: float = 0.1,
    common_positive_fraction: float = 1 / 3,
    min_cells_per_type: int = 1,
    allow_missing_genes: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Create BIDCell reference and marker CSVs from an annotated AnnData.

    The returned dictionary can be passed directly to
    :func:`bidcell_segmentation` with ``**reference_files``. Expression values
    are averaged per cell type without further normalization, so ``X`` (or the
    selected layer/raw matrix) should contain non-negative counts or normalized
    expression rather than centered/scaled values.
    """
    if not cell_type_col:
        raise ValueError("cell_type_col must be a non-empty AnnData obs column")
    if layer is not None and use_raw:
        raise ValueError("Choose either layer or use_raw=True, not both")
    if not 0 <= negative_quantile < positive_quantile <= 1:
        raise ValueError(
            "Require 0 <= negative_quantile < positive_quantile <= 1"
        )
    if not 0 < common_positive_fraction <= 1:
        raise ValueError("common_positive_fraction must be in (0, 1]")
    if int(min_cells_per_type) <= 0:
        raise ValueError("min_cells_per_type must be positive")

    genes = []
    seen = set()
    for gene in spatial_genes:
        if pd.isna(gene):
            continue
        name = str(gene)
        if name and name not in seen:
            genes.append(name)
            seen.add(name)
    if not genes:
        raise ValueError("spatial_genes must contain at least one gene")

    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "Automatic BIDCell reference generation requires anndata"
        ) from exc

    if isinstance(reference_adata, (str, os.PathLike)):
        adata_path = Path(reference_adata)
        if not adata_path.is_file():
            raise FileNotFoundError(f"AnnData reference not found: {adata_path}")
        adata = ad.read_h5ad(adata_path)
        default_reference_name = adata_path.stem
    elif isinstance(reference_adata, ad.AnnData):
        adata = reference_adata
        default_reference_name = "anndata_reference"
    else:
        raise TypeError("reference_adata must be an AnnData object or .h5ad path")

    if cell_type_col not in adata.obs:
        raise KeyError(f"AnnData obs does not contain {cell_type_col!r}")
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but AnnData.raw is empty")
        matrix = adata.raw.X
        var_names = pd.Index(adata.raw.var_names.astype(str))
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"AnnData layers does not contain {layer!r}")
        matrix = adata.layers[layer]
        var_names = pd.Index(adata.var_names.astype(str))
    else:
        matrix = adata.X
        var_names = pd.Index(adata.var_names.astype(str))

    if var_names.has_duplicates:
        duplicates = sorted(set(var_names[var_names.duplicated()].tolist()))
        raise ValueError(f"AnnData gene names are duplicated: {duplicates[:10]}")
    missing_genes = [gene for gene in genes if gene not in var_names]
    if missing_genes and not allow_missing_genes:
        preview = ", ".join(missing_genes[:10])
        suffix = "..." if len(missing_genes) > 10 else ""
        raise ValueError(
            f"AnnData is missing {len(missing_genes)} spatial genes: {preview}{suffix}. "
            "Pass allow_missing_genes=True to include them as unmarked zero-expression genes."
        )
    available_genes = [gene for gene in genes if gene in var_names]
    if not available_genes:
        raise ValueError("AnnData and spatial transcript panel have no genes in common")

    labels_raw = adata.obs[cell_type_col]
    valid_labels = labels_raw.notna() & labels_raw.astype("string").str.strip().ne("")
    if not valid_labels.all():
        print(f"[INFO] Ignoring {int((~valid_labels).sum())} cells without a cell type")
    labels = labels_raw.astype("string")
    counts = labels.loc[valid_labels].value_counts()
    cell_types = sorted(
        counts.loc[counts >= int(min_cells_per_type)].index.astype(str).tolist()
    )
    if not cell_types:
        raise ValueError("No annotated cell types satisfy min_cells_per_type")
    excluded_types = counts.loc[counts < int(min_cells_per_type)]
    if not excluded_types.empty:
        print(
            f"[INFO] Excluding {len(excluded_types)} cell types with fewer than "
            f"{int(min_cells_per_type)} cells"
        )

    keep = valid_labels.to_numpy() & labels.isin(cell_types).to_numpy()
    kept_labels = labels.to_numpy(dtype=str)[keep]
    gene_indices = var_names.get_indexer(available_genes)
    selected = matrix[keep, :][:, gene_indices]
    averages = np.zeros((len(cell_types), len(genes)), dtype=np.float64)
    gene_positions = {gene: position for position, gene in enumerate(genes)}
    available_positions = np.asarray(
        [gene_positions[gene] for gene in available_genes]
    )
    for row, cell_type in enumerate(cell_types):
        values = np.asarray(selected[kept_labels == cell_type].mean(axis=0)).ravel()
        averages[row, available_positions] = values

    if not np.isfinite(averages).all():
        raise ValueError("AnnData reference produced non-finite mean expression values")
    if np.any(averages < 0):
        raise ValueError(
            "BIDCell reference expression must be non-negative; select a counts or "
            "log-normalized layer instead of centered/scaled expression"
        )

    mean_expression = pd.DataFrame(averages, index=cell_types, columns=genes)
    positive_thresholds = mean_expression.quantile(positive_quantile, axis=1)
    negative_thresholds = mean_expression.quantile(negative_quantile, axis=1)
    positive = mean_expression.ge(positive_thresholds, axis=0) & mean_expression.gt(0)
    negative = mean_expression.le(negative_thresholds, axis=0)

    # A marker must occur in at least two cell types before it can reasonably
    # be called "common"; this also keeps the one-third heuristic usable for
    # small references containing only two or three cell types.
    common_count = max(
        2, int(np.ceil(common_positive_fraction * len(cell_types)))
    )
    common_positive = positive.sum(axis=0) >= common_count
    positive.loc[:, common_positive] = False
    negative &= ~positive
    if missing_genes:
        positive.loc[:, missing_genes] = False
        negative.loc[:, missing_genes] = False
        print(
            f"[INFO] Added {len(missing_genes)} missing spatial genes as unmarked "
            "zero-expression reference columns"
        )
    if not positive.to_numpy().any():
        raise ValueError(
            "Marker selection produced no positive markers; adjust the quantile or "
            "common_positive_fraction"
        )
    if not negative.to_numpy().any():
        raise ValueError("Marker selection produced no negative markers")

    reference = mean_expression.reset_index(drop=True)
    reference["ct_idx"] = np.arange(len(cell_types), dtype=int)
    reference["cell_type"] = cell_types
    reference["atlas"] = reference_name or default_reference_name
    positive.index.name = None
    negative.index.name = None

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "reference_path": destination / "reference.csv",
        "positive_markers_path": destination / "positive_markers.csv",
        "negative_markers_path": destination / "negative_markers.csv",
    }
    existing = [str(path) for path in files.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"BIDCell reference output already exists: {existing}. "
            "Pass overwrite=True to replace it."
        )
    reference.to_csv(files["reference_path"])
    positive.astype(np.int8).to_csv(files["positive_markers_path"])
    negative.astype(np.int8).to_csv(files["negative_markers_path"])
    print(
        f"[INFO] Prepared BIDCell reference for {len(cell_types)} cell types and "
        f"{len(genes)} spatial genes in: {destination}"
    )
    return files


def _validate_bidcell_reference_files(
    reference_path: Path,
    positive_markers_path: Path,
    negative_markers_path: Path,
    *,
    spatial_genes: Iterable[str],
) -> None:
    reference = pd.read_csv(reference_path, index_col=0)
    positive = pd.read_csv(positive_markers_path, index_col=0)
    negative = pd.read_csv(negative_markers_path, index_col=0)
    metadata = ["ct_idx", "cell_type", "atlas"]
    if reference.columns[-3:].tolist() != metadata:
        raise ValueError(
            "BIDCell reference must end with columns: ct_idx, cell_type, atlas"
        )
    genes = list(dict.fromkeys(str(gene) for gene in spatial_genes))
    for name, frame in (
        ("reference", reference),
        ("positive markers", positive),
        ("negative markers", negative),
    ):
        missing = sorted(set(genes) - set(frame.columns))
        if missing:
            raise ValueError(f"BIDCell {name} is missing spatial genes: {missing[:10]}")
    if set(positive.index.astype(str)) != set(negative.index.astype(str)):
        raise ValueError("Positive and negative marker files have different cell types")
    reference_types = set(reference["cell_type"].astype(str))
    marker_types = set(positive.index.astype(str))
    if reference_types != marker_types:
        raise ValueError("Reference and marker files have different cell types")
    if not positive.index.is_unique or not negative.index.is_unique:
        raise ValueError("BIDCell marker files must have one row per cell type")
    mapping = reference[["ct_idx", "cell_type"]].drop_duplicates()
    if (
        mapping.groupby("cell_type")["ct_idx"].nunique().max() != 1
        or mapping.groupby("ct_idx")["cell_type"].nunique().max() != 1
    ):
        raise ValueError("Each BIDCell cell type must map to exactly one ct_idx")
    expected_ids = set(range(len(marker_types)))
    actual_ids = set(mapping["ct_idx"].astype(int))
    if actual_ids != expected_ids:
        raise ValueError("BIDCell ct_idx values must be contiguous integers starting at 0")
    pos_values = positive.loc[:, genes].to_numpy()
    neg_values = negative.loc[:, genes].to_numpy()
    reference_values = (
        reference.loc[:, genes].apply(pd.to_numeric, errors="coerce").to_numpy()
    )
    if not np.isfinite(reference_values).all() or np.any(reference_values < 0):
        raise ValueError("BIDCell reference expression must be finite and non-negative")
    if not np.isin(pos_values, [0, 1]).all() or not np.isin(neg_values, [0, 1]).all():
        raise ValueError("BIDCell marker matrices must contain only 0 and 1")
    if np.any((pos_values == 1) & (neg_values == 1)):
        raise ValueError("A BIDCell gene cannot be both positive and negative for one cell type")


def _build_bidcell_config(
    *,
    data_dir: Path,
    transcripts_path: Path,
    image_path: Path,
    reference_path: Path,
    positive_markers_path: Path,
    negative_markers_path: Path,
    pixel_size_um: float,
    target_pixel_size_um: float,
    cpus: int,
    patch_size: int,
    elongated_cell_types: Sequence[str],
    total_steps: int,
    test_step: int,
    seed_diameter: int | None,
) -> dict:
    return {
        "cpus": int(cpus),
        "files": {
            "data_dir": str(data_dir.resolve()),
            "fp_dapi": str(image_path.resolve()),
            "fp_transcripts": str(transcripts_path.resolve()),
            "fp_ref": str(reference_path.resolve()),
            "fp_pos_markers": str(positive_markers_path.resolve()),
            "fp_neg_markers": str(negative_markers_path.resolve()),
        },
        "nuclei_fovs": {"stitch_nuclei_fovs": False},
        "nuclei": {"diameter": seed_diameter},
        "transcripts": {
            "shift_to_origin": False,
            "x_col": "x",
            "y_col": "y",
            "gene_col": "gene",
            "transcripts_to_filter": [],
        },
        "affine": {
            "target_pix_um": float(target_pixel_size_um),
            "base_pix_x": float(pixel_size_um),
            "base_pix_y": float(pixel_size_um),
            "base_ts_x": float(pixel_size_um),
            "base_ts_y": float(pixel_size_um),
            "global_shift_x": 0,
            "global_shift_y": 0,
        },
        "model_params": {
            "name": "custom",
            "patch_size": int(patch_size),
            "elongated": list(elongated_cell_types),
        },
        "training_params": {"total_epochs": 1, "total_steps": int(total_steps)},
        "testing_params": {"test_epoch": 1, "test_step": int(test_step)},
        "experiment_dirs": {"dir_id": "last"},
    }


def bidcell_segmentation(
    transcripts: TableLike,
    image: PathLike,
    *,
    region: str,
    reference_path: PathLike | None = None,
    positive_markers_path: PathLike | None = None,
    negative_markers_path: PathLike | None = None,
    reference_adata=None,
    cell_type_col: str | None = None,
    reference_layer: str | None = None,
    reference_use_raw: bool = False,
    reference_name: str | None = None,
    positive_quantile: float = 0.9,
    negative_quantile: float = 0.1,
    common_positive_fraction: float = 1 / 3,
    min_reference_cells_per_type: int = 1,
    allow_missing_reference_genes: bool = False,
    input_dir: PathLike | None = None,
    output_dir_prefix: PathLike | None = None,
    output_path: PathLike | None = None,
    gene_col: str = "target",
    x_col: str = "xc",
    y_col: str = "yc",
    quality_col: str | None = None,
    min_quality: float | None = None,
    pixel_size_um: float = 1.0,
    target_pixel_size_um: float = 1.0,
    bidcell_python: Command = sys.executable,
    cpus: int = 8,
    patch_size: int = 48,
    elongated_cell_types: Sequence[str] = (),
    total_steps: int = 4000,
    test_step: int | None = None,
    seed_diameter: int | None = None,
    gpu_id: int | None = None,
    work_dir: PathLike | None = None,
    input_image_type: str = "stitched",
    overwrite: bool = False,
) -> tuple[np.ndarray, coo_matrix]:
    """Run BIDCell and convert its connected TIFF to the pipeline's ``.npz``.

    BIDCell requires an image and a biological reference. Pass either
    ``reference_adata`` plus ``cell_type_col`` to generate its internal CSVs
    automatically, or the three legacy CSV paths. The wrapper invokes BIDCell,
    locates the connected label TIFF, resizes it back to the input-image shape,
    and saves the common sparse mask.
    """
    if pixel_size_um <= 0 or target_pixel_size_um <= 0:
        raise ValueError("pixel_size_um and target_pixel_size_um must be > 0")
    if cpus <= 0 or patch_size <= 0 or total_steps <= 0:
        raise ValueError("cpus, patch_size, and total_steps must be positive")
    if test_step is not None and (test_step <= 0 or test_step > total_steps):
        raise ValueError("test_step must be positive and no larger than total_steps")
    if gpu_id is not None and gpu_id < 0:
        raise ValueError("gpu_id must be >= 0")
    final_path = _resolve_output_path(
        "bidcell",
        region=region,
        input_dir=input_dir,
        output_dir_prefix=output_dir_prefix,
        output_path=output_path,
        input_image_type=input_image_type,
    )
    if final_path.exists() and not overwrite:
        raise FileExistsError(f"Segmentation output already exists: {final_path}")

    image_path = Path(image)
    if not image_path.is_file():
        raise FileNotFoundError(f"BIDCell image not found: {image_path}")
    image_shape = _load_2d_image(image_path).shape

    table = normalize_transcript_table(
        transcripts,
        gene_col=gene_col,
        x_col=x_col,
        y_col=y_col,
        quality_col=quality_col,
        min_quality=min_quality,
    )
    work = Path(work_dir) if work_dir is not None else final_path.parent / f"{region}_bidcell_work"
    work.mkdir(parents=True, exist_ok=True)
    spatial_genes = pd.unique(table["gene"]).tolist()

    explicit_references = (
        reference_path,
        positive_markers_path,
        negative_markers_path,
    )
    if reference_adata is not None:
        if any(value is not None for value in explicit_references):
            raise ValueError(
                "Pass reference_adata or the three reference CSV paths, not both"
            )
        if cell_type_col is None:
            raise ValueError("cell_type_col is required with reference_adata")
        generated = prepare_bidcell_reference(
            reference_adata,
            cell_type_col=cell_type_col,
            spatial_genes=spatial_genes,
            output_dir=work / "bidcell_reference",
            layer=reference_layer,
            use_raw=reference_use_raw,
            reference_name=reference_name,
            positive_quantile=positive_quantile,
            negative_quantile=negative_quantile,
            common_positive_fraction=common_positive_fraction,
            min_cells_per_type=min_reference_cells_per_type,
            allow_missing_genes=allow_missing_reference_genes,
            overwrite=overwrite,
        )
        references = [
            generated["reference_path"],
            generated["positive_markers_path"],
            generated["negative_markers_path"],
        ]
    else:
        if any(value is None for value in explicit_references):
            raise ValueError(
                "Pass reference_adata with cell_type_col, or all three of "
                "reference_path, positive_markers_path, and negative_markers_path"
            )
        references = [Path(value) for value in explicit_references]
        missing = [str(path) for path in references if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"BIDCell reference file(s) not found: {missing}")

    _validate_bidcell_reference_files(
        references[0],
        references[1],
        references[2],
        spatial_genes=spatial_genes,
    )
    transcript_path = work / "transcripts.csv"
    table[["gene", "x", "y"]].to_csv(transcript_path, index=False)
    effective_test_step = int(total_steps if test_step is None else test_step)
    config = _build_bidcell_config(
        data_dir=work,
        transcripts_path=transcript_path,
        image_path=image_path,
        reference_path=references[0],
        positive_markers_path=references[1],
        negative_markers_path=references[2],
        pixel_size_um=pixel_size_um,
        target_pixel_size_um=target_pixel_size_um,
        cpus=cpus,
        patch_size=patch_size,
        elongated_cell_types=elongated_cell_types,
        total_steps=total_steps,
        test_step=effective_test_step,
        seed_diameter=seed_diameter,
    )
    config_path = work / "bidcell-config.yaml"
    # JSON is valid YAML and avoids adding PyYAML to the base ISS environment.
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    runner = Path(__file__).with_name("_bidcell_runner.py")
    command = _as_command(bidcell_python) + [str(runner), str(config_path)]
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    _run_command(command, cwd=work, env=env)

    candidates = list((work / "model_outputs").glob("*/test_output/*_connected.tif"))
    if not candidates:
        candidates = list((work / "model_outputs").glob("**/*_connected.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"BIDCell completed but no *_connected.tif was found under {work / 'model_outputs'}"
        )
    connected_path = max(candidates, key=lambda path: path.stat().st_mtime)
    labels = _load_label_mask(connected_path)
    if labels.shape != image_shape:
        print(f"[INFO] Resizing BIDCell mask from {labels.shape} to {image_shape}")
        labels = resize(
            labels,
            image_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.uint32)
    return save_segmentation_mask(labels, final_path, overwrite=overwrite)


def run_transcript_segmentation(
    method: str,
    transcripts: TableLike,
    image: ArrayLike | None = None,
    **kwargs,
) -> tuple[np.ndarray, coo_matrix]:
    """Dispatch to a transcript-aware segmentation wrapper by method name."""
    methods = {
        "segger": segger_segmentation,
        "proseg": proseg_segmentation,
        "bidcell": bidcell_segmentation,
    }
    key = str(method).lower()
    if key not in methods:
        raise ValueError(f"Unknown method {method!r}; choose one of {sorted(methods)}")
    return methods[key](transcripts, image, **kwargs)


__all__ = [
    "bidcell_segmentation",
    "geojson_to_sparse_mask",
    "load_transcript_table",
    "normalize_transcript_table",
    "polygons_to_label_mask",
    "prepare_bidcell_reference",
    "proseg_segmentation",
    "run_transcript_segmentation",
    "save_segmentation_mask",
    "segger_segmentation",
]
