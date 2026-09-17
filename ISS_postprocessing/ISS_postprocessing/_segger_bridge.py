"""File-format bridge executed inside a Segger environment.

This module is intentionally standalone: it lets the ISS notebook remain in a
Python 3.10 environment while Segger (currently Python >=3.11 with a separate
CUDA/RAPIDS stack) performs Parquet/GeoParquet work in its own environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects
from scipy.sparse import coo_matrix, save_npz
from skimage.draw import polygon as draw_polygon
from skimage.measure import find_contours
from skimage.segmentation import expand_labels


def _mask_vertices(mask: np.ndarray, scale: float) -> pd.DataFrame:
    cell_ids: list[str] = []
    vertex_x: list[float] = []
    vertex_y: list[float] = []

    # ``mask == label`` over the full image for every cell is quadratic in the
    # number of labels and becomes unusable for whole-slide masks. Find all
    # bounding boxes in one pass, then contour only the small crop belonging to
    # each label. Padding supplies a background rim for cells whose bounding
    # box is completely filled (or touches the image edge).
    for label, bounds in enumerate(find_objects(mask), start=1):
        if bounds is None:
            continue
        y_slice, x_slice = bounds
        cropped = np.pad(mask[y_slice, x_slice] == label, 1)
        contours = find_contours(cropped, 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        if len(contour) < 3:
            continue
        y_offset = int(y_slice.start) - 1
        x_offset = int(x_slice.start) - 1
        for y, x in contour:
            cell_ids.append(str(label))
            vertex_x.append((x + x_offset) * scale)
            vertex_y.append((y + y_offset) * scale)
    if not cell_ids:
        raise ValueError("The initial mask does not contain usable boundaries")
    return pd.DataFrame(
        {"cell_id": cell_ids, "vertex_x": vertex_x, "vertex_y": vertex_y}
    )


def prepare(args) -> None:
    table = pd.read_csv(args.transcripts)
    mask = np.load(args.mask).astype(np.uint32, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D seed mask, got {mask.shape}")
    if args.pixel_size_um <= 0:
        raise ValueError("pixel-size-um must be > 0")
    if args.cell_expansion_distance < 0:
        raise ValueError("cell-expansion-distance must be >= 0")

    if args.cell_expansion_distance == 0:
        cell_mask = mask
    else:
        cell_mask = expand_labels(mask, distance=args.cell_expansion_distance).astype(
            np.uint32
        )
    xs = np.floor(table["x"].to_numpy(dtype=float)).astype(np.int64)
    ys = np.floor(table["y"].to_numpy(dtype=float)).astype(np.int64)
    in_bounds = (xs >= 0) & (xs < mask.shape[1]) & (ys >= 0) & (ys < mask.shape[0])
    nucleus_ids = np.zeros(len(table), dtype=np.uint32)
    cell_ids = np.zeros(len(table), dtype=np.uint32)
    nucleus_ids[in_bounds] = mask[ys[in_bounds], xs[in_bounds]]
    cell_ids[in_bounds] = cell_mask[ys[in_bounds], xs[in_bounds]]

    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    tx = pd.DataFrame(
        {
            "x_location": table["x"].to_numpy(dtype=float) * args.pixel_size_um,
            "y_location": table["y"].to_numpy(dtype=float) * args.pixel_size_um,
            "feature_name": table["gene"].astype(str),
            "cell_id": [str(value) if value else "UNASSIGNED" for value in cell_ids],
            "overlaps_nucleus": (nucleus_ids > 0).astype(np.int8),
            "qv": np.full(len(table), 100, dtype=np.int16),
        }
    )
    tx.to_parquet(out / "transcripts.parquet", index=False)
    _mask_vertices(cell_mask, args.pixel_size_um).to_parquet(
        out / "cell_boundaries.parquet", index=False
    )
    _mask_vertices(mask, args.pixel_size_um).to_parquet(
        out / "nucleus_boundaries.parquet", index=False
    )
    (out / "experiment.xenium").write_text(
        json.dumps({"analysis_sw_version": "xenium-2.0.0"}), encoding="utf-8"
    )


def _parts(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type == "MultiPolygon":
        yield from geometry.geoms


def rasterize(args) -> None:
    import geopandas as gpd

    if args.pixel_size_um <= 0:
        raise ValueError("pixel-size-um must be > 0")
    boundaries = gpd.read_parquet(args.boundaries).sort_index()
    shape = (args.height, args.width)
    mask = np.zeros(shape, dtype=np.uint32)
    for label, (_, row) in enumerate(boundaries.iterrows(), start=1):
        for poly in _parts(row.geometry):
            exterior = np.asarray(poly.exterior.coords)
            rr, cc = draw_polygon(
                exterior[:, 1] / args.pixel_size_um,
                exterior[:, 0] / args.pixel_size_um,
                shape=shape,
            )
            empty = mask[rr, cc] == 0
            mask[rr[empty], cc[empty]] = label
            for ring in poly.interiors:
                hole = np.asarray(ring.coords)
                hrr, hcc = draw_polygon(
                    hole[:, 1] / args.pixel_size_um,
                    hole[:, 0] / args.pixel_size_um,
                    shape=shape,
                )
                own = mask[hrr, hcc] == label
                mask[hrr[own], hcc[own]] = 0
    if mask.max(initial=0) == 0:
        raise ValueError("Segger boundary output did not contain any rasterizable cells")
    save_npz(args.output, coo_matrix(mask), compressed=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--transcripts", required=True)
    prep.add_argument("--mask", required=True)
    prep.add_argument("--output-directory", required=True)
    prep.add_argument("--pixel-size-um", type=float, required=True)
    prep.add_argument("--cell-expansion-distance", type=int, default=20)
    prep.set_defaults(func=prepare)

    rast = subparsers.add_parser("rasterize")
    rast.add_argument("--boundaries", required=True)
    rast.add_argument("--output", required=True)
    rast.add_argument("--height", type=int, required=True)
    rast.add_argument("--width", type=int, required=True)
    rast.add_argument("--pixel-size-um", type=float, required=True)
    rast.set_defaults(func=rasterize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
