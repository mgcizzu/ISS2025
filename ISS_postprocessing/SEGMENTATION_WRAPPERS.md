# Transcript-aware segmentation wrappers

`ISS_postprocessing.segmentation` exposes adapters for **Segger**, **Proseg**,
and **BIDCell**. The external programs keep their native intermediate outputs,
but every adapter finishes with the same pipeline artifact:

```text
<output root>/<region>/postprocessing/segmentation/
    <region>_<method>_stitched_expanded.npz
```

The `.npz` is a SciPy sparse two-dimensional label image. Background is `0`
and cells are positive integers. It can be loaded with the existing
`SEG.load_sparse_mask(...)` and inspected with
`SEG.inspect_and_work_with_segmentation(...)` functions.

## Common input contract

All three wrappers take a transcript table as their first argument and an
optional image as their second argument:

```python
import ISS_postprocessing.segmentation as SEG

labels, labels_coo = SEG.run_transcript_segmentation(
    "proseg",
    transcripts="/data/R1/decoding/2_decoded/R1_decoded.csv",
    image="/data/R1/preprocessing/Cycle1/3_stitched/Cycle1_ch4.tif",
    region="R1",
    input_dir="/data",
    gene_col="target",
    x_col="xc",
    y_col="yc",
    pixel_size_um=0.325,
)
```

`transcripts` may be a pandas `DataFrame` or a CSV, TSV, compressed CSV/TSV,
or Parquet path. The decoded ISS defaults are `target`, `xc`, and `yc`.
Coordinates are assumed to be image pixels. `pixel_size_um` is used when a
method works in physical coordinates.

An image, initial mask, or explicit `mask_shape=(height, width)` is required to
define the output canvas. When Segger or Proseg receives an image but no
`initial_mask`, the wrapper generates a Cellpose prior. Passing an existing
mask is faster and more reproducible.

## Why separate environments are supported

The dependency stacks are different:

- Proseg is a Rust executable and can be called directly from the normal ISS
  notebook environment after `cargo install proseg`.
- Segger currently requires Python 3.11+ and a CUDA/RAPIDS stack, while the ISS
  environment uses Python 3.10 and CUDA 11.8-era packages. Keep Segger in its
  upstream-recommended environment.
- BIDCell has its own PyTorch environment and requires a single-cell reference
  plus positive and negative marker files.

Commands are passed as argument lists and never through a shell. For example,
Segger can be launched from an ISS notebook with:

```python
labels, labels_coo = SEG.segger_segmentation(
    transcripts=transcripts_file,
    image=dapi_file,
    region="R1",
    input_dir=input_dir,
    pixel_size_um=0.325,
    segger_command=["conda", "run", "-n", "segger", "segger"],
    segger_python=["conda", "run", "-n", "segger", "python"],
)
```

The wrapper prepares a minimal standardized input bundle, runs `segger
segment`, exports cell boundaries, rasterizes them at the original image
resolution, and writes the standard sparse mask.

## Proseg

Proseg needs a prior estimate of the cells or nuclei. Use one of:

1. `cell_id_col="..."` if the transcript table already has preliminary cell
   assignments;
2. `initial_mask=...` for a NumPy, TIFF, or sparse `.npz` label mask; or
3. `image=...` to generate a Cellpose prior automatically.

```python
labels, labels_coo = SEG.proseg_segmentation(
    transcripts_file,
    dapi_file,
    region="R1",
    input_dir=input_dir,
    pixel_size_um=0.325,
    nthreads=16,
    # Proseg-native options can be appended when needed:
    extra_args=["--voxel-size", "1.0", "--samples", "200"],
)
```

The wrapper requests Proseg's consensus 2D GeoJSON polygons and rasterizes
them. Proseg is stochastic, so repeated runs can differ slightly.

See `Notebooks/Proseg segmentation.ipynb` for the notebook workflow.

## Segger

Segger requires initial boundaries for training. Supply `initial_mask`, or a
DAPI image from which the wrapper can generate a Cellpose seed. Its work
directory contains the standardized input, native segmentation Parquet, and
exported boundary GeoParquet for reproducibility.

`extra_args` is forwarded to `segger segment`, for example:

```python
extra_args=["--n-epochs", "20", "--prediction-mode", "cell"]
```

## BIDCell

BIDCell always requires a DAPI image and three biological reference files:

```python
labels, labels_coo = SEG.bidcell_segmentation(
    transcripts_file,
    dapi_file,
    region="R1",
    input_dir=input_dir,
    pixel_size_um=0.325,
    reference_path="/refs/sc_reference.csv",
    positive_markers_path="/refs/markers_positive.csv",
    negative_markers_path="/refs/markers_negative.csv",
    bidcell_python=["conda", "run", "-n", "bidcell", "python"],
    cpus=8,
    total_steps=4000,
)
```

The generated configuration uses the source image/transcript pixel size and a
configurable `target_pixel_size_um`. BIDCell's final `*_connected.tif` is
resized to the source DAPI dimensions with nearest-neighbour interpolation and
then written as the common sparse mask.

## Intermediate files and reruns

Each wrapper keeps method-native intermediate files next to the final mask in
`<region>_<method>_work/`. This is intentional: the files are useful for
diagnostics and preserve information that cannot be represented in the raster
mask. Final masks are not replaced unless `overwrite=True` is passed.
