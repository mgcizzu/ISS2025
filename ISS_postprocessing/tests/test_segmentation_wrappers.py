import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from tifffile import imwrite

from ISS_postprocessing import segmentation_wrappers as wrappers
from ISS_postprocessing._segger_bridge import _mask_vertices


class TranscriptWrapperTests(unittest.TestCase):
    @staticmethod
    def _write_reference_anndata(path):
        import anndata as ad

        alpha = np.tile([10.0, 1.0, 2.0, 0.0], (6, 1))
        beta = np.tile([1.0, 0.0, 10.0, 2.0], (6, 1))
        adata = ad.AnnData(
            X=csr_matrix(np.vstack([alpha, beta])),
            obs=pd.DataFrame(
                {"cell_type": ["Alpha"] * 6 + ["Beta"] * 6},
                index=[f"cell-{i}" for i in range(12)],
            ),
            var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "GeneD"]),
        )
        adata.write_h5ad(path)

    def test_prepare_bidcell_reference_from_h5ad(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adata_path = root / "reference.h5ad"
            self._write_reference_anndata(adata_path)

            files = wrappers.prepare_bidcell_reference(
                adata_path,
                cell_type_col="cell_type",
                spatial_genes=["GeneA", "GeneB", "GeneC", "GeneD"],
                output_dir=root / "bidcell-reference",
            )

            reference = pd.read_csv(files["reference_path"], index_col=0)
            positive = pd.read_csv(files["positive_markers_path"], index_col=0)
            negative = pd.read_csv(files["negative_markers_path"], index_col=0)
            self.assertEqual(
                reference.columns.tolist(),
                ["GeneA", "GeneB", "GeneC", "GeneD", "ct_idx", "cell_type", "atlas"],
            )
            self.assertEqual(set(positive.index), {"Alpha", "Beta"})
            self.assertEqual(positive.loc["Alpha", "GeneA"], 1)
            self.assertEqual(positive.loc["Beta", "GeneC"], 1)
            self.assertEqual(negative.loc["Alpha", "GeneD"], 1)
            self.assertEqual(negative.loc["Beta", "GeneB"], 1)
            self.assertFalse(((positive == 1) & (negative == 1)).any().any())

    def test_bidcell_wrapper_generates_reference_from_h5ad(self):
        def fake_run(command, *, cwd=None, env=None):
            config = json.loads(Path(command[-1]).read_text())
            for key in ("fp_ref", "fp_pos_markers", "fp_neg_markers"):
                self.assertTrue(Path(config["files"][key]).is_file())
            result_dir = (
                Path(config["files"]["data_dir"])
                / "model_outputs"
                / "run"
                / "test_output"
            )
            result_dir.mkdir(parents=True)
            mask = np.zeros((8, 9), dtype=np.uint32)
            mask[2:6, 3:7] = 1
            imwrite(result_dir / "epoch_1_step_1_connected.tif", mask)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            wrappers, "_run_command", fake_run
        ):
            root = Path(directory)
            adata_path = root / "reference.h5ad"
            image_path = root / "dapi.tif"
            self._write_reference_anndata(adata_path)
            imwrite(image_path, np.ones((8, 9), dtype=np.uint16))
            transcripts = pd.DataFrame(
                {
                    "target": ["GeneA", "GeneB", "GeneC", "GeneD"],
                    "xc": [2, 3, 4, 5],
                    "yc": [2, 3, 4, 5],
                }
            )

            labels, sparse = wrappers.bidcell_segmentation(
                transcripts,
                image_path,
                region="R1",
                reference_adata=adata_path,
                cell_type_col="cell_type",
                output_path=root / "bidcell.npz",
                work_dir=root / "work",
                total_steps=1,
                test_step=1,
            )

            self.assertEqual(labels.shape, (8, 9))
            self.assertEqual(labels.max(), 1)
            np.testing.assert_array_equal(labels, sparse.toarray())

    def test_segger_mask_vertices_uses_label_bounding_boxes(self):
        mask = np.zeros((10, 12), dtype=np.uint32)
        mask[2:6, 3:8] = 1
        mask[7:10, 9:12] = 3

        vertices = _mask_vertices(mask, scale=2.0)

        self.assertEqual(set(vertices["cell_id"]), {"1", "3"})
        first = vertices.loc[vertices["cell_id"] == "1"]
        self.assertGreaterEqual(first["vertex_x"].min(), 5.0)
        self.assertLessEqual(first["vertex_x"].max(), 15.0)
        self.assertGreaterEqual(first["vertex_y"].min(), 3.0)
        self.assertLessEqual(first["vertex_y"].max(), 11.0)

    def test_normalize_transcript_table_uses_iss_defaults(self):
        source = pd.DataFrame(
            {
                "target": ["GeneA", "GeneB", None],
                "xc": [2, "4.5", 8],
                "yc": [3, 5, 9],
                "quality_minimum": [0.9, 0.2, 1.0],
            }
        )
        result = wrappers.normalize_transcript_table(
            source,
            quality_col="quality_minimum",
            min_quality=0.5,
        )
        self.assertEqual(list(result.columns), ["gene", "x", "y", "quality"])
        self.assertEqual(
            result.to_dict("records"),
            [{"gene": "GeneA", "x": 2.0, "y": 3, "quality": 0.9}],
        )

    def test_polygons_to_label_mask_handles_scale_multipolygon_and_hole(self):
        features = [
            {
                "type": "Feature",
                "properties": {"cell": 10},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[2, 2], [12, 2], [12, 12], [2, 12], [2, 2]],
                        [[6, 6], [8, 6], [8, 8], [6, 8], [6, 6]],
                    ],
                },
            },
            {
                "type": "Feature",
                "properties": {"cell": 20},
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": [
                        [[[20, 2], [26, 2], [26, 8], [20, 8], [20, 2]]],
                        [[[20, 12], [24, 12], [24, 16], [20, 16], [20, 12]]],
                    ],
                },
            },
        ]
        mask = wrappers.polygons_to_label_mask(
            features,
            mask_shape=(12, 16),
            coordinate_scale=2.0,
        )
        self.assertEqual(mask.shape, (12, 16))
        self.assertEqual(set(np.unique(mask)), {0, 1, 2})
        self.assertEqual(mask[3, 3], 0)
        self.assertEqual(mask[2, 11], 2)

    def test_geojson_conversion_saves_scipy_sparse_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geojson = root / "cells.geojson"
            output = root / "mask.npz"
            geojson.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"cell": 0},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]
                                    ],
                                },
                            }
                        ],
                    }
                )
            )
            dense, sparse = wrappers.geojson_to_sparse_mask(
                geojson,
                output,
                mask_shape=(8, 8),
            )
            self.assertTrue(output.is_file())
            np.testing.assert_array_equal(load_npz(output).toarray(), dense)
            np.testing.assert_array_equal(sparse.toarray(), dense)
            self.assertEqual(dense.max(), 1)

    def test_geojson_conversion_detects_gzip_without_gz_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geojson = root / "cells.geojson"
            output = root / "mask.npz"
            document = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"cell": 7},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[1, 1], [5, 1], [5, 5], [1, 5], [1, 1]]
                            ],
                        },
                    }
                ],
            }
            with gzip.open(geojson, "wt", encoding="utf-8") as handle:
                json.dump(document, handle)

            dense, _ = wrappers.geojson_to_sparse_mask(
                geojson,
                output,
                mask_shape=(8, 8),
            )

            self.assertTrue(output.is_file())
            self.assertEqual(dense.max(), 1)

    def test_proseg_wrapper_builds_command_and_converts_output(self):
        commands = []

        def fake_run(command, *, cwd=None, env=None):
            commands.append(command)
            output_dir = Path(command[command.index("--output-path") + 1])
            polygon_name = command[command.index("--output-cell-polygons") + 1]
            (output_dir / polygon_name).write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"cell": 0},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]
                                    ],
                                },
                            }
                        ],
                    }
                )
            )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            wrappers, "_run_command", fake_run
        ):
            root = Path(directory)
            transcripts = pd.DataFrame(
                {"target": ["A", "B"], "xc": [3, 6], "yc": [3, 6]}
            )
            seed = np.zeros((12, 12), dtype=np.uint32)
            seed[2:9, 2:9] = 1
            output = root / "proseg.npz"
            dense, _ = wrappers.proseg_segmentation(
                transcripts,
                region="R1",
                initial_mask=seed,
                output_path=output,
                proseg_command="proseg",
                nthreads=2,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(dense.max(), 1)
            self.assertIn("--cellpose-masks", commands[0])
            self.assertIn("--nthreads", commands[0])

    def test_dispatch_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown method"):
            wrappers.run_transcript_segmentation("not-a-method", pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
