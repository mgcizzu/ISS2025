import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from ISS_postprocessing import segmentation_wrappers as wrappers
from ISS_postprocessing._segger_bridge import _mask_vertices


class TranscriptWrapperTests(unittest.TestCase):
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
