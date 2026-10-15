import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ISS_postprocessing import segmentation_free, spatial_domains
from ISS_postprocessing._sainsc_runner import _add_physical_coordinates


class SainscWrapperTests(unittest.TestCase):
    @staticmethod
    def transcripts():
        return pd.DataFrame(
            {
                "target": ["GeneA", "GeneB", "GeneC"],
                "xc": [2.0, 4.0, 6.0],
                "yc": [3.0, 5.0, 7.0],
            }
        )

    def test_prepare_sainsc_calibrates_grid_and_signatures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signatures = pd.DataFrame(
                {"TypeA": [1.0, 0.2], "TypeB": [0.1, 1.0]},
                index=["GeneA", "GeneB"],
            )
            paths = segmentation_free.prepare_sainsc_analysis(
                self.transcripts(),
                region="R1",
                output_dir=root,
                pixel_size_um=0.25,
                grid_size_um=0.5,
                min_peak_distance_um=4.0,
                signatures=signatures,
            )

            config = json.loads(paths["config"].read_text())
            self.assertEqual(config["coordinates"]["source_unit"], "pixel")
            self.assertEqual(config["coordinates"]["source_resolution_nm"], 250.0)
            self.assertEqual(config["coordinates"]["grid_size_source"], 2.0)
            self.assertEqual(config["coordinates"]["grid_origin_index"], [1, 1])
            self.assertEqual(config["analysis"]["min_peak_distance_grid"], 8)
            self.assertTrue(paths["signatures"].is_file())
            self.assertEqual(
                pd.read_csv(paths["transcripts"])["gene"].tolist(),
                ["GeneA", "GeneB", "GeneC"],
            )

    def test_sainsc_runner_supports_external_environment_prefix(self):
        commands = []

        def fake_run(command, *, cwd=None, env=None):
            commands.append(command)
            config = json.loads(Path(command[-1]).read_text())
            Path(config["outputs"]["local_maxima"]).write_bytes(b"test")
            Path(config["outputs"]["metadata"]).write_text("{}")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            segmentation_free, "_run_command", fake_run
        ):
            paths = segmentation_free.run_sainsc_analysis(
                self.transcripts(),
                region="R1",
                output_dir=directory,
                pixel_size_um=0.25,
                sainsc_python=["conda", "run", "-n", "sainsc", "python"],
            )

            self.assertTrue(paths["local_maxima"].is_file())
            self.assertEqual(
                commands[0][:5],
                ["conda", "run", "-n", "sainsc", "python"],
            )

    def test_sainsc_coordinates_restore_shifted_grid_origin(self):
        import anndata as ad

        adata = ad.AnnData(
            X=np.ones((2, 1)),
            obsm={"spatial": np.array([[0, 1], [2, 3]], dtype=float)},
        )
        _add_physical_coordinates(
            adata,
            {
                "source_unit": "pixel",
                "pixel_size_um": 0.25,
                "grid_size_um": 0.5,
                "grid_size_source": 2.0,
                "grid_origin_index": [5, 10],
            },
        )
        np.testing.assert_allclose(adata.obsm["spatial"], [[10, 22], [14, 26]])
        np.testing.assert_allclose(adata.obsm["spatial_um"], [[2.5, 5.5], [3.5, 6.5]])


class SfumatoWrapperTests(unittest.TestCase):
    @staticmethod
    def transcripts():
        return pd.DataFrame(
            {
                "target": ["GeneA", "GeneB", "GeneC", "GeneA"],
                "xc": [2.0, 4.0, 6.0, 8.0],
                "yc": [3.0, 5.0, 7.0, 9.0],
            }
        )

    def test_prepare_sfumato_converts_pixels_to_micrometres(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = spatial_domains.prepare_sfumato_analysis(
                self.transcripts(),
                region="R1",
                output_dir=directory,
                pixel_size_um=0.25,
                k_list=[5, 10, 5],
                bin_width_um=8,
            )

            table = pd.read_csv(paths["transcripts"])
            np.testing.assert_allclose(table["x"], [0.5, 1.0, 1.5, 2.0])
            np.testing.assert_allclose(table["y"], [0.75, 1.25, 1.75, 2.25])
            config = json.loads(paths["config"].read_text())
            self.assertEqual(config["preprocess"]["k_list"], [5, 10])
            self.assertEqual(config["preprocess"]["bin_width"], 8)
            self.assertEqual(config["preprocess"]["comp"], 2)

    def test_sfumato_runner_standardizes_bin_outputs(self):
        commands = []

        def fake_run(command, *, cwd=None, env=None):
            commands.append(command)
            config_path = Path(command[command.index("--config") + 1])
            config = json.loads(config_path.read_text())
            outroot = Path(config["bgm"]["outroot"])
            result_dir = outroot / "K5" / "SVD2"
            result_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "x": [4.0, 12.0],
                    "y": [8.0, 8.0],
                    "bin_id": [1, 2],
                    "cluster": [0, 1],
                    "second_cluster": [1, 0],
                    "p1": [0.9, 0.7],
                    "p2": [0.1, 0.3],
                    "compl_p1": [0.1, 0.3],
                    "mix_t": [0.2, 0.4],
                    "cmap_pos": [0.0, 1.0],
                }
            ).to_csv(result_dir / "demo_binlevel_FULL.csv", index=False)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            spatial_domains, "_run_command", fake_run
        ):
            root = Path(directory)
            sfumato_root = root / "SFUMATO"
            sfumato_root.mkdir()
            (sfumato_root / "run_sfumato.py").write_text("# test runner\n")
            output_dir = root / "output"

            paths = spatial_domains.run_sfumato_analysis(
                self.transcripts(),
                region="R1",
                output_dir=output_dir,
                pixel_size_um=0.25,
                sfumato_root=sfumato_root,
                sfumato_python=["conda", "run", "-n", "sfumato", "python"],
                k_list=[5],
                bin_width_um=8,
            )

            self.assertTrue(paths["bins_K5"].is_file())
            self.assertTrue(paths["manifest"].is_file())
            self.assertEqual(
                commands[0][:5],
                ["conda", "run", "-n", "sfumato", "python"],
            )

            import anndata as ad

            result = ad.read_h5ad(paths["bins_K5"])
            np.testing.assert_allclose(result.obsm["spatial_um"], [[4, 8], [12, 8]])
            np.testing.assert_allclose(result.obsm["spatial"], [[16, 32], [48, 32]])
            self.assertEqual(result.uns["sfumato"]["semantic_unit"], "spatial_domain")

    def test_sfumato_rejects_fractional_native_bin_width(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "whole number"):
                spatial_domains.prepare_sfumato_analysis(
                    self.transcripts(),
                    region="R1",
                    output_dir=directory,
                    pixel_size_um=0.25,
                    bin_width_um=7.5,
                )


if __name__ == "__main__":
    unittest.main()
