import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from script.bank_data import image_records, read_json, write_json
from script.feature_bank import load_sample
from script.probe_source import extract_roles, prepare_source


class ProbeSourceTests(unittest.TestCase):
    def test_parallel_extraction_alignment_resume_and_stage_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = {}
            for role, label in (("bank", 0), ("train_fake", 1), ("evaluation", 1)):
                frames = []
                for i in range(4):
                    path = root / f"{role}_{i}.jpg"
                    path.write_bytes(b"source")
                    stat = path.stat()
                    frames.append(dict(image_path=str(path), frame_index=i, feature_path=f"video/{i}.npy",
                                       size=stat.st_size, mtime_ns=stat.st_mtime_ns))
                groups[role] = [dict(video_id=role, group_id=role, label=label, crop_settings={}, frames=frames)]
            plan = dict(groups=groups)
            bank, cache = root / "normal", root / "cache"
            args = SimpleNamespace(batch_size=2, dtype="float16")

            def encode(paths, model, processor, local, image_pool=None):
                features = [np.full((2, 2, 3), int(Path(path).stem.rsplit("_", 1)[1]), dtype=np.float16)
                            for path in paths]
                return None, np.stack(features)

            with patch("script.probe_source.load_encoder", return_value=(None, None)), \
                    patch("script.probe_source.encode", side_effect=encode) as encoder:
                extract_roles(plan, ("bank", "train_fake"), bank, cache, args, ["cuda:4", "cuda:5"], 4)
                self.assertFalse((cache / "evaluation").exists())
                for role in ("bank", "train_fake"):
                    directory = bank if role == "bank" else cache / role
                    for row in image_records(groups[role], role):
                        _, values = load_sample(directory, row)
                        np.testing.assert_array_equal(values, np.full((2, 2, 3), row["frame_index"]))
                encoder.reset_mock()
                extract_roles(plan, ("bank", "train_fake"), bank, cache, args, ["cuda:4", "cuda:5"], 4)
                encoder.assert_not_called()
                (bank / "video/0.json").unlink()  # An interrupted NPY/metadata pair is regenerated.
                extract_roles(plan, ("bank",), bank, cache, args, ["cuda:4"], 2)
                encoder.assert_called_once()

    def test_prepare_source_preserves_split_and_defers_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = root / "normal/demo"
            args = SimpleNamespace(source_experiment="demo", results_dir="outputs", cache_dir="cache",
                                   bank_dir="normal", extract_batch_size=8, devices=[], workers=2, stage="stage1")
            plan = {"groups": {role: [] for role in ("bank", "train_fake", "calibration", "evaluation")}}
            with patch("script.probe_source.parse_args", return_value=SimpleNamespace(seed=42)), \
                    patch("script.probe_source.experiment_spec", return_value={"seed": 42}), \
                    patch("script.probe_source.prepare_flat_plan", return_value=plan) as split, \
                    patch("script.probe_source.extract_roles") as extract, \
                    patch("script.bank_export.build_bank") as preview:
                prepare_source(root, args, bank)
                self.assertEqual(extract.call_args.args[1], ("bank", "train_fake"))
                preview.assert_called_once()
                args.stage = "stage2"
                prepare_source(root, args, bank)
                self.assertEqual(extract.call_args.args[1], ("calibration", "evaluation"))
                split.assert_called_once()
                self.assertEqual(read_json(bank / "splits.json"), plan)
                write_json(bank / "bank_config.json", {"seed": 99, "source_kind": "probe_raw_v1"})
                with self.assertRaisesRegex(ValueError, "設定已改變"):
                    prepare_source(root, args, bank)


if __name__ == "__main__":
    unittest.main()
