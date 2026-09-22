import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from script.bank_probe import descriptor, fit_indices, fit_model, read_descriptors, score, validation_families
from script.probe_compute import map_devices


class BankProbeTests(unittest.TestCase):
    def test_threaded_reader_keeps_record_feature_alignment(self):
        rows = [dict(image_path=str(i)) for i in range(30)]
        with patch("script.bank_probe.image_records", return_value=rows), \
                patch("script.bank_probe.load_sample", side_effect=lambda directory, row: (None, row)), \
                patch("script.bank_probe.descriptor", side_effect=lambda row, path: np.array([int(path)])):
            actual, x = read_descriptors({"groups": {"bank": []}}, ["bank"], Path("unused"), Path("unused"), workers=4)
        self.assertEqual(actual, rows)
        np.testing.assert_array_equal(x[:, 0], np.arange(30))

    @unittest.skipUnless(torch.cuda.device_count() > 2, "GPU 1 and 2 required")
    def test_multi_gpu_matches_weighted_cpu_objective_and_scores(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(200, 24))
        labels = (x[:, 0] + .4 * x[:, 1] > .3).astype(int)
        rows = [dict(label=int(label), video_id=str(i // 2)) for i, label in enumerate(labels)]
        indices = np.arange(len(rows))
        cpu = fit_model(x, rows, indices, 24, .01, .1)

        def fit(c, device):
            state = fit_model(x, rows, indices, 24, c, .1, device)
            np.testing.assert_allclose(score(x, cpu), score(x, state), atol=1e-6)
            return device

        devices = ["cuda:1", "cuda:2"]
        self.assertEqual(map_devices(fit, [.01, .01, .01], devices), ["cuda:1", "cuda:2", "cuda:1"])
        np.testing.assert_allclose(score(x, cpu), score(x, cpu, devices), atol=1e-10)
        statistical = {key: cpu[key] for key in ("center", "scale")}
        np.testing.assert_allclose(score(x, statistical), score(x, statistical, devices), atol=1e-10)

    def test_foreground_pooling_ignores_black_padding_and_preserves_spatial_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "face.png"
            pixels = np.full((2, 2, 3), 255, dtype=np.uint8)
            pixels[0, 0] = 0
            Image.fromarray(pixels).save(path)
            patches = np.array([[[100, 0], [0, 1]], [[0, 1], [0, 1]]], dtype=np.float32)
            features = descriptor(patches, path)
            np.testing.assert_allclose(features[:4], [0, 1, 0, 0])
            np.testing.assert_allclose(features[4:], [0, 0, 0, 1, 0, 1, 0, 1])
            Image.fromarray(np.zeros_like(pixels)).save(path)
            with self.assertRaisesRegex(ValueError, "foreground"):
                descriptor(patches, path)

    def test_family_holdout_and_fake_budget(self):
        rows = [dict(group_id=f"r{i}", video_id=f"r{i}", label=0) for i in range(20)]
        rows += [dict(group_id=f"r{i}", video_id=f"f{i}", label=1) for i in range(10)]
        rows += [dict(group_id=f"g{i}", video_id=f"g{i}_{j}", label=1)
                 for i in range(20) for j in range(3)]
        held = validation_families(rows, 73, .2)
        eligible = [i for i, r in enumerate(rows) if r["group_id"] not in held]
        chosen = fit_indices(rows, eligible, 4, 73)
        self.assertFalse(held & {rows[i]["group_id"] for i in chosen})
        self.assertEqual(len({rows[i]["video_id"] for i in chosen if rows[i]["label"]}), 4)
        self.assertTrue(all(rows[i]["label"] == 0 for i in fit_indices(rows, eligible, 0, 73)))
        self.assertEqual(held, validation_families(list(reversed(rows)), 73, .2))

    def test_real_only_normalization_and_serialized_score(self):
        rng = np.random.default_rng(4)
        x = np.concatenate([rng.normal(0, .1, (20, 3)), rng.normal(3, .1, (10, 3))])
        rows = [dict(video_id=str(i), label=int(i >= 20)) for i in range(30)]
        state = fit_model(x, rows, np.arange(30), 3, .1, .1)
        np.testing.assert_allclose(state["center"], x[:20].mean(0))
        values = score(x, state)
        self.assertGreater(values[20:].min(), values[:20].max())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            np.savez(path, **state)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_allclose(values, score(x, dict(data)))


if __name__ == "__main__":
    unittest.main()
