import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from Stage2.bank_search import TopKPatchBank
from Stage2.evaluation import load_saved_match, match_batch, match_image


class Stage2EvaluationTests(unittest.TestCase):
    def test_batched_search_matches_individual_search_and_resumes(self):
        bank = np.array([[1, 0], [.9, .1], [0, 1], [.1, .9]], dtype=np.float32)
        sources = [{"image_path": "bank.jpg", "video_id": "video", "group_id": "family",
                    "grid": [2, 2]}]
        arrays = {"origins": np.zeros(4, dtype=np.int32), "patch_ids": np.arange(4)}
        info = {"foreground_minimum": .5, "top_fraction": .5}
        features = [np.array([[[1., 0.], [.8, .2]], [[0., 1.], [.2, .8]]], dtype=np.float32),
                    np.array([[[.7, .3], [.6, .4]], [[.3, .7], [.4, .6]]], dtype=np.float32)]
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "face.jpg"
            Image.new("RGB", (8, 8), "white").save(image)
            rows = [{"image_path": str(image), "sample_id": i} for i in range(2)]
            individual_search = TopKPatchBank(
                bank, "cpu", 3, 2, {"neighbors": 2, "temperature": .07})
            individual = [match_image(feature, str(image), individual_search, info,
                                      sources, arrays, 1) for feature in features]
            batched_search = TopKPatchBank(bank, "cpu", 3, 2, {"neighbors": 2, "temperature": .07})
            batched = match_batch(list(zip(rows, features)), batched_search, info, sources, arrays, 1)

            for expected, (_, prediction, distances, neighbors, details) in zip(individual, batched):
                self.assertAlmostEqual(expected[0]["score"], prediction["score"], places=6)
                self.assertEqual(expected[0]["comparison_scores"], prediction["comparison_scores"])
                np.testing.assert_allclose(expected[1], distances, atol=1e-6)
                np.testing.assert_array_equal(expected[2], neighbors)

                saved = Path(temporary) / f"{len(list(Path(temporary).glob('*.npz')))}.npz"
                np.savez(saved, distances=distances, neighbors=neighbors,
                         **{key: value for key, value in details.items() if key != "scores"})
                resumed = load_saved_match(saved, rows[len(list(Path(temporary).glob('*.npz'))) - 1],
                                           features[len(list(Path(temporary).glob('*.npz'))) - 1], info,
                                           sources, arrays, 1, "topk")
                self.assertAlmostEqual(prediction["score"], resumed["score"], places=6)
                self.assertEqual(prediction["comparison_scores"], resumed["comparison_scores"])


if __name__ == "__main__":
    unittest.main()
