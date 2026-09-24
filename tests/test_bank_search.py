import unittest

import numpy as np
import torch
from unittest.mock import patch
from types import SimpleNamespace

from script.bank_search import PatchBank
from script.bank_retrieval import _bank_patch_subset, match_image


class PatchBankTests(unittest.TestCase):
    def test_augmented_patch_subset_is_bounded_and_deterministic(self):
        values = np.arange(40).reshape(10, 4)
        ids = np.arange(10)
        row = {'augmentation': {'bank_patch_fraction': .25,
                                'parameters': {'patch_seed': 17}}}
        first_values, first_ids = _bank_patch_subset(values, ids, row)
        second_values, second_ids = _bank_patch_subset(values, ids, row)
        self.assertEqual(len(first_ids), 3)
        np.testing.assert_array_equal(first_ids, second_ids)
        np.testing.assert_array_equal(first_values, values[first_ids])
        original_values, original_ids = _bank_patch_subset(values, ids, {})
        self.assertIs(original_values, values)
        self.assertIs(original_ids, ids)

    def test_weighted_score_matches_explanations_and_preserves_raw_distances(self):
        distances = np.array([[.9, .1, .1, .1, .8, .1, .1, .1, .1]], dtype=np.float32)
        search = SimpleNamespace(score=lambda *args: (np.array([.9]), distances, np.zeros((1, 9), dtype=int)))
        sources = [dict(image_path="reference", video_id="real", group_id="family", grid=[3, 3])]
        arrays = dict(origins=np.array([0]), patch_ids=np.array([0]))
        info = dict(foreground_minimum=.5, top_fraction=.1, boundary_weight=.5)
        with patch("script.bank_retrieval.foreground_patches",
                   return_value=(np.ones((9, 2)), np.arange(9), (3, 3))):
            result, distance_map, _ = match_image(None, "unused", search, info, sources, arrays, 1)
            self.assertAlmostEqual(result["score"], .8, places=6)
            self.assertEqual(result["matches"][0]["query_patch"], [1, 1])
            self.assertAlmostEqual(result["matches"][0]["weighted_evidence"], .8, places=6)
            np.testing.assert_array_equal(distance_map.flatten(), distances[0])
            info.pop("boundary_weight")
            legacy, _, _ = match_image(None, "unused", search, info, sources, arrays, 1)
            self.assertAlmostEqual(legacy["score"], .9, places=6)
            self.assertEqual(legacy["matches"][0]["query_patch"], [0, 0])

    def test_chunked_search_matches_cosine_nearest_neighbor(self):
        bank = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float16)
        patches = np.array([[[0.9, 0.1, 0], [0, 0.2, 0.8]]], dtype=np.float32)
        search = PatchBank(bank, "cpu", query_chunk_size=1, bank_chunk_size=2)

        scores, distances, neighbors = search.score(patches, top_fraction=0.5)
        normalized_bank = bank.astype(np.float32)
        normalized_bank /= np.linalg.norm(normalized_bank, axis=1, keepdims=True)
        normalized_query = patches[0] / np.linalg.norm(patches[0], axis=1, keepdims=True)
        expected_neighbors = (normalized_query @ normalized_bank.T).argmax(axis=1)
        expected_distances = 1 - (normalized_query @ normalized_bank.T).max(axis=1)

        self.assertEqual(search.bank.dtype, torch.float32)
        np.testing.assert_array_equal(neighbors[0], expected_neighbors)
        np.testing.assert_allclose(distances[0], expected_distances, atol=1e-6)
        self.assertAlmostEqual(float(scores[0]), float(expected_distances.max()), places=6)

    def test_rejects_zero_norm_bank_patch(self):
        with self.assertRaisesRegex(ValueError, "zero-norm"):
            PatchBank(np.zeros((1, 3), dtype=np.float16), "cpu", bank_chunk_size=1)


if __name__ == "__main__":
    unittest.main()
