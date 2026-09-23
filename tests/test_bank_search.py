import unittest

import numpy as np
import torch

from script.bank_search import PatchBank


class PatchBankTests(unittest.TestCase):
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
