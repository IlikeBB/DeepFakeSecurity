import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from script.bank_attention import CrossAttention, ReferenceBank, attention_info, family_split, load_attention, train_attention
from script.bank_data import read_json, write_json
from script.bank_search import PatchBank


class AttentionTests(unittest.TestCase):
    def config(self):
        return dict(neighbors=3, temperature=.07, hidden=8, heads=2, epochs=8, patience=2,
                    batch_size=64, patches_per_image=4, validation_fraction=.25,
                    learning_rate=.001, weight_decay=0., noise_std=.02)

    def test_chunked_topk_and_family_exclusion_match_dense_search(self):
        rng = np.random.default_rng(1)
        bank, query = rng.normal(size=(19, 8)).astype("float32"), rng.normal(size=(5, 8)).astype("float32")
        families, excluded = np.arange(19) % 4, np.arange(5) % 4
        search = ReferenceBank(bank, "cpu", 2, 2, self.config(), families=families)
        scores, ids = search.nearest(query, excluded)
        dense = F.normalize(torch.from_numpy(query), dim=-1) @ F.normalize(torch.from_numpy(bank), dim=-1).T
        dense[torch.from_numpy(excluded[:, None] == families)] = -torch.inf
        expected, expected_ids = dense.topk(3, dim=1)
        torch.testing.assert_close(scores, expected)
        torch.testing.assert_close(ids, expected_ids)
        self.assertFalse(np.any(families[ids.numpy()] == excluded[:, None]))
        actual = search.score(query[None], .4)
        baseline = PatchBank(bank, "cpu", 2, 2).score(query[None], .4)
        self.assertAlmostEqual(search.details["scores"]["nearest"], float(baseline[0][0]), places=6)
        np.testing.assert_array_equal(actual[2], baseline[2])
        with self.assertRaisesRegex(ValueError, "Not enough"):
            ReferenceBank(bank, "cpu", 2, 3, self.config(), families=np.zeros(19)).nearest(query, np.zeros(5))

    def test_existing_topk_caps_each_source_family(self):
        bank = np.eye(8, dtype=np.float32)
        query = np.array([[1., 1., 1., 1., .8, .7, .6, .5]], dtype=np.float32)
        families = np.array([0, 0, 0, 0, 1, 2, 3, 4])
        search = ReferenceBank(bank, "cpu", 2, 3, self.config(), families=families)
        distance, keep = search.score_candidates(query, np.arange(8)[None], 1)
        np.testing.assert_array_equal(keep, [[True, False, False, False, True, True, True, True]])
        self.assertTrue(np.isfinite(distance).all())

    def test_attention_has_gradients_and_can_only_mix_reference_values(self):
        torch.manual_seed(7)
        model = CrossAttention(8, 8, 2)
        query, candidates = F.normalize(torch.randn(5, 8), dim=-1), F.normalize(torch.randn(5, 3, 8), dim=-1)
        result, weights = model(query, candidates)
        torch.testing.assert_close(weights.sum(-1), torch.ones(5))
        self.assertTrue((weights >= 0).all())
        torch.testing.assert_close(result, (weights[..., None] * candidates).sum(1))
        loss = (1 - F.cosine_similarity(result, query)).mean()
        loss.backward()
        for parameter in model.parameters():
            self.assertGreater(float(parameter.grad.norm()), 0.)
        # 當所有候選相同時，改變 query 無法把輸出變成 query 的副本。
        same = candidates[:, :1].expand(-1, 3, -1)
        torch.testing.assert_close(model(-query, same)[0], same[:, 0])

    def test_real_only_training_validation_and_saved_model(self):
        config = self.config()
        sources = [dict(role="bank", label=0, group_id=f"family{i}") for i in range(6)]
        train, validation = family_split(sources, .25, 42)
        self.assertFalse(set(train) & set(validation))
        with self.assertRaisesRegex(ValueError, "bank real"):
            family_split([dict(sources[0], label=1)] + sources[1:], .25, 42)
        rng = np.random.default_rng(3)
        arrays = dict(features=rng.normal(size=(24, 8)).astype("float32"), origins=np.repeat(np.arange(6), 4))
        args = SimpleNamespace(attention=config, seed=42, devices=[], gpu_ids=[], query_chunk_size=4, bank_chunk_size=5)
        with tempfile.TemporaryDirectory() as tmp:
            bank, output = Path(tmp) / "bank", Path(tmp) / "output"
            write_json(output / "stage1/retrieval.json", {"fixture": True})
            # 零步長讓 validation 持平，驗證 patience 與最佳 checkpoint 選擇。
            with patch("torch.optim.AdamW.step"):
                train_attention(sources, arrays, args, bank, output)
            info = attention_info(output)
            self.assertEqual(info["best_epoch"], 1)
            self.assertEqual(info["epochs_completed"], 3)
            self.assertTrue(info["stopped_early"])
            self.assertEqual(info["reference_patches"], 20)
            model = load_attention(info, "cpu")
            search = ReferenceBank(arrays["features"], "cpu", 4, 5, config, model)
            score, _, _ = search.score(arrays["features"][:4][None], .5)
            self.assertTrue(np.isfinite(score).all())
            self.assertEqual(set(search.details["scores"]), {"nearest", "topk", "cross_attention"})
            before = Path(info["checkpoint"]).read_bytes()
            train_attention(sources, arrays, args, bank, output)
            self.assertEqual(before, Path(info["checkpoint"]).read_bytes())
            Path(info["checkpoint"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                attention_info(output)


if __name__ == "__main__":
    unittest.main()
