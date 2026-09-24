import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.patch_mil import PatchRelationMIL
from script.patch_mil import _run_epoch


class PatchRelationMILTests(unittest.TestCase):
    def test_returns_image_and_patch_scores_with_foreground_mask(self):
        torch.manual_seed(3)
        model = PatchRelationMIL(input_dim=8, hidden_dim=16, heads=4, layers=1, dropout=.1,
                                 top_fraction=.5, grid_size=(2, 2)).eval()
        patches = torch.randn(2, 2, 2, 8)
        foreground = torch.tensor([[[True, True], [True, False]], [[True, False], [True, True]]])
        output = model(patches, foreground)

        self.assertEqual(tuple(output["image_logit"].shape), (2,))
        self.assertEqual(tuple(output["patch_logits"].shape), (2, 2, 2))
        self.assertEqual(tuple(output["attention"].shape), (2, 2, 2))
        self.assertTrue(torch.isfinite(output["image_logit"]).all())
        self.assertTrue(torch.isneginf(output["patch_logits"][~foreground]).all())
        self.assertTrue(torch.allclose(output["attention"][foreground].reshape(2, -1).sum(1), torch.ones(2)))

    def test_masked_background_tokens_do_not_change_prediction(self):
        torch.manual_seed(4)
        model = PatchRelationMIL(input_dim=8, hidden_dim=16, heads=4, layers=1, dropout=.1,
                                 top_fraction=.5, grid_size=(2, 2)).eval()
        foreground = torch.tensor([[[True, True], [True, False]]])
        patches = torch.randn(1, 2, 2, 8)
        changed = patches.clone()
        changed[:, 1, 1] = 1000
        self.assertTrue(torch.allclose(model(patches, foreground)["image_logit"],
                                       model(changed, foreground)["image_logit"], atol=1e-6))

    def test_training_epoch_accepts_patch_batches(self):
        torch.manual_seed(5)
        model = PatchRelationMIL(input_dim=8, hidden_dim=16, heads=4, layers=1, dropout=.1,
                                 top_fraction=.5, grid_size=(2, 2))
        features = torch.randn(4, 2, 2, 8)
        foreground = torch.ones(4, 2, 2, dtype=torch.bool)
        labels = torch.tensor([0., 1., 0., 1.])
        indices = torch.arange(4)
        loader = DataLoader(TensorDataset(features, foreground, labels, indices), batch_size=2)
        result = _run_epoch(model, loader, torch.optim.AdamW(model.parameters(), lr=1e-3),
                            torch.device("cpu"), patch_weight=.05, train=True)
        self.assertTrue(0 <= result["auroc"] <= 1)
        self.assertTrue(0 <= result["average_precision"] <= 1)


if __name__ == "__main__":
    unittest.main()
