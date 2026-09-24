import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.real_patch_bank import RealPatchBank, spatial_region_ids


def make_model():
    regions, dimensions = 4, 2
    prototypes = np.zeros((regions, 1, dimensions), dtype=np.float32)
    prototypes[..., 0] = 1
    gaussian_means = prototypes[:, 0].copy()
    return RealPatchBank(
        prototypes=prototypes,
        prototype_counts=np.ones(regions, dtype=np.int64),
        pca_mean=np.zeros(dimensions, dtype=np.float32),
        pca_components=np.eye(dimensions, dtype=np.float32),
        gaussian_means=gaussian_means,
        gaussian_precisions=np.repeat(np.eye(dimensions, dtype=np.float32)[None], regions, axis=0),
        gaussian_valid=np.ones(regions, dtype=bool),
        relation_means=np.ones((regions, regions), dtype=np.float32),
        relation_stds=np.full((regions, regions), .1, dtype=np.float32),
        relation_valid=np.ones((regions, regions), dtype=bool),
        score_centers=np.zeros(3, dtype=np.float32),
        score_scales=np.ones(3, dtype=np.float32),
        score_weights=np.array([.5, .3, .2], dtype=np.float32),
        grid_size=(2, 2), spatial_grid=(2, 2), top_fraction=.5)


class RealPatchBankTests(unittest.TestCase):
    def test_global_prototypes_remove_position_constraint_and_survive_reload(self):
        model = make_model().eval()
        model.prototypes[3, 0] = torch.tensor([0., 1.])
        features = torch.zeros((1, 2, 2, 2))
        features[..., 1] = 1
        foreground = torch.ones((1, 2, 2), dtype=torch.bool)
        spatial = model(features, foreground)["components"][0, 0]
        model.spatial_restriction = False
        model.clip_scores = False
        model.score_weights.copy_(torch.tensor([1., 0., 0.]))
        model.score_centers.fill_(.5)
        output = model(features, foreground)
        self.assertGreater(float(spatial), float(output["components"][0, 0]))
        self.assertAlmostEqual(float(output["score"][0]), -.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            model.save(path)
            metadata = dict(grid_size=[2, 2], spatial_grid=[2, 2], top_fraction=.5,
                            spatial_restriction=False, clip_scores=False)
            restored = RealPatchBank.load(path, metadata)
            self.assertTrue(torch.allclose(output["score"], restored(features, foreground)["score"]))

    def test_spatial_regions_cover_grid(self):
        ids = spatial_region_ids((4, 4), (2, 2)).reshape(4, 4)
        self.assertTrue(torch.equal(ids[:2, :2], torch.zeros((2, 2), dtype=torch.long)))
        self.assertEqual(set(ids.flatten().tolist()), {0, 1, 2, 3})

    def test_real_prototype_scores_below_unseen_patch(self):
        model = make_model().eval()
        foreground = torch.ones((2, 2, 2), dtype=torch.bool)
        features = torch.zeros((2, 2, 2, 2))
        features[0, ..., 0] = 1
        features[1, ..., 1] = 1
        output = model(features, foreground)
        self.assertEqual(tuple(output["anomaly_map"].shape), (2, 2, 2))
        self.assertLess(float(output["score"][0]), float(output["score"][1]))
        self.assertAlmostEqual(float(output["components"][0, 0]), 0., places=6)

    def test_background_features_do_not_change_score(self):
        model = make_model().eval()
        foreground = torch.tensor([[[True, True], [True, False]]])
        features = torch.zeros((1, 2, 2, 2))
        features[..., 0] = 1
        changed = features.clone()
        changed[0, 1, 1] = torch.tensor([1000., -1000.])
        first = model(features, foreground)
        second = model(changed, foreground)
        self.assertTrue(torch.allclose(first["score"], second["score"], atol=1e-6))
        self.assertEqual(float(first["anomaly_map"][0, 1, 1]), -1.)

    def test_checkpoint_round_trip(self):
        model = make_model().eval()
        metadata = {"grid_size": [2, 2], "spatial_grid": [2, 2], "top_fraction": .5}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            model.save(path)
            restored = RealPatchBank.load(path, metadata).eval()
            features = torch.tensor([[[[1., 0.], [1., 0.]], [[1., 0.], [0., 1.]]]])
            foreground = torch.ones((1, 2, 2), dtype=torch.bool)
            self.assertTrue(torch.allclose(model(features, foreground)["score"],
                                           restored(features, foreground)["score"]))

    def test_single_visible_region_has_finite_relation_fallback(self):
        model = make_model().eval()
        features = torch.zeros((1, 2, 2, 2))
        features[..., 0] = 1
        foreground = torch.tensor([[[True, False], [False, False]]])
        output = model(features, foreground)
        self.assertTrue(torch.isfinite(output["score"]).all())
        self.assertEqual(float(output["components"][0, 2]), 0.)


if __name__ == "__main__":
    unittest.main()
