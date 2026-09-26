import unittest
import copy
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import save_file
import torch
from torch import nn

from Stage1.encoder_tuning import (configure_partial_finetuning, ema_momentum, encoder_state, exclude_plan_images,
                                   load_encoder_tuning, update_ema)
from Stage1.fsfm import (FeatureDecoder, crfrp_mask, crop_semantic_map,
                         feature_reconstruction_loss, patchify_labels)


class FSFMTests(unittest.TestCase):
    def test_invalid_segface_pair_is_removed_from_plan(self):
        plan = {"audit": {}, "groups": {
            "bank": [{"frames": [{"image_path": "valid.jpg"}, {"image_path": "invalid.jpg"}]}],
            "calibration": [{"frames": [{"image_path": "calibration.jpg"}]}],
            "evaluation": [],
        }}

        removed = exclude_plan_images(plan, {"invalid.jpg"})

        self.assertEqual(removed, 1)
        self.assertEqual(plan["groups"]["bank"][0]["frames"], [{"image_path": "valid.jpg"}])
        self.assertEqual(plan["audit"]["segface_alignment_excluded"], 1)

    def test_partial_finetuning_and_ema_teacher(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])
                self.norm = nn.LayerNorm(4)

        student = Encoder()
        teacher = copy.deepcopy(student).requires_grad_(False)
        parameters = configure_partial_finetuning(student, 2)

        self.assertFalse(any(parameter.requires_grad for layer in student.layer[:4]
                             for parameter in layer.parameters()))
        self.assertTrue(all(parameter.requires_grad for layer in student.layer[4:]
                            for parameter in layer.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in student.norm.parameters()))
        self.assertEqual([id(parameter) for parameter in parameters],
                         [id(parameter) for parameter in student.parameters() if parameter.requires_grad])

        name, parameter = next((name, parameter) for name, parameter in student.named_parameters()
                               if parameter.requires_grad)
        before = dict(teacher.named_parameters())[name].clone()
        with torch.no_grad():
            parameter.add_(2.)
        update_ema(student, teacher, .75)
        torch.testing.assert_close(dict(teacher.named_parameters())[name], before + .5)
        self.assertAlmostEqual(ema_momentum({"ema_start": .996, "ema_end": 1.}, 0, 10), .996)
        self.assertAlmostEqual(ema_momentum({"ema_start": .996, "ema_end": 1.}, 9, 10), 1.)

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "partial.safetensors"
            save_file(encoder_state(student), str(checkpoint))
            restored = Encoder()
            load_encoder_tuning(restored, checkpoint, {"unfreeze_blocks": 2}, "cpu")
            for key, value in encoder_state(student).items():
                torch.testing.assert_close(encoder_state(restored)[key], value)

    def test_crfrp_is_deterministic_and_masks_requested_foreground_ratio(self):
        labels = np.full((224, 224), 2, dtype=np.uint8)
        labels[32:64, 32:96] = 6
        labels[64:96, 32:96] = 8
        labels[96:144, 80:144] = 10
        labels[144:192, 64:160] = 11
        semantic = patchify_labels(labels, 14)
        foreground = np.ones((14, 14), dtype=bool)

        first = crfrp_mask(semantic, foreground, .75, 42)
        second = crfrp_mask(semantic, foreground, .75, 42)

        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertEqual(first[2], second[2])
        self.assertEqual(int(first[0].sum()), round(196 * .75))
        self.assertTrue(np.all(~first[1] | first[0]))
        self.assertGreater(first[1].sum(), 0)

    def test_semantics_reproduce_segmented_crop_before_resize(self):
        labels = np.zeros((224, 224), dtype=np.uint8)
        labels[42:182, 37:177] = 2
        config = {"face_ids": [2], "background_ids": [0], "dilation_ratio": 0.,
                  "closing_ratio": 0., "min_face_area": .01}
        pixels = np.indices((224, 224)).sum(0).astype(np.uint8)
        source = Image.fromarray(np.repeat(pixels[:, :, None], 3, axis=2))
        segmented = source.crop((37, 42, 177, 182))

        aligned = crop_semantic_map(labels, config, source, segmented, 224)

        self.assertEqual(aligned.shape, (224, 224))
        self.assertTrue(np.all(aligned == 2))

    def test_semantics_uses_saved_image_to_recover_a_different_crop(self):
        labels = np.zeros((224, 224), dtype=np.uint8)
        labels[42:182, 37:177] = 2
        config = {"face_ids": [2], "background_ids": [0], "dilation_ratio": 0.,
                  "closing_ratio": 0., "min_face_area": .01}
        rng = np.random.default_rng(7)
        source = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
        segmented = source.crop((28, 16, 223, 220))

        aligned = crop_semantic_map(labels, config, source, segmented, 224)

        self.assertEqual(aligned.shape, (224, 224))
        unrelated = Image.fromarray(np.full((180, 180, 3), 255, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "does not match"):
            crop_semantic_map(labels, config, source, unrelated, 224)

        dark_pixels = rng.integers(0, 35, (224, 224, 3), dtype=np.uint8)
        dark_source = Image.fromarray(dark_pixels)
        dark_segmented = dark_source.crop((40, 23, 201, 209))
        dark_aligned = crop_semantic_map(labels, config, dark_source, dark_segmented, 224)
        self.assertEqual(dark_aligned.shape, (224, 224))

    def test_feature_decoder_sends_gradients_to_tokens_and_decoder(self):
        config = {"decoder_heads": 4, "decoder_mlp_ratio": 2., "decoder_dropout": 0.,
                  "decoder_layers": 1, "region_weight": .25, "global_weight": .1}
        decoder = FeatureDecoder(32, 16, config)
        online = torch.randn(2, 16, 32, requires_grad=True)
        target = torch.randn(2, 16, 32)
        mask = torch.zeros(2, 4, 4, dtype=torch.bool)
        mask[:, :3] = True
        region = torch.zeros_like(mask)
        region[:, 1] = True

        loss, components = feature_reconstruction_loss(decoder, online, target, mask, region, config)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(components), {"masked_reconstruction", "region_reconstruction", "local_global"})
        self.assertIsNotNone(online.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in decoder.parameters()))


if __name__ == "__main__":
    unittest.main()
