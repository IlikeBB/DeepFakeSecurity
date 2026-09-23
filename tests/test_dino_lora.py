import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from script.bank_data import write_json
from script.dino_lora import (LoRALinear, _compress, _local_discrepancy, _split_real, attach_lora,
                              cross_compression_loss, load_lora, local_anomaly_loss, lora_parameters,
                              lora_state, set_lora_enabled, tuning_info)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TinyAttention()


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([TinyLayer(), TinyLayer()])


class DinoLoRATests(unittest.TestCase):
    def test_single_image_local_discrepancy_and_masked_margin(self):
        config = {"quality": [10, 10], "scale": .35, "minimum_fraction": .25,
                  "maximum_fraction": .25, "mask_threshold": .25, "margin": .1,
                  "weight": .5, "background_weight": 1.}
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "face.png"
            pixels = np.full((64, 64, 3), 160, dtype=np.uint8)
            pixels[16:48, 16:48] = [220, 80, 30]
            Image.fromarray(pixels).save(source)
            pseudo, mask = _local_discrepancy(source, 9, config)
            self.assertEqual(pseudo.size, (64, 64))
            self.assertGreater(np.asarray(mask).max(), 0)
            clean = torch.randn(1, 16, 8, requires_grad=True)
            altered = clean.detach().clone()
            altered[:, 5:11] += .5
            anomaly, background = local_anomaly_loss(clean, altered, [mask], config)
            self.assertTrue(torch.isfinite(anomaly + background))
            (anomaly + background).backward()
            self.assertTrue(torch.isfinite(clean.grad).all())

    def test_cross_compression_preserves_patch_identity_and_affinity(self):
        torch.manual_seed(4)
        clean = torch.randn(2, 9, 4, requires_grad=True)
        zero, consistency, affinity = cross_compression_loss(clean, clean, clean, .1)
        self.assertAlmostEqual(float(zero), 0., places=6)
        altered = clean.detach() + torch.randn_like(clean) * .2
        loss, consistency, affinity = cross_compression_loss(clean, altered, altered, .1)
        self.assertGreater(float(consistency), 0.)
        self.assertGreater(float(affinity), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(clean.grad).all())

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "face.png"
            pixels = torch.arange(32 * 24 * 3, dtype=torch.uint8).reshape(24, 32, 3).numpy()
            Image.fromarray(pixels).save(source)
            compressed = _compress(source, 3, [30, 30], .5)
            self.assertEqual(compressed.size, (32, 24))
            self.assertFalse(np.array_equal(pixels, np.asarray(compressed)))

    def test_adapter_starts_as_exact_identity_and_round_trips(self):
        torch.manual_seed(3)
        model = TinyModel().requires_grad_(False)
        values = torch.randn(2, 4)
        expected = model.layer[-1].attention.q_proj(values)
        attach_lora(model, rank=2, alpha=2.)
        self.assertTrue(all(isinstance(getattr(model.layer[-1].attention, name), LoRALinear)
                            for name in ("q_proj", "k_proj", "v_proj")))
        torch.testing.assert_close(model.layer[-1].attention.q_proj(values), expected)
        self.assertEqual(sum(p.numel() for p in lora_parameters(model)), 48)
        self.assertTrue(all(p.requires_grad for p in lora_parameters(model)))
        with torch.no_grad():
            model.layer[-1].attention.q_proj.lora_b.fill_(.2)
        adapted = model.layer[-1].attention.q_proj(values)
        self.assertFalse(torch.equal(adapted, expected))
        set_lora_enabled(model, False)
        torch.testing.assert_close(model.layer[-1].attention.q_proj(values), expected)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "lora.safetensors"
            save_file(lora_state(model), str(checkpoint))
            torch.manual_seed(3)
            restored = TinyModel().requires_grad_(False)
            load_lora(restored, checkpoint, {"rank": 2, "alpha": 2.}, "cpu")
            torch.testing.assert_close(restored.layer[-1].attention.q_proj(values), adapted)

    def test_real_validation_is_family_disjoint(self):
        rows = [dict(group_id=f"family{i // 2}") for i in range(10)]
        train, validation = _split_real(rows, .2, 42)
        self.assertFalse({r["group_id"] for r in train} & {r["group_id"] for r in validation})
        self.assertEqual(len({r["group_id"] for r in validation}), 1)

    def test_metadata_detects_checkpoint_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank, output = root / "bank", root / "output"
            write_json(bank / "splits.json", {"groups": {}})
            checkpoint = output / "stage1/dino_lora.safetensors"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"weights")
            from script.bank_data import sha256
            config = {"enabled": True}
            write_json(output / "stage1/dino_lora.json", {
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "plan_sha256": sha256(bank / "splits.json"), "config": config})
            args = SimpleNamespace(encoder_tuning=config)
            tuning_info(output, args, bank)
            self.assertEqual(args.encoder_checkpoint, str(checkpoint))
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "權重已改變"):
                tuning_info(output, args, bank)


if __name__ == "__main__":
    unittest.main()
