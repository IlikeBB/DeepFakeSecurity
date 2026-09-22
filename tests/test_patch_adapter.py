import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from script.bank_adapter import (adapt_features, fake_ranking_loss, load_adapter,
                                 matching_scores, train_adapter)
from script.bank_data import read_json, write_json
from script.bank_flat import prepare_flat_plan, select_full_groups


class PatchAdapterTests(unittest.TestCase):
    def test_stage_boundary_and_completion_guard(self):
        from script.feature_bank import main
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(output_dir=tmp + "/bank", results_dir=tmp + "/results", experiment="trial",
                                   device="cpu", stage="prepare", input_layout="flat", train_adapter=True)
            plan = {"groups": {"bank": [], "train_fake": [], "calibration": [], "evaluation": []}}
            with patch("script.feature_bank.parse_args", return_value=args), \
                    patch("script.feature_bank.experiment_spec", return_value={"fixture": True}), \
                    patch("script.bank_flat.prepare_flat_plan", return_value=plan), \
                    patch("script.feature_bank.extract_features") as extract, \
                    patch("script.bank_adapter.train_adapter") as train, \
                    patch("script.bank_adapter.load_adapter"), \
                    patch("script.feature_bank.load_bank"), \
                    patch("script.feature_bank.evaluate") as evaluate:
                main(profile="patch_bank")
                args.stage = "stage2"
                with self.assertRaisesRegex(ValueError, "Complete stage1"):
                    main(profile="patch_bank")
                extract.assert_not_called()
                args.stage = "stage1"
                main(profile="patch_bank")
                self.assertEqual(extract.call_args.kwargs["roles"], {"bank", "train_fake"})
                train.assert_called_once()
                evaluate.assert_not_called()
                args.stage = "stage2"
                main(profile="patch_bank")
                self.assertEqual(extract.call_args.kwargs["roles"], {"calibration", "evaluation"})
                train.assert_called_once()
                evaluate.assert_called_once()

    def test_full_split_uses_every_video_and_keeps_families_separate(self):
        candidates = [dict(video_id=f"r{i}", group_id=f"g{i}", label=0) for i in range(20)]
        candidates += [dict(video_id=f"f{i}_{j}", group_id=f"g{i}", label=1)
                       for i in range(10) for j in range(3)]
        candidates += [dict(video_id=f"other{i}", group_id=f"other{i}", label=1) for i in range(10)]
        groups = select_full_groups(candidates, 42)
        self.assertEqual(groups, select_full_groups(list(reversed(candidates)), 42))
        selected = [v["video_id"] for values in groups.values() for v in values]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertCountEqual(selected, [v["video_id"] for v in candidates])
        training = {v["group_id"] for key in ("bank", "train_fake") for v in groups[key]}
        calibration = {v["group_id"] for v in groups["calibration"]}
        evaluation = {v["group_id"] for v in groups["evaluation"]}
        self.assertFalse(training & calibration or training & evaluation or calibration & evaluation)
        self.assertEqual(len(groups["bank"]), 14)
        self.assertEqual(len(groups["calibration"]), 3)
        self.assertTrue(all(v["label"] == 0 for v in groups["bank"] + groups["calibration"]))

    def test_flat_source_families_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata, rows = {}, ["filename,label"]
            for label, category in ((0, "normal"), (1, "anomaly")):
                (root / category).mkdir()
                for i in range(10):
                    video = f"{'real' if label == 0 else 'fake'}{i}"
                    rows.append(f"{video}.mp4,{label}")
                    metadata[video + ".mp4"] = dict(label="FAKE" if label else "REAL")
                    if label:
                        metadata[video + ".mp4"]["original"] = f"real{i}.mp4"
                    for frame in (0, 10, 20):
                        (root / category / f"dfdc_train_part_0-{video}-frame_{frame:06d}.jpg").write_bytes(b"fixture")
            write_json(root / "source/dfdc_train_part_0/metadata.json", metadata)
            (root / "labels.csv").write_text("\n".join(rows))
            config = dict(normal_dir=str(root / "normal"), anomaly_dir=str(root / "anomaly"),
                          source_root=str(root / "source"), label_csv=str(root / "labels.csv"), seed=42,
                          bank_videos=3, calibration_videos=2, eval_real_videos=2, eval_fake_videos=1,
                          train_fake_videos=1, max_frames=2)
            plan = prepare_flat_plan(config)
            sets = [{v["group_id"] for v in videos} for videos in plan["groups"].values()]
            for i, group in enumerate(sets):
                for other in sets[i + 1:]:
                    self.assertFalse(group & other)
            for role, videos in plan["groups"].items():
                for video in videos:
                    self.assertEqual([f["frame_index"] for f in video["frames"]], [0, 20])
                    self.assertTrue(video["frames"][0]["feature_path"].startswith(video["video_id"] + "/"))
                    if role in ("bank", "calibration"):
                        self.assertEqual(video["label"], 0)
            pure = prepare_flat_plan(dict(config, train_fake_videos=0))
            self.assertNotIn("train_fake", pure["groups"])
            (root / "anomaly/dfdc_train_part_0-real0-frame_000000.jpg").write_bytes(b"wrong label")
            with self.assertRaisesRegex(ValueError, "label mismatch"):
                prepare_flat_plan(config)

    def test_matching_excludes_same_video_and_fake_is_image_supervised(self):
        reference = torch.eye(3)
        query = reference[:2].reshape(1, 2, 3)
        score = matching_scores(query, reference, .5, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(score, torch.ones(1))
        with self.assertRaises(ValueError):
            matching_scores(query, reference, .5, torch.ones(1, 3, dtype=torch.bool))
        # Only the hardest patch contributes to the image-level fake ranking in this example.
        query = torch.tensor([[[.9, .1, 0.], [.1, .1, .1]]], requires_grad=True)
        scores = matching_scores(query, reference, .5)
        fake_ranking_loss(scores, torch.tensor([.8]), .2).backward()
        self.assertEqual(query.grad[0, 0].abs().sum().item(), 0.)
        self.assertGreater(query.grad[0, 1].abs().sum().item(), 0.)

    def test_training_schedule_checkpoint_and_real_only(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(3)
        features = rng.normal(size=(6, 2, 2, 6)).astype(np.float32)
        videos = [dict(video_id=f"video{i}", group_id=f"group{i}", label=0, crop_settings={},
                       frames=[dict(image_path=f"real{i}_0"), dict(image_path=f"real{i}_1")]) for i in range(2)]
        fake = [dict(video_id="fake", group_id="fakegroup", label=1, crop_settings={},
                     frames=[dict(image_path="fake0"), dict(image_path="fake1")])]
        plan = dict(groups=dict(bank=videos, train_fake=fake, calibration=[{"poison": True}], evaluation=[{"poison": True}]))
        args = SimpleNamespace(train_adapter=True, seed=42, device="cpu", adapter_hidden=4, learning_rate=.01,
                               reference_patches=8, train_epochs=3, train_batch_size=2, top_fraction=.5,
                               anchor_weight=1., fake_interval=2, fake_batch_size=1, fake_weight=.1, fake_margin=.5)
        accessed = []

        def sample(directory, record):
            accessed.append(record["role"])
            self.assertIn(record["role"], ("bank", "train_fake"))
            offset = 4 if record["role"] == "train_fake" else 0
            return None, features[offset + record["sample_id"]]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "bank_config.json", {"test": True})
            write_json(root / "splits.json", plan)
            with patch("script.feature_bank.load_sample", side_effect=sample):
                train_adapter(plan, args, root, root / "results")
            info = read_json(root / "results/training/adapter.json")
            self.assertEqual(info["steps"], 6)
            self.assertEqual(info["fake_updates"], 3)
            self.assertEqual(info["fake_images"], 2)
            adapter = load_adapter(args, root, root / "results")
            actual = adapt_features(features[:2], adapter)
            original = features[:2] / np.linalg.norm(features[:2], axis=-1, keepdims=True)
            self.assertGreater(np.abs(actual - original).max(), 1e-5)
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(np.linalg.norm(actual, axis=-1), 1., atol=1e-6)
            with patch("script.feature_bank.load_sample", side_effect=AssertionError("Must reuse completed training")):
                train_adapter(plan, args, root, root / "results")
            pure = dict(groups={key: value for key, value in plan["groups"].items() if key != "train_fake"})
            accessed.clear()
            with patch("script.feature_bank.load_sample", side_effect=sample):
                train_adapter(pure, args, root, root / "pure")
            self.assertEqual(set(accessed), {"bank"})
            self.assertEqual(read_json(root / "pure/training/adapter.json")["fake_updates"], 0)
            write_json(root / "splits.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_adapter(args, root, root / "results")


if __name__ == "__main__":
    unittest.main()
