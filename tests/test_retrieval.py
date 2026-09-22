import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from script.bank_data import image_records, read_json, save_array, sha256, write_json
from script.bank_retrieval import build, evaluate, foreground_patches, load_index
from script.retrieval import main, parse_args


class RetrievalTests(unittest.TestCase):
    def test_attention_evaluation_reports_three_methods_and_reference_evidence(self):
        from safetensors.torch import save_file
        from script.bank_attention import CrossAttention
        with tempfile.TemporaryDirectory() as tmp:
            plan, args, bank, cache, output = self.fixture(Path(tmp))
            args.method = "cross_attention"
            args.attention = dict(neighbors=2, temperature=.07, hidden=4, heads=2)
            build(plan, args, bank, output)
            checkpoint = bank / "attention.safetensors"
            save_file(CrossAttention(2, 4, 2).state_dict(), str(checkpoint))
            write_json(output / "stage1/attention.json", dict(
                checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                retrieval_sha256=sha256(output / "stage1/retrieval.json"), dimensions=2, config=args.attention))
            evaluate(plan, args, bank, cache, output)
            report = read_json(output / "stage2/comparison.json")
            self.assertEqual(set(report["methods"]), {"nearest", "topk", "cross_attention"})
            self.assertTrue(all(item["auroc"] == 1. for item in report["methods"].values()))
            fake = read_json(output / "stage2/evaluation_scores.json")[1]
            self.assertEqual(fake["matches"][0]["cosine_similarity"], 0.)
            with np.load(fake["patch_matches"]) as matches:
                self.assertEqual(matches["candidate_ids"].shape, (4, 2))
                np.testing.assert_allclose(matches["weights"].sum(-1), 1.)
                self.assertEqual(matches["query_patch_ids"].tolist(), [0, 1, 2, 3])

    def test_main_uses_raw_features_and_only_exports_previews_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, fixture_args, bank, cache, output = self.fixture(root)
            args = parse_args(["--exper", "FB", "--gpus", "--stage", "all"])
            vars(args).update(vars(fixture_args), results_dir=str(root / "outputs"), export_previews=False, method="nearest")
            model = root / "model"
            model.mkdir()
            for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
                (model / name).write_bytes(b"fixture fingerprint")
            labels = root / "labels.csv"
            labels.write_text("fixture labels")
            args.model_path, args.label_csv = str(model), str(labels)
            (bank / "bank_config.json").unlink()
            (bank / "splits.json").unlink()
            with patch("script.retrieval.parse_args", return_value=args), \
                 patch("script.retrieval.prepare_flat_plan", return_value=plan), \
                 patch("script.retrieval.extract_roles"), \
                 patch("script.bank_export.build_bank") as preview:
                main([])
                preview.assert_not_called()
                info, _, arrays = load_index(plan, bank, output)
                self.assertNotIn("alignment", info)
                np.testing.assert_array_equal(arrays["features"], np.tile([1, 0], (4, 1)))
                self.assertEqual(read_json(output / "stage2/metrics.json")["image"]["auroc"], 1.)
                # 完成索引後仍可補預覽，不改動檢索設定或重新建立索引。
                completion = (output / "stage1/retrieval.json").read_bytes()
                args.export_previews, args.stage = True, "stage1"
                main([])
                preview.assert_called_once_with(plan, args, bank)
                self.assertEqual((output / "stage1/retrieval.json").read_bytes(), completion)

    def test_rejects_old_aligned_index_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, args, bank, _, output = self.fixture(Path(tmp))
            build(plan, args, bank, output)
            path = output / "stage1/retrieval.json"
            write_json(path, dict(read_json(path), alignment={"loss_type": "asa"}))
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "已移除的文字對齊"):
                load_index(plan, bank, output)
            self.assertEqual(path.read_bytes(), original)

    def fixture(self, root):
        bank, cache, output = root / "normal/FB", root / "normal/FB/cache", root / "outputs/FB"
        bank.mkdir(parents=True)
        groups = {role: [] for role in ("bank", "train_fake", "calibration", "evaluation")}
        for name, role, label in (("reference", "bank", 0), ("cal", "calibration", 0),
                                  ("real", "evaluation", 0), ("fake", "evaluation", 1)):
            path = root / (name + ".png")
            Image.new("RGB", (32, 32), "white").save(path)
            stat = path.stat()
            groups[role].append(dict(video_id=name, group_id=name, label=label, crop_settings={}, frames=[
                dict(image_path=str(path), feature_path=name + "/frame.npy", frame_index=0,
                     size=stat.st_size, mtime_ns=stat.st_mtime_ns)]))
        plan = dict(protocol="synthetic fixture", groups=groups)
        write_json(bank / "splits.json", plan)
        write_json(bank / "bank_config.json", {"fixture": True})
        for role in groups:
            for row in image_records(groups[role], role):
                path = (bank if role == "bank" else cache / role) / row["feature_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                values = np.tile([0, 1] if row["label"] else [1, 0], (2, 2, 1)).astype(np.float16)
                save_array(path, values)
                write_json(path.with_suffix(".json"), dict(image=row, shape=list(values.shape), dtype="float16"))
        args = SimpleNamespace(experiment="FB", bank_dir=str(root / "normal"), workers=2, foreground_minimum=.5,
                               top_fraction=.5, devices=[], query_chunk_size=2, bank_chunk_size=3,
                               match_count=2, threshold_quantile=.99)
        return plan, args, bank, cache, output

    def test_cosine_decisions_traceability_and_stage_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, args, bank, cache, output = self.fixture(Path(tmp))
            build(plan, args, bank, output)
            evaluate(plan, args, bank, cache, output)
            results = read_json(output / "stage2/evaluation_scores.json")
            self.assertEqual([r["prediction"] for r in results], ["normal", "anomaly"])
            self.assertEqual([r["score"] for r in results], [0, 1])
            self.assertEqual(results[1]["matches"][0]["source_video"], "reference")
            self.assertEqual(results[1]["matches"][0]["cosine_similarity"], 0)
            self.assertEqual(read_json(output / "stage2/thresholds.json")["threshold"], 0)
            self.assertEqual(read_json(output / "stage1/retrieval.json")["patch_count"], 4)
            with np.load(results[1]["patch_matches"]) as matched:
                np.testing.assert_array_equal(matched["distances"], np.ones((2, 2)))
                np.testing.assert_array_equal(matched["neighbors"], np.zeros((2, 2)))

    def test_real_only_bank_leakage_and_integrity_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, args, bank, cache, output = self.fixture(Path(tmp))
            plan["groups"]["bank"][0]["label"] = 1
            with self.assertRaisesRegex(ValueError, "real training"):
                build(plan, args, bank, output)
            plan["groups"]["bank"][0]["label"] = 0
            build(plan, args, bank, output)
            plan["groups"]["evaluation"][0]["group_id"] = "reference"
            with self.assertRaisesRegex(ValueError, "own source family"):
                evaluate(plan, args, bank, cache, output)
            info = read_json(output / "stage1/retrieval.json")
            Path(info["files"]["origins"]["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "index changed"):
                load_index(plan, bank, output)

    def test_foreground_ignores_black_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "face.png"
            pixels = np.full((2, 2, 3), 255, dtype=np.uint8)
            pixels[0, 0] = 0
            Image.fromarray(pixels).save(image)
            _, ids, _ = foreground_patches(np.ones((2, 2, 3)), image, .5)
            np.testing.assert_array_equal(ids, [1, 2, 3])

    def test_stage2_requires_stage1_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(bank_dir=tmp + "/normal", results_dir=tmp + "/outputs", experiment="FB", stage="stage2")
            with patch("script.retrieval.parse_args", return_value=args):
                with self.assertRaisesRegex(ValueError, "stage1 --exper FB"):
                    main([])
            self.assertFalse(Path(args.results_dir).exists())


if __name__ == "__main__":
    unittest.main()
