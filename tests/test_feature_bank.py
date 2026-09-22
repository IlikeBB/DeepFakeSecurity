import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from script.bank_data import prepare_plan, save_array, select_groups, write_json
from script.bank_search import PatchBank
from script.feature_bank import load_sample, metrics, visualize_features
from script.bank_export import build_bank, load_feature


class FeatureBankTests(unittest.TestCase):
    def test_real_bank_npy_preview_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            faces, output = root / "faces", root / "bank"
            output.mkdir()
            frames = []
            for i in range(2):
                image = faces / "part/video" / f"frame_{i:06d}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (320, 160)).save(image)
                stat = image.stat()
                frames.append(dict(image_path=str(image), frame_index=i, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
            plan = {"groups": {"bank": [dict(video_id="part/video", group_id="part/video", label=0,
                                             crop_settings={}, frames=frames)]}}
            args = SimpleNamespace(faces_dir=str(faces), pca_images=2, seed=42, batch_size=8, jpeg_quality=95)
            features = np.random.default_rng(3).normal(size=(2, 2, 3, 5)).astype(np.float16)
            with patch("script.bank_export.load_encoder", return_value=(object(), None)), \
                    patch("script.bank_export.encode", return_value=(None, features)) as encode:
                build_bank(plan, args, output)
                for i in range(2):
                    np.testing.assert_array_equal(np.load(output / f"part/video/frame_{i:06d}.npy"), features[i])
                self.assertEqual(len(list(output.rglob("*.jpg"))), 1)
                with Image.open(output / "part/video/preview.jpg") as preview:
                    self.assertEqual(preview.format, "JPEG")
                encode.reset_mock()
                build_bank(plan, args, output)
                encode.assert_not_called()
                (output / "part/video/preview.jpg").unlink()
                build_bank(plan, args, output)
                encode.assert_not_called()
                metadata = json.loads((output / "part/video/frame_000000.json").read_text())
                with self.assertRaises(ValueError):
                    load_feature(output, dict(metadata["image"], label=1))

    def test_exact_chunked_search_and_traceable_neighbors(self):
        rng = np.random.default_rng(3)
        reference = rng.normal(size=(13, 5)).astype(np.float32)
        queries = rng.normal(size=(3, 7, 5)).astype(np.float32)
        bank = PatchBank(reference, "cpu", query_chunk_size=4, bank_chunk_size=5)
        scores, distances, neighbors = bank.score(queries, top_fraction=.3)
        normalized = reference / np.linalg.norm(reference, axis=-1, keepdims=True)
        query = queries / np.linalg.norm(queries, axis=-1, keepdims=True)
        similarities = query @ normalized.T
        expected = 1 - similarities.max(-1)
        np.testing.assert_allclose(distances, expected, atol=1e-6)
        np.testing.assert_array_equal(neighbors, similarities.argmax(-1))
        np.testing.assert_allclose(scores, np.sort(expected)[:, -3:].mean(-1), atol=1e-6)

    def test_splits_are_reproducible_and_source_disjoint(self):
        candidates = []
        for i in range(20):
            for label in (0, 1):
                candidates.append(dict(video_id=f"{i}_{label}", group_id=str(i), label=label))
        config = dict(seed=42, bank_videos=5, calibration_videos=3, eval_real_videos=2, eval_fake_videos=2)
        groups = select_groups(candidates, config)
        self.assertEqual(groups, select_groups(list(reversed(candidates)), config))
        sets = [{v["group_id"] for v in values} for values in groups.values()]
        for i, first in enumerate(sets):
            for second in sets[i + 1:]:
                self.assertFalse(first & second)
        self.assertTrue(all(v["label"] == 0 for v in groups["bank"] + groups["calibration"]))
        with self.assertRaises(ValueError):
            select_groups(candidates, dict(config, bank_videos=100))

    def test_npy_roundtrip_and_stale_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = root / "features"
            files.mkdir()
            record = {"sample_id": 0, "video_id": "real", "frame_index": 4}
            cls = np.ones(5, dtype=np.float16)
            patches = np.ones((2, 3, 5), dtype=np.float16)
            save_array(files / "cls_00000000.npy", cls)
            save_array(files / "patches_00000000.npy", patches)
            write_json(files / "image_00000000.json", dict(image=record, cls_shape=list(cls.shape),
                       patch_shape=list(patches.shape), dtype="float16"))
            _, loaded = load_sample(root, record)
            self.assertIsInstance(loaded, np.memmap)
            np.testing.assert_array_equal(loaded, patches)
            with self.assertRaises(ValueError):
                load_sample(root, dict(record, video_id="other"))

    def test_jpg_layout_shared_projection_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            faces = root / "faces"
            source = faces / "part/video/frame_000003.jpg"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (224, 224)).save(source)
            stat = source.stat()
            video = dict(video_id="part/video", group_id="part/video", label=0, crop_settings={},
                         frames=[dict(image_path=str(source), frame_index=3, size=stat.st_size, mtime_ns=stat.st_mtime_ns)])
            plan = {"groups": {"bank": [video]}}
            args = SimpleNamespace(faces_dir=str(faces), pca_images=1, seed=42, batch_size=8,
                                   visualization_size=224, jpeg_quality=95)
            out = root / "bank"
            out.mkdir()
            features = np.random.default_rng(1).normal(size=(1, 14, 14, 5)).astype(np.float32)
            with patch('script.feature_bank.load_encoder', return_value=(None, None)), \
                    patch('script.feature_bank.encode', return_value=(np.zeros((1, 5)), features)) as encode:
                visualize_features(plan, args, out)
                projection = (out / 'visualization.json').read_bytes()
                target = out / 'part/video/frame_000003.jpg'
                with Image.open(target) as image:
                    self.assertEqual(image.size, (224, 224))
                    self.assertEqual(image.format, 'JPEG')
                self.assertEqual(encode.call_count, 2)
                encode.reset_mock()
                visualize_features(plan, args, out)
                encode.assert_not_called()
                self.assertEqual(projection, (out / 'visualization.json').read_bytes())
            self.assertFalse(list(out.rglob('*.npy')))

    def test_prepare_uses_crop_records_and_audits_missing_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            faces, source = root / "faces", root / "source"
            labels = ["filename,label"]
            original = {}
            for i in range(7):
                name = f"real{i}"
                labels.append(f"real/{name},0")
                original[name + ".mp4"] = {"label": "REAL"}
                directory = faces / "part" / name
                directory.mkdir(parents=True)
                (directory / "frame.jpg").write_bytes(b"test crop")
                write_json(directory / "metadata.json", dict(label=0, source=str(source/name), settings={},
                           frames=[dict(status="ok", file="frame.jpg", frame_index=4)] if i < 6 else []))
            name = "fake0"
            labels.extend(["fake/fake0,1", "real/missing,0"])
            original[name + ".mp4"] = {"label": "FAKE", "original": "real0.mp4"}
            directory = faces / "part" / name
            directory.mkdir(parents=True)
            (directory / "frame.jpg").write_bytes(b"fake crop")
            write_json(directory / "metadata.json", dict(label=1, source=str(source/name), settings={},
                       frames=[dict(status="ok", file="frame.jpg", frame_index=8)]))
            write_json(source / "part/metadata.json", original)
            csv = root / "test.csv"
            csv.write_text("\n".join(labels))
            config = dict(faces_dir=str(faces), source_root=str(source), label_csv=str(csv), seed=42,
                          max_frames=32, bank_videos=2, calibration_videos=1, eval_real_videos=1, eval_fake_videos=1)
            plan = prepare_plan(config)
            self.assertEqual(plan["audit"]["missing_manifests"], 1)
            self.assertEqual(plan["audit"]["zero_face_videos"], 1)
            self.assertEqual(len(plan["groups"]["bank"]), 2)
            self.assertIn("mtime_ns", plan["groups"]["bank"][0]["frames"][0])
            # Building a real bank must not depend on fake metadata or evaluation sizes.
            (faces / "part/fake0/metadata.json").write_text("invalid fake metadata")
            real_plan = prepare_plan(dict(config, bank_videos=20, eval_fake_videos=100), bank_only=True)
            self.assertEqual(set(real_plan["groups"]), {"bank"})
            self.assertEqual(len(real_plan["groups"]["bank"]), 6)
            self.assertTrue(all(v["label"] == 0 for v in real_plan["groups"]["bank"]))

    def test_invalid_vectors_and_metric_direction(self):
        with self.assertRaises(ValueError):
            PatchBank(np.zeros((2, 3)), "cpu")
        rows = [dict(label=0, score=.1), dict(label=0, score=.2),
                dict(label=1, score=.7), dict(label=1, score=.8)]
        report = metrics(rows, .5)
        self.assertEqual(report["auroc"], 1.)
        self.assertEqual(report["false_positive_rate"], 0.)
        self.assertEqual(report["true_positive_rate"], 1.)


if __name__ == "__main__":
    unittest.main()
