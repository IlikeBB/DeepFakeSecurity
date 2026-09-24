import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from script.bank_augmentation import (load_bank_rows, prepare_bank_rows,
                                      validate_augmentation_config)
from script.bank_data import (prepare_plan, merge_duplicate_families, save_array,
                              select_full_groups, sha256, write_json)
from script.bank_retrieval import build, load_index
from script.retrieval_io import extract_roles


class BankDataTests(unittest.TestCase):
    def test_real_bank_augmentation_is_deterministic_and_traceable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'real.png'
            pixels = np.zeros((32, 32, 3), dtype=np.uint8)
            pixels[6:26, 7:25] = np.arange(20, dtype=np.uint8)[:, None, None] + [80, 100, 120]
            Image.fromarray(pixels).save(source)
            stat = source.stat()
            frame = dict(image_path=str(source), frame_index=0, feature_path='part/video/frame.npy',
                         size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         content_sha256=sha256(source))
            plan = {'groups': {'bank': [dict(video_id='part/video', group_id='part/video', label=0,
                                              crop_settings={'face_source': 'segface'}, frames=[frame])]}}
            config = dict(enabled=True, views_per_image=1, seed=9,
                          brightness=[.85, 1.15], noise_std=[.005, .02],
                          rotation_degrees=[2., 7.], scale=[.97, 1.03],
                          shear_degrees=2., translate_fraction=.02,
                          bank_patch_fraction=.25)
            bank = root / 'bank'
            write_json(bank / 'bank_config.json', {'bank_augmentation': config})
            first = prepare_bank_rows(plan, bank, config, workers=2)
            second = prepare_bank_rows(plan, bank, config, workers=1)
            self.assertEqual(first, second)
            self.assertEqual(load_bank_rows(plan, bank), first)
            augmented = first[1]
            self.assertEqual(augmented['source_image_path'], str(source))
            self.assertEqual(augmented['augmentation']['bank_patch_fraction'], .25)
            self.assertTrue(Path(augmented['image_path']).is_file())
            with Image.open(augmented['image_path']) as image:
                result = np.asarray(image)
            self.assertTrue((result[0, 0] == 0).all())
            self.assertFalse(np.array_equal(result, pixels))
            self.assertGreater(abs(augmented['augmentation']['parameters']['rotation_degrees']), 0)
            self.assertGreater(augmented['augmentation']['parameters']['noise_std'], 0)

            write_json(bank / 'splits.json', plan)
            for index, record in enumerate(first):
                feature = np.full((2, 2, 3), index + 1, dtype=np.float16)
                path = bank / record['feature_path']
                path.parent.mkdir(parents=True, exist_ok=True)
                save_array(path, feature)
                write_json(path.with_suffix('.json'), {
                    'image': record, 'shape': list(feature.shape), 'dtype': str(feature.dtype)})
            args = SimpleNamespace(workers=1, foreground_minimum=.25, top_fraction=.1,
                                   boundary_weight=.5, face_source='segface',
                                   bank_dir=str(root), experiment='bank',
                                   bank_augmentation=config)
            output = root / 'output'
            build(plan, args, bank, output, first)
            info, sources, arrays = load_index(plan, bank, output)
            self.assertEqual(info['original_image_count'], 1)
            self.assertEqual(info['augmented_image_count'], 1)
            self.assertEqual(len(sources), 2)
            self.assertEqual(len(arrays['features']), 5)  # Four original + one augmented patch.

    def test_augmentation_config_rejects_unsafe_ranges(self):
        with self.assertRaisesRegex(ValueError, 'rotation_degrees'):
            validate_augmentation_config(dict(
                enabled=True, views_per_image=1, seed=1, brightness=[.8, 1.2],
                noise_std=[0., .01], rotation_degrees=[0., 30.], scale=[.9, 1.1],
                shear_degrees=2., translate_fraction=.02, bank_patch_fraction=.25))

    def test_duplicate_family_union_is_transitive_and_keeps_derivatives(self):
        with tempfile.TemporaryDirectory() as tmp:
            def video(name, group, label, contents):
                frames = []
                for i, content in enumerate(contents):
                    path = Path(tmp) / f"{name}-{i}.jpg"
                    path.write_bytes(content)
                    frames.append({"image_path": str(path)})
                return dict(video_id=name, group_id=group, label=label, frames=frames)
            videos = [video("real_a", "a", 0, [b"first"]),
                      video("real_b", "b", 0, [b"first", b"second"]),
                      video("real_c", "c", 0, [b"second"]),
                      video("fake_b", "b", 1, [b"fake"])]
            videos += [video(f"real_{i}", f"other_{i}", 0, [str(i).encode()]) for i in range(20)]
            audit = merge_duplicate_families(videos, workers=2)
            self.assertEqual(audit["merged_families"], [["a", "b", "c"]])
            self.assertEqual(audit["cross_family_duplicate_hashes"], 2)
            self.assertEqual({v["group_id"] for v in videos[:4]}, {"a"})
            self.assertEqual(videos[3]["source_group_id"], "b")
            groups = select_full_groups(videos, seed=42)
            partition = {v["video_id"]: ("training" if role in ("bank", "train_fake") else role)
                         for role, values in groups.items() for v in values}
            self.assertEqual(len({partition[v["video_id"]] for v in videos[:4]}), 1)
            owners = {}
            for role, values in groups.items():
                role = "training" if role in ("bank", "train_fake") else role
                for v in values:
                    for frame in v["frames"]:
                        self.assertEqual(owners.setdefault(frame["content_sha256"], role), role)

    def test_segface_source_keeps_metadata_and_skips_missing_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            faces = root / "faces"
            source = root / "source"
            normal = root / "normal"
            anomaly = root / "anomaly"
            part, video = "dfdc_train_part_0", "abcdefghij"
            manifest = faces / part / video / "metadata.json"
            manifest.parent.mkdir(parents=True)
            normal.mkdir()
            anomaly.mkdir()
            manifest.write_text(json.dumps({
                "source": str(source / part / f"{video}.mp4"),
                "label": 0,
                "settings": {"image_size": 224},
                "frames": [
                    {"status": "ok", "frame_index": 0, "file": "frame_000000.jpg"},
                    {"status": "ok", "frame_index": 1, "file": "frame_000001.jpg"},
                ],
            }))
            raw = source / part / "metadata.json"
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps({f"{video}.mp4": {"label": "REAL"}}))
            label_csv = root / "labels.csv"
            with label_csv.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=("filename", "label"))
                writer.writeheader()
                writer.writerow({"filename": f"{video}.mp4", "label": 0})
            segmented = normal / f"{part}-{video}-frame_000000.jpg"
            segmented.write_bytes(b"segmented face")

            plan = prepare_plan({
                "faces_dir": str(faces),
                "source_root": str(source),
                "label_csv": str(label_csv),
                "face_source": "segface",
                "segface_normal_dir": str(normal),
                "segface_anomaly_dir": str(anomaly),
                "seed": 1,
                "bank_videos": 1,
                "max_frames": 10,
            }, bank_only=True)

            row = plan["groups"]["bank"][0]
            self.assertEqual(row["crop_settings"]["face_source"], "segface")
            self.assertEqual([frame["image_path"] for frame in row["frames"]], [str(segmented)])
            self.assertEqual(row["frames"][0]["frame_index"], 0)
            self.assertEqual(row["frames"][0]["feature_path"],
                             f"{part}/{video}/frame_000000.npy")

            # Splits created before feature_path was added must remain resumable.
            row["frames"][0].pop("feature_path")
            bank, cache = root / "bank", root / "cache"
            args = SimpleNamespace(devices=[], workers=1, batch_size=1, dtype="float16")
            features = np.ones((1, 14, 14, 3), dtype=np.float16)
            with patch("script.retrieval_io.load_encoder", return_value=(object(), object())), \
                    patch("script.retrieval_io.encode", return_value=(None, features)):
                extract_roles(plan, ("bank",), bank, cache, args)
            feature = bank / part / video / "frame_000000.npy"
            self.assertTrue(feature.is_file())
            self.assertTrue(feature.with_suffix(".json").is_file())


if __name__ == "__main__":
    unittest.main()
