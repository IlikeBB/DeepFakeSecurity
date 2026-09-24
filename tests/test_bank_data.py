import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from script.bank_data import prepare_plan, merge_duplicate_families, select_full_groups
from script.retrieval_io import extract_roles


class BankDataTests(unittest.TestCase):
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
