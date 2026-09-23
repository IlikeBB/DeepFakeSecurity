import csv
import json
import tempfile
import unittest
from pathlib import Path

from script.bank_data import prepare_plan


class BankDataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
