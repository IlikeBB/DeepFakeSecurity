import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch

from script.bank_data import write_json
from script.face_crop import crop_masked_face, expand_mask
from script.face_parser import masks_from_logits
from script.segment_face import collect_manifests, flat_name, sample_frames, segment_video


class SegmentationTests(unittest.TestCase):
    def test_sampling_and_part_selection(self):
        frames = [dict(status="ok", frame_index=i) for i in range(32)]
        sampled = sample_frames(list(reversed(frames)), 10)
        self.assertEqual([f["frame_index"] for f in sampled], [0, 3, 6, 10, 13, 17, 20, 24, 27, 31])
        self.assertEqual(sample_frames(frames[:6], 10), frames[:6])
        self.assertEqual(sample_frames([dict(status="no_face", frame_index=0)], 10), [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part in (0, 1, 10, 11):
                file = root / f"dfdc_train_part_{part}/video/metadata.json"
                file.parent.mkdir(parents=True)
                file.write_text("{}")
            args = SimpleNamespace(input_dir=str(root), parts=[10, 0, 1])
            self.assertEqual([p.parent.parent.name for p in collect_manifests(args)],
                             ["dfdc_train_part_0", "dfdc_train_part_1", "dfdc_train_part_10"])

    def test_expansion_fills_holes_and_preserves_face(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[4:16, 8:22] = 1
        mask[8:12, 12:18] = 255
        expanded = expand_mask(mask, .1, 0, .02)
        self.assertTrue(expanded[mask == 1].all())
        self.assertTrue(expanded[8:12, 12:18].all())
        image = np.full((20, 30, 3), 120, dtype=np.uint8)
        crop, box = crop_masked_face(image, expanded, "black")
        self.assertEqual(box, [6, 2, 24, 18])
        self.assertEqual(crop.shape, (16, 18, 3))
        self.assertTrue(np.any(crop == 0))
        keep, _ = crop_masked_face(image, expanded, "keep")
        self.assertTrue((keep == 120).all())
        self.assertIsNone(expand_mask(np.full((20, 30), 255, dtype=np.uint8), .1, .01, .02))

    def test_segface_class_mapping_and_confidence(self):
        logits = torch.full((1, 19, 2, 3), -10.)
        # SegFace skin=2, neck=1, glasses=15; low-confidence pixel is ignored.
        logits[0, 2, :, 0] = 10
        logits[0, 1, :, 1] = 10
        logits[0, 15, :, 2] = 10
        logits[0, :, 1, 2] = 0
        config = dict(face_ids=[2, 15], background_ids=[1], pixel_confidence=.5)
        mask = masks_from_logits(logits, [Image.new("RGB", (3, 2))], config)[0]
        np.testing.assert_array_equal(mask, [[1, 0, 1], [1, 0, 255]])

    def test_flat_images_only_and_resume(self):
        self.assertEqual(flat_name(Path("dfdc_train_part_0/adwbthsgqb/frame_000.jpg")),
                         "dfdc_train_part_0-adwbthsgqb-frame_000.jpg")
        with self.assertRaises(ValueError):
            flat_name(Path("part-with-dash/video/frame.jpg"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root / "source", root / "output"
            directory = source / "part/video"
            directory.mkdir(parents=True)
            output.mkdir()
            for i in range(2):
                Image.new("RGB", (30, 20), color=(150, 100, 50)).save(directory / f"frame_{i:03d}.jpg")
            data = dict(source="/original/video.mp4", label=0,
                        frames=[dict(status="ok", file=f"frame_{i:03d}.jpg", frame_index=i) for i in range(2)])
            manifest = directory / "metadata.json"
            write_json(manifest, data)
            args = SimpleNamespace(input_dir=str(source), output_dir=str(output), batch_size=2,
                                   dilation_ratio=.1, closing_ratio=0, min_face_area=.02, background="black", jpeg_quality=95,
                                   overwrite=False, frames_per_video=None)
            mask = np.zeros((20, 30), dtype=np.uint8)
            mask[4:16, 8:22] = 1
            detector = Mock(return_value=[mask, np.zeros_like(mask)])
            result = segment_video(manifest, data, args, detector)
            self.assertEqual(result, dict(saved=1, no_face=1, skipped=False))
            self.assertTrue((output / "part-video-frame_000.jpg").is_file())
            self.assertFalse((output / "part-video-frame_001.jpg").exists())
            self.assertEqual([f.name for f in output.iterdir()], ["part-video-frame_000.jpg"])
            # Missing/no-face images can retry; existing JPGs need no model inference.
            detector.return_value = [np.zeros_like(mask)]
            result = segment_video(manifest, data, args, detector)
            self.assertEqual(result, dict(saved=1, no_face=1, skipped=False))
            Image.new("RGB", (30, 20)).save(output / "part-video-frame_001.jpg")
            detector.reset_mock()
            self.assertTrue(segment_video(manifest, data, args, detector)["skipped"])
            detector.assert_not_called()
            (output / "part-video-frame_000.jpg").unlink()
            detector.return_value = [mask]
            self.assertFalse(segment_video(manifest, data, args, detector)["skipped"])
            detector.assert_called_once()
            self.assertTrue(all(f.suffix == ".jpg" for f in output.iterdir()))
            # Explicit overwrite removes a stale crop if the new segmentation finds no face.
            args.overwrite = True
            detector.return_value = [mask, np.zeros_like(mask)]
            result = segment_video(manifest, data, args, detector)
            self.assertEqual(result["saved"], 1)
            self.assertFalse((output / "part-video-frame_001.jpg").exists())



if __name__ == "__main__":
    unittest.main()
