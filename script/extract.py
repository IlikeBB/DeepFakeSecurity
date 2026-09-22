"""Frozen DINOv3 CLS features, saved separately for each Celeb-DF video."""

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


@torch.inference_mode()
def video_features(path, model, processor, device, num_frames, batch_size):
    capture = cv2.VideoCapture(str(path))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not capture.isOpened() or frame_count < 1:
            raise RuntimeError(f"Cannot open video or determine frame count: {path}")
        indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=np.int64)
        features = []
        for start in range(0, len(indices), batch_size):
            frames = []
            for index in indices[start:start + batch_size]:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {index}: {path}")
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(images=frames, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state
            features.append(hidden[:, 0].float().cpu())
        frame_features = torch.cat(features)
        return {
            "frame_features": frame_features,
            "video_feature": frame_features.mean(dim=0),
            "frame_indices": torch.from_numpy(indices),
            "frame_count": frame_count,
            "fps": fps,
        }
    finally:
        capture.release()


def extract(args):
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == data_root or data_root in output_dir.parents:
        raise ValueError("Output directory must be outside the source dataset.")
    test_list = data_root / "List_of_testing_videos.txt"
    test_paths = {line.split(maxsplit=1)[1] for line in test_list.read_text().splitlines() if line.strip()}
    videos = []
    # Our labels: real=0, fake=1 (the official test list uses the opposite convention).
    for category, label in [("Celeb-real", 0), ("Celeb-synthesis", 1), ("YouTube-real", 0)]:
        for path in sorted((data_root / category).glob("*.mp4")):
            relative = path.relative_to(data_root)
            split = "test" if relative.as_posix() in test_paths else "train"
            if args.split in ("all", split):
                videos.append((path, relative, label, split))
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise ValueError(f"No videos found for split={args.split} under {data_root}")

    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(args.device).eval()
    settings = {
        "model_path": str(args.model_path.resolve()),
        "num_frames": args.num_frames,
        "representation": "last_hidden_state_cls",
        "video_pooling": "mean",
    }
    for number, (path, relative, label, split) in enumerate(videos, 1):
        destination = output_dir / relative.with_suffix(".pt")
        if destination.exists() and not args.overwrite:
            print(f"[{number}/{len(videos)}] exists, skipped: {relative}", flush=True)
            continue
        result = video_features(path, model, processor, args.device, args.num_frames, args.batch_size)
        result.update(source=str(path), label=label, split=split, settings=settings)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(result, temporary)
        temporary.replace(destination)
        print(f"[{number}/{len(videos)}] saved: {destination}", flush=True)
