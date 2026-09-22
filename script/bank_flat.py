"""Read segmented JPGs without changing datasets or adding dataset sidecars."""

import csv
from pathlib import Path
import re
import random

import numpy as np
from tqdm.auto import tqdm

from script.bank_data import read_json, select_groups


def select_full_groups(candidates, seed):
    """Use every video; hold out real-only families for real-only calibration."""
    families = {}
    for video in sorted(candidates, key=lambda row: row["video_id"]):
        families.setdefault(video["group_id"], []).append(video)
    real_families = sorted(key for key, values in families.items() if any(v["label"] == 0 for v in values))
    real_only = [key for key in real_families if all(v["label"] == 0 for v in families[key])]
    fake_only = sorted(set(families) - set(real_families))
    rng = random.Random(seed)
    rng.shuffle(real_only)
    calibration_count = max(1, int(len(real_families) * .15))
    bank_count = int(len(real_families) * .70)
    if len(real_only) < calibration_count or bank_count < 2 or bank_count + calibration_count >= len(real_families):
        raise ValueError("Full-data split needs enough real-only families for 15% calibration and independent training/test")
    calibration = set(real_only[:calibration_count])
    remaining = sorted(set(real_families) - calibration)
    rng.shuffle(remaining)
    training = set(remaining[:bank_count])
    rng.shuffle(fake_only)
    training.update(fake_only[:int(len(fake_only) * .70)])
    groups = {"bank": [], "train_fake": [], "calibration": [], "evaluation": []}
    for key, values in families.items():
        for video in values:
            role = ("calibration" if key in calibration else
                    ("bank" if video["label"] == 0 else "train_fake") if key in training else "evaluation")
            groups[role].append(video)
    return groups


def prepare_flat_plan(config):
    with Path(config["label_csv"]).open(encoding="utf-8-sig", newline="") as file:
        labels = {}
        for row in csv.DictReader(file):
            stem, label = Path(row["filename"]).stem, int(row["label"])
            if stem in labels or label not in (0, 1):
                raise ValueError(f"Duplicate video or invalid CSV label: {stem}")
            labels[stem] = label
    videos, metadata = {}, {}
    pattern = re.compile(r"(dfdc_train_part_\d+)-([^-]+)-(frame_(\d+))\.jpg")
    for label, key in ((0, "normal_dir"), (1, "anomaly_dir")):
        directory = Path(config[key])
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for image in tqdm(sorted(directory.glob("*.jpg")), desc=f"檢查 {key}", unit="image", dynamic_ncols=True):
            match = pattern.fullmatch(image.name)
            if not match:
                raise ValueError(f"Expected part-video-frame filename: {image}")
            part, stem, frame, index = match.groups()
            if labels.get(stem) != label:
                raise ValueError(f"Directory/CSV label mismatch: {image}")
            video_id = f"{part}/{stem}"
            if video_id not in videos:
                if part not in metadata:
                    metadata[part] = read_json(Path(config["source_root"]) / part / "metadata.json")
                original = metadata[part][stem + ".mp4"]
                if original["label"] != ("FAKE" if label else "REAL"):
                    raise ValueError(f"Source metadata label mismatch: {image}")
                group = Path(original["original"]).stem if label else stem
                videos[video_id] = dict(video_id=video_id, group_id=f"{part}/{group}", label=label,
                                        crop_settings={"input": "existing segmented JPG; all patches retained"}, frames=[])
            if videos[video_id]["label"] != label:
                raise ValueError(f"Video appears in both class directories: {video_id}")
            videos[video_id]["frames"].append(dict(image_path=str(image.resolve()), frame_index=int(index),
                                                  feature_path=f"{video_id}/{frame}.npy"))
    full_data = config.get("full_data", False)
    groups = (select_full_groups(list(videos.values()), config["seed"]) if full_data else
              select_groups(list(videos.values()), config))
    for selected in groups.values():
        for video in selected:
            frames = sorted(video["frames"], key=lambda row: row["frame_index"])
            if len({f["frame_index"] for f in frames}) != len(frames):
                raise ValueError(f"Duplicate frame indices: {video['video_id']}")
            count = len(frames) if full_data else min(len(frames), config["max_frames"])
            indices = np.linspace(0, len(frames) - 1, count, dtype=int)
            video["frames"] = [frames[i] for i in indices]
            for frame in video["frames"]:
                stat = Path(frame["image_path"]).stat()
                frame.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return {"protocol": ("All segmented DFDC images; training/calibration/evaluation source-family-disjoint; "
                         "not identity-disjoint or official test results" if full_data else
                         "DFDC test-list pilot; source-family-disjoint, not identity-disjoint or official test results"),
            "audit": {"eligible_real_videos": sum(v["label"] == 0 for v in videos.values()),
                      "eligible_fake_videos": sum(v["label"] == 1 for v in videos.values())},
            "groups": groups}
