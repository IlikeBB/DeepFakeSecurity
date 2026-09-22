"""Reproducible DFDC pilot splits and atomic artifact storage."""

import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_array(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, value, allow_pickle=False)
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_groups(candidates, config):
    """Keep an original video and its known derivatives in the same role."""
    rng = random.Random(config["seed"])
    real = sorted((v for v in candidates if v["label"] == 0), key=lambda v: v["video_id"])
    fake = sorted((v for v in candidates if v["label"] == 1), key=lambda v: v["video_id"])
    rng.shuffle(real)
    rng.shuffle(fake)
    used = set()

    def take(pool, count):
        chosen = []
        for video in pool:
            if video["group_id"] not in used:
                chosen.append(video)
                used.add(video["group_id"])
                if len(chosen) == count:
                    return chosen
        raise ValueError(f"Not enough independent source groups: need {count}, found {len(chosen)}")

    # Reserve fake source families before selecting real reference samples.
    eval_fake = take(fake, config["eval_fake_videos"])
    bank = take(real, config["bank_videos"])
    calibration = take(real, config["calibration_videos"])
    eval_real = take(real, config["eval_real_videos"])
    return {"bank": bank, "calibration": calibration, "evaluation": eval_real + eval_fake}


def prepare_plan(config, bank_only=False):
    faces_root = Path(config["faces_dir"])
    source_root = Path(config["source_root"])
    if not faces_root.is_dir():
        raise FileNotFoundError(f"Face crop directory does not exist: {faces_root}")
    with Path(config["label_csv"]).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    labels = {}
    for row in rows:
        stem, label = Path(row["filename"]).stem, int(row["label"])
        if stem in labels or label not in (0, 1):
            raise ValueError(f"Duplicate video or invalid label: {stem}")
        labels[stem] = label
    candidates, found, raw_metadata = [], set(), {}
    no_faces = 0
    for manifest in sorted(faces_root.glob("*/*/metadata.json")):
        stem = manifest.parent.name
        if stem not in labels:
            continue
        if stem in found:
            raise ValueError(f"Duplicate cropped video: {stem}")
        found.add(stem)
        if bank_only and labels[stem] != 0:
            continue
        data = read_json(manifest)
        if data["label"] != labels[stem]:
            raise ValueError(f"Crop/CSV label mismatch: {stem}")
        part = manifest.parent.parent.name
        if part not in raw_metadata:
            raw_metadata[part] = read_json(source_root / part / "metadata.json")
        original = raw_metadata[part][stem + ".mp4"]
        if original["label"] != ("FAKE" if labels[stem] else "REAL"):
            raise ValueError(f"Source/CSV label mismatch: {stem}")
        group = Path(original["original"]).stem if labels[stem] else stem
        frames = sorted((f for f in data["frames"] if f["status"] == "ok"), key=lambda f: f["frame_index"])
        if not frames:
            no_faces += 1
            continue
        indices = np.linspace(0, len(frames) - 1, min(config["max_frames"], len(frames)), dtype=int)
        selected = []
        for i in indices:
            frame = frames[i]
            path = (manifest.parent / frame["file"]).resolve()
            if manifest.parent.resolve() not in path.parents:
                raise ValueError(f"Crop path escapes its video directory: {path}")
            selected.append({"image_path": str(path), "frame_index": frame["frame_index"]})
        candidates.append({"video_id": f"{part}/{stem}", "group_id": f"{part}/{group}",
                           "label": labels[stem], "source": data["source"],
                           "crop_settings": data["settings"], "frames": selected})
    if bank_only:
        random.Random(config["seed"]).shuffle(candidates)
        if not candidates:
            raise ValueError("No real face images found")
        groups = {"bank": candidates[:config["bank_videos"]]}
    else:
        groups = select_groups(candidates, config)
    # Only stat selected images, rather than every frame in the full dataset.
    for videos in groups.values():
        for video in videos:
            for frame in video["frames"]:
                stat = Path(frame["image_path"]).stat()
                frame.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    return {"protocol": ("Real-only independent image bank; no calibration or evaluation" if bank_only else
                         "DFDC test-list pilot; source-family-disjoint internal splits, not official test results"),
            "audit": {"csv_videos": len(labels), "missing_manifests": len(labels.keys() - found),
                      "zero_face_videos": no_faces, "eligible_videos": len(candidates)},
            "groups": groups}


def image_records(videos, role):
    records = []
    for video in videos:
        for frame in video["frames"]:
            records.append(dict(frame, sample_id=len(records), role=role, video_id=video["video_id"],
                                group_id=video["group_id"], label=video["label"],
                                crop_settings=video["crop_settings"]))
    return records
