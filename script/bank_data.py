"""Reproducible DFDC pilot splits and atomic artifact storage."""

import csv
import hashlib
import json
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm.auto import tqdm


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


def save_concatenated_array(path, values, desc):
    """Atomically concatenate arrays on disk while reporting written rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if not values:
        raise ValueError(f"Cannot concatenate an empty array list: {path}")
    first = values[0]
    shape = first.shape[1:]
    dtype = first.dtype
    if any(value.shape[1:] != shape or value.dtype != dtype for value in values):
        raise ValueError(f"Cannot concatenate arrays with different shapes or dtypes: {path}")
    total = sum(len(value) for value in values)
    if not total:
        raise ValueError(f"Cannot save an empty concatenated array: {path}")
    temporary.unlink(missing_ok=True)
    mapped = None
    try:
        mapped = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=dtype, shape=(total, *shape))
        offset = 0
        with tqdm(total=total, desc=desc, unit="patch", unit_scale=True,
                  dynamic_ncols=True) as progress:
            for value in values:
                end = offset + len(value)
                mapped[offset:end] = value
                offset = end
                progress.update(len(value))
        mapped.flush()
        del mapped
        mapped = None
        temporary.replace(path)
    except BaseException:
        if mapped is not None:
            del mapped
        temporary.unlink(missing_ok=True)
        raise


def write_json_list(path, values, desc, unit="row"):
    """Atomically stream a large JSON list while reporting completed items."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            file.write("[\n")
            for index, value in enumerate(tqdm(values, total=len(values), desc=desc, unit=unit,
                                               dynamic_ncols=True)):
                if index:
                    file.write(",\n")
                rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                file.write("  " + rendered.replace("\n", "\n  "))
            file.write("\n]")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def sha256(path, desc=None):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file, tqdm(
            total=path.stat().st_size, desc=desc, unit="B", unit_scale=True,
            unit_divisor=1024, dynamic_ncols=True, disable=desc is None) as progress:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            progress.update(len(chunk))
    return digest.hexdigest()


def merge_duplicate_families(candidates, workers=8):
    """Keep exact image duplicates and all their source-family derivatives together."""
    parents = {video["group_id"]: video["group_id"] for video in candidates}

    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    frames = [(video["group_id"], frame) for video in candidates for frame in video["frames"]]
    seen, duplicate_hashes = {}, set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        hashes = pool.map(sha256, (frame["image_path"] for _, frame in frames))
        for (group, frame), digest in tqdm(zip(frames, hashes), total=len(frames),
                                          desc="Stage 1：合併重複內容家族", unit="image"):
            frame["content_sha256"] = digest
            previous = seen.setdefault(digest, group)
            if previous != group:
                duplicate_hashes.add(digest)
            left, right = sorted((root(previous), root(group)))
            parents[right] = left
    merged = {}
    for group in parents:
        merged.setdefault(root(group), []).append(group)
    for video in candidates:
        original = video["group_id"]
        video["source_group_id"] = original
        video["group_id"] = root(original)
    return {"method": "exact-jpeg-sha256-family-union-v1", "images_hashed": len(frames),
            "cross_family_duplicate_hashes": len(duplicate_hashes),
            "merged_families": [sorted(values) for _, values in sorted(merged.items()) if len(values) > 1]}


def select_groups(candidates, config):
    """Keep an original video and its known derivatives in the same role."""
    rng = random.Random(config["seed"])
    real = sorted((v for v in candidates if v["label"] == 0), key=lambda v: v["video_id"])
    fake = sorted((v for v in candidates if v["label"] == 1), key=lambda v: v["video_id"])
    rng.shuffle(real)
    rng.shuffle(fake)
    used = set()

    def take(pool, count):
        if count == 0:
            return []
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
    train_fake = take(fake, config.get("train_fake_videos", 0))
    bank = take(real, config["bank_videos"])
    calibration = take(real, config["calibration_videos"])
    eval_real = take(real, config["eval_real_videos"])
    groups = {"bank": bank, "calibration": calibration, "evaluation": eval_real + eval_fake}
    if train_fake:
        groups["train_fake"] = train_fake
    return groups


def select_full_groups(candidates, seed):
    """Use every cropped video while keeping source families disjoint."""
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
        raise ValueError("Full-data split needs enough real-only families for calibration and evaluation")
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


def prepare_plan(config, bank_only=False):
    faces_root = Path(config["faces_dir"])
    source_root = Path(config["source_root"])
    face_source = config.get("face_source", "retinaface")
    if face_source not in ("retinaface", "segface"):
        raise ValueError("face_source must be retinaface or segface")
    segmented = ({0: Path(config["segface_normal_dir"]), 1: Path(config["segface_anomaly_dir"])}
                 if face_source == "segface" else None)
    if not faces_root.is_dir():
        raise FileNotFoundError(f"Face crop directory does not exist: {faces_root}")
    if segmented:
        for directory in segmented.values():
            if not directory.is_dir():
                raise FileNotFoundError(f"SegFace directory does not exist: {directory}")
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
        frames = []
        for frame in sorted((f for f in data["frames"] if f["status"] == "ok"),
                            key=lambda f: f["frame_index"]):
            if segmented:
                filename = f"{part}-{stem}-{Path(frame['file']).with_suffix('.jpg').name}"
                path = (segmented[labels[stem]] / filename).resolve()
                if not path.is_file():
                    continue
            else:
                path = (manifest.parent / frame["file"]).resolve()
                if manifest.parent.resolve() not in path.parents:
                    raise ValueError(f"Crop path escapes its video directory: {path}")
            frames.append((frame, path))
        if not frames:
            no_faces += 1
            continue
        count = len(frames) if config.get("full_data", False) else min(config["max_frames"], len(frames))
        indices = np.linspace(0, len(frames) - 1, count, dtype=int)
        selected = []
        for i in indices:
            frame, path = frames[i]
            selected.append({"image_path": str(path), "frame_index": frame["frame_index"],
                             "feature_path": f"{part}/{stem}/frame_{frame['frame_index']:06d}.npy"})
        candidates.append({"video_id": f"{part}/{stem}", "group_id": f"{part}/{group}",
                           "label": labels[stem], "source": data["source"],
                           "crop_settings": dict(data["settings"], face_source=face_source), "frames": selected})
    duplicate_audit = None
    if config.get("deduplicate_content", False):
        duplicate_audit = merge_duplicate_families(candidates, config.get("workers", 8))
    if bank_only:
        random.Random(config["seed"]).shuffle(candidates)
        if not candidates:
            raise ValueError("No real face images found")
        groups = {"bank": candidates[:config["bank_videos"]]}
    elif config.get("full_data", False):
        groups = select_full_groups(candidates, config["seed"])
    else:
        groups = select_groups(candidates, config)
    # Only stat selected images, rather than every frame in the full dataset.
    for videos in groups.values():
        for video in videos:
            for frame in video["frames"]:
                stat = Path(frame["image_path"]).stat()
                frame.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    protocol = ("Real-only independent image bank; no calibration or evaluation" if bank_only else
                f"All {face_source}-cropped DFDC images; source-family-disjoint internal splits, "
                "not identity-disjoint or official test results" if config.get("full_data", False) else
                "DFDC test-list pilot; source-family-disjoint internal splits, not official test results")
    if duplicate_audit:
        protocol += "; exact JPEG duplicates merged before splitting (not near-duplicate-disjoint)"
    return {"protocol": protocol,
            "audit": {"csv_videos": len(labels), "missing_manifests": len(labels.keys() - found),
                      "zero_face_videos": no_faces, "eligible_videos": len(candidates),
                      **({"content_deduplication": duplicate_audit} if duplicate_audit else {})},
            "groups": groups}


def image_records(videos, role):
    records = []
    for video in videos:
        for frame in video["frames"]:
            feature_path = frame.get(
                "feature_path", f"{video['video_id']}/frame_{frame['frame_index']:06d}.npy")
            records.append(dict(frame, feature_path=feature_path, sample_id=len(records), role=role,
                                video_id=video["video_id"],
                                group_id=video["group_id"], label=video["label"],
                                crop_settings=video["crop_settings"]))
    return records
