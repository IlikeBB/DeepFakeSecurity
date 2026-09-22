"""Sample labeled videos and crop the largest RetinaFace detection per frame."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import yaml


def parse_args():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "utils/config.yaml").read_text(encoding="utf-8"))
    defaults = {key: config[key] for key in ("data_root", "num_frames", "split", "limit", "overwrite")}
    defaults.update(config["crop_face"])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "output-dir", "cache-dir", "device", "label-csv"):
        parser.add_argument(f"--{name}")
    for name in ("num-frames", "limit", "workers", "cpu-threads", "image-size", "detection-size"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--split", choices=("all", "train", "test"))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    for name in ("num_frames", "workers", "cpu_threads", "image_size", "detection_size", "limit"):
        value = getattr(args, name)
        if name == "limit" and value is None:
            continue
        if type(value) is not int or value < 1:
            parser.error(f"{name} must be a positive integer")
    if not 0 < args.threshold <= 1 or not 0 <= args.margin <= 1:
        parser.error("threshold must be in (0, 1]; margin must be in [0, 1]")
    if args.split not in ("all", "train", "test") or not isinstance(args.overwrite, bool):
        parser.error("Invalid split or overwrite setting")
    if args.device != "cpu" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("device must be cpu or cuda:<index>")
    for name in ("data_root", "output_dir", "cache_dir"):
        path = Path(getattr(args, name)).expanduser()
        setattr(args, name, (root / path).resolve())
    if args.label_csv:
        args.label_csv = (root / Path(args.label_csv).expanduser()).resolve()
        if args.split == "train":
            parser.error("--label-csv describes a test set; use --split test or all")
    for path in (args.output_dir, args.cache_dir):
        if path == args.data_root or args.data_root in path.parents:
            parser.error("Outputs and model cache must be outside the source dataset")
    return args


def init_detector(args):
    global detector, model
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if args.device == "cpu" else args.device[5:]
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["DEEPFACE_HOME"] = str(args.cache_dir)
    import tensorflow as tf
    from retinaface import RetinaFace

    tf.config.threading.set_intra_op_parallelism_threads(args.cpu_threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if args.device != "cpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError(f"TensorFlow cannot access {args.device}; use --device cpu or install compatible CUDA libraries")
    cv2.setNumThreads(1)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    # Serialize initialization so concurrent workers cannot download the same weights twice.
    with (args.cache_dir / "download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        model = RetinaFace.build_model()
    detector = RetinaFace


def crop_video(path, args, split, label):
    relative = path.relative_to(args.data_root)
    destination = args.output_dir / relative.with_suffix("")
    settings = {key: getattr(args, key) for key in
                ("num_frames", "threshold", "margin", "image_size", "detection_size")}
    manifest = destination / "metadata.json"
    if manifest.exists() and not args.overwrite:
        previous = json.loads(manifest.read_text())
        if (previous.get("settings") != settings or previous.get("label") != label
                or previous.get("split") != split or previous.get("source") != str(path)):
            raise ValueError(f"Settings or labels changed for {relative}; use --overwrite or another output directory")
        return f"skipped: {relative}"
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Incomplete output: {destination}; use --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.stem}-", dir=destination.parent))
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or total < 1:
            raise RuntimeError(f"Cannot read video: {path}")
        indices = np.linspace(0, total - 1, min(args.num_frames, total), dtype=np.int64)
        records = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {index}: {path}")
            height, width = frame.shape[:2]
            scale = min(1.0, args.detection_size / max(height, width))
            small = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
            faces = detector.detect_faces(small, model=model, threshold=args.threshold, allow_upscaling=False)
            record = {"frame_index": int(index), "file": None, "status": "no_face"}
            if isinstance(faces, dict) and faces:
                face = max(faces.values(), key=lambda f: max(0, f["facial_area"][2] - f["facial_area"][0]) *
                           max(0, f["facial_area"][3] - f["facial_area"][1]))
                box = np.asarray(face["facial_area"], dtype=float)
                box *= [width / small.shape[1], height / small.shape[0]] * 2
                x1, y1, x2, y2 = box
                dx, dy = (x2 - x1) * args.margin, (y2 - y1) * args.margin
                left, top = max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy)))
                right, bottom = min(width, int(np.ceil(x2 + dx))), min(height, int(np.ceil(y2 + dy)))
                if right <= left or bottom <= top:
                    raise RuntimeError(f"Invalid face box at frame {index}: {path}")
                crop = cv2.resize(frame[top:bottom, left:right], (args.image_size, args.image_size))
                filename = f"frame_{index:06d}.jpg"
                if not cv2.imwrite(str(temporary / filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise OSError(f"Cannot write crop: {temporary / filename}")
                record.update(file=filename, status="ok", score=float(face["score"]),
                              bbox=box.tolist(), crop_box=[left, top, right, bottom])
            records.append(record)
        metadata = {"source": str(path), "label": label,
                    "label_csv": str(args.label_csv) if args.label_csv else None,
                    "split": split, "fps": float(capture.get(cv2.CAP_PROP_FPS)), "frame_count": total,
                    "settings": settings, "frames": records}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        saved = sum(record["status"] == "ok" for record in records)
        return f"saved {saved}/{len(records)} faces: {relative}"
    finally:
        capture.release()
        if temporary.exists():
            shutil.rmtree(temporary)


def collect_videos(args):
    if args.label_csv:
        with args.label_csv.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not {"filename", "label"}.issubset(reader.fieldnames or []):
                raise ValueError("Label CSV must contain filename and label columns")
            rows = list(reader)
        index = {}
        for path in args.data_root.rglob("*.mp4"):
            index.setdefault(path.stem, []).append(path)
        videos, seen = [], set()
        for row in rows:
            stem = Path(row["filename"].strip()).stem
            label = row["label"].strip()
            if label not in ("0", "1"):
                raise ValueError(f"Invalid label for {stem}: {label}")
            matches = index.get(stem, [])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one video for {stem}, found {len(matches)}")
            if stem in seen:
                raise ValueError(f"Duplicate CSV filename: {stem}")
            seen.add(stem)
            videos.append((matches[0], "test", int(label)))
        return videos
    test_paths = {line.split(maxsplit=1)[1] for line in
                  (args.data_root / "List_of_testing_videos.txt").read_text().splitlines() if line.strip()}
    videos = []
    for category in ("Celeb-real", "Celeb-synthesis", "YouTube-real"):
        for path in sorted((args.data_root / category).glob("*.mp4")):
            split = "test" if path.relative_to(args.data_root).as_posix() in test_paths else "train"
            if args.split in ("all", split):
                videos.append((path, split, int(category == "Celeb-synthesis")))
    return videos


def main():
    args = parse_args()
    videos = collect_videos(args)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise ValueError("No matching videos found")
    print(f"Processing {len(videos)} videos; workers={args.workers}; device={args.device}; "
          f"frames={args.num_frames}; output={args.output_dir}", flush=True)
    failures = 0
    with ProcessPoolExecutor(max_workers=min(args.workers, len(videos)),
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_detector, initargs=(args,)) as pool:
        futures = {pool.submit(crop_video, path, args, split, label): path for path, split, label in videos}
        for number, future in enumerate(as_completed(futures), 1):
            try:
                message = future.result()
            except Exception as error:
                failures += 1
                message = f"FAILED {futures[future]}: {error}"
            print(f"[{number}/{len(videos)}] {message}", flush=True)
    print(f"Finished: {len(videos) - failures}/{len(videos)} videos processed or already complete; "
          f"{failures} failed", flush=True)
    if failures:
        raise SystemExit(f"{failures} video(s) failed; see errors above")


if __name__ == "__main__":
    main()
