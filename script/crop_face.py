"""Sample labeled videos and crop the largest RetinaFace detection per frame."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from tqdm.auto import tqdm
import yaml


def gpu_ids(value):
    if value.lower() in ("none", "cpu"):
        return []
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated GPU IDs, e.g. 4,5,6, or none") from None
    if not values or any(gpu < 0 for gpu in values) or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("GPU IDs must be nonnegative and unique")
    return values


def parse_args():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "utils/config.yaml").read_text(encoding="utf-8"))
    defaults = {key: config[key] for key in ("data_root", "num_frames", "split", "limit", "overwrite")}
    defaults.update(config["crop_face"])
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "output-dir", "cache-dir", "device", "label-csv"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--gpus", dest="gpu_ids", type=gpu_ids,
                        help="Comma-separated GPU IDs; one RetinaFace worker per GPU")
    for name in ("num-frames", "limit", "workers", "cpu-threads", "image-size", "detection-size",
                 "detection-batch-size"):
        parser.add_argument(f"--{name}", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--split", choices=("all", "train", "test"))
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    for name in ("num_frames", "workers", "cpu_threads", "image_size", "detection_size",
                 "detection_batch_size", "limit"):
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
    if (not isinstance(args.gpu_ids, list) or any(type(gpu) is not int or gpu < 0 for gpu in args.gpu_ids)
            or len(args.gpu_ids) != len(set(args.gpu_ids))):
        parser.error("gpu_ids must be a list of unique nonnegative integers")
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
    # CPU mode intentionally hides TensorFlow's harmless CUDA probe; GPU mode keeps errors visible.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" if args.device == "cpu" else "2"
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


def detect_faces_batch(frames, args):
    """Run one RetinaFace forward pass and reuse its outputs for per-image postprocessing."""
    if not frames:
        return []
    if len({frame.shape for frame in frames}) != 1:
        return [detector.detect_faces(frame, model=model, threshold=args.threshold,
                                      allow_upscaling=False) for frame in frames]
    # RetinaFace preprocessing only converts BGR to RGB when upscaling is disabled for these
    # already-resized frames. Its underlying TensorFlow model accepts a dynamic batch axis.
    inputs = np.stack([frame[:, :, ::-1] for frame in frames]).astype(np.float32)
    outputs = model(inputs)
    results = []
    for index, frame in enumerate(frames):
        def cached_model(_values, position=index):
            return [value[position:position + 1] for value in outputs]
        results.append(detector.detect_faces(frame, model=cached_model, threshold=args.threshold,
                                             allow_upscaling=False))
    return results


def video_is_complete(path, args, split, label):
    if args.overwrite:
        return False
    destination = args.output_dir / path.relative_to(args.data_root).with_suffix("")
    manifest = destination / "metadata.json"
    if not manifest.is_file():
        return False
    try:
        previous = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    settings = {key: getattr(args, key) for key in
                ("num_frames", "threshold", "margin", "image_size", "detection_size")}
    if (previous.get("settings") != settings or previous.get("label") != label
            or previous.get("split") != split or previous.get("source") != str(path)):
        raise ValueError(f"Settings or labels changed for {path.relative_to(args.data_root)}; "
                         "use --overwrite or another output directory")
    return True


def crop_video(path, args, split, label):
    relative = path.relative_to(args.data_root)
    destination = args.output_dir / relative.with_suffix("")
    settings = {key: getattr(args, key) for key in
                ("num_frames", "threshold", "margin", "image_size", "detection_size")}
    if video_is_complete(path, args, split, label):
        return f"skipped: {relative}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.stem}-", dir=destination.parent))
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or total < 1:
            raise RuntimeError(f"Cannot read video: {path}")
        indices = np.linspace(0, total - 1, min(args.num_frames, total), dtype=np.int64)
        records = []
        pending = []

        def process_batch():
            faces_batch = detect_faces_batch([item[2] for item in pending], args)
            for (index, frame, small), faces in zip(pending, faces_batch):
                height, width = frame.shape[:2]
                record = {"frame_index": index, "file": None, "status": "no_face"}
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
            pending.clear()

        # Sequential decoding is substantially faster than seeking separately to every sampled frame.
        position = 0
        for target in indices:
            while position <= int(target):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {target}: {path}")
                position += 1
            height, width = frame.shape[:2]
            scale = min(1.0, args.detection_size / max(height, width))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            small = cv2.resize(frame, size) if size != (width, height) else frame
            pending.append((int(target), frame, small))
            if len(pending) == args.detection_batch_size:
                process_batch()
        if pending:
            process_batch()
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
        videos, seen = [], {}
        for row in rows:
            stem = Path(row["filename"].strip()).stem
            label = row["label"].strip()
            if label not in ("0", "1"):
                raise ValueError(f"Invalid label for {stem}: {label}")
            matches = index.get(stem, [])
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one video for {stem}, found {len(matches)}")
            if stem in seen:
                if seen[stem] != label:
                    raise ValueError(f"Conflicting labels for duplicate CSV filename: {stem}")
                continue
            seen[stem] = label
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
    pending = [video for video in videos if not video_is_complete(video[0], args, video[1], video[2])]
    skipped = len(videos) - len(pending)
    devices = [f"cuda:{gpu}" for gpu in args.gpu_ids] if args.gpu_ids else [args.device]
    worker_count = min(len(pending), len(devices) if args.gpu_ids else args.workers)
    print(f"Processing {len(videos)} videos; workers={worker_count}; devices={devices}; "
          f"frames={args.num_frames}; detection_batch={args.detection_batch_size}; "
          f"skipped={skipped}; output={args.output_dir}", flush=True)
    if not pending:
        print(f"Finished: {skipped}/{len(videos)} videos already complete; 0 failed", flush=True)
        return
    failures = 0
    context = multiprocessing.get_context("spawn")
    with ExitStack() as stack:
        futures = {}
        if args.gpu_ids:
            for index, device in enumerate(devices[:worker_count]):
                worker_args = argparse.Namespace(**vars(args))
                worker_args.device = device
                pool = stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=context,
                                           initializer=init_detector, initargs=(worker_args,)))
                for path, split, label in pending[index::worker_count]:
                    futures[pool.submit(crop_video, path, worker_args, split, label)] = path
        else:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=worker_count, mp_context=context,
                                       initializer=init_detector, initargs=(args,)))
            futures = {pool.submit(crop_video, path, args, split, label): path
                       for path, split, label in pending}
        with tqdm(total=len(videos), initial=skipped, desc="RetinaFace", unit="video",
                  dynamic_ncols=True, mininterval=.5) as progress:
            for number, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as error:
                    failures += 1
                    progress.write(f"FAILED {futures[future]}: {error}")
                progress.update()
                progress.set_postfix(processed=number - failures, skipped=skipped, failed=failures)
    print(f"Finished: {len(videos) - failures}/{len(videos)} videos processed or already complete; "
          f"{failures} failed", flush=True)
    if failures:
        raise SystemExit(f"{failures} video(s) failed; see errors above")


if __name__ == "__main__":
    main()
