"""Segment existing DFDC face crops and export flat, traceable image filenames."""

import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch
from tqdm.auto import tqdm
import yaml

from script.bank_data import read_json
from script.face_crop import crop_masked_face, expand_mask
from script.face_parser import SegFaceParser


def parse_args(argv=None):
    project = Path(__file__).resolve().parents[1]
    defaults = yaml.safe_load((project / "utils/config.yaml").read_text())["segment_face"]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("input_dir", "output_dir", "normal_output_dir", "anomaly_output_dir", "device", "background", "labels"):
        parser.add_argument("--" + name.replace("_", "-"))
    for name in ("batch_size", "limit", "frames_per_video", "cpu_threads"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--dilation-ratio", type=float)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--parts", nargs="+", type=int)
    parser.add_argument("--manifest-list", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--lock-fds", nargs="+", type=int, default=[], help=argparse.SUPPRESS)
    parser.add_argument("--progress-position", type=int, default=0, help=argparse.SUPPRESS)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for name in ("input_dir", "normal_output_dir", "anomaly_output_dir", "model_dir"):
        setattr(args, name, str((project / Path(getattr(args, name)).expanduser()).resolve()))
    if args.output_dir:
        if args.labels == "all":
            parser.error("Use --normal-output-dir and --anomaly-output-dir when labels=all")
        name = "normal_output_dir" if args.labels == "real" else "anomaly_output_dir"
        setattr(args, name, str((project / Path(args.output_dir).expanduser()).resolve()))
    source = Path(args.input_dir)
    outputs = [Path(args.normal_output_dir), Path(args.anomaly_output_dir)]
    for output in outputs:
        if source == output or source in output.parents or output in source.parents:
            parser.error("Input and output must be separate, non-overlapping directories")
    if outputs[0] == outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:
        parser.error("Normal and anomaly outputs must be separate directories")
    if not source.is_dir():
        parser.error(f"Input directory does not exist: {source}")
    if args.batch_size < 1 or args.cpu_threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch_size, cpu_threads and limit must be positive")
    if args.progress_position < 0:
        parser.error("progress_position must be nonnegative")
    if args.frames_per_video is not None and args.frames_per_video < 1:
        parser.error("frames_per_video must be positive or null")
    if args.parts is not None and (not args.parts or any(type(i) is not int or i < 0 for i in args.parts)):
        parser.error("parts must be a nonempty list of nonnegative integers or null")
    if not 0 <= args.dilation_ratio <= .5 or not 0 <= args.closing_ratio <= .5:
        parser.error("Dilation and closing ratios must be in [0, 0.5]")
    if not 0 < args.min_face_area <= 1 or not 0 < args.pixel_confidence <= 1:
        parser.error("min_face_area and pixel_confidence must be in (0, 1]")
    if args.background not in ("black", "keep") or args.labels not in ("all", "real", "fake"):
        parser.error("Invalid background or labels setting")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg_quality must be in [1, 100]")
    if not args.face_ids or set(args.face_ids) & set(args.background_ids):
        parser.error("Face and background class IDs must be disjoint")
    return args


def flat_name(relative):
    if len(relative.parts) != 3 or any("-" in part for part in relative.parts[:2]):
        raise ValueError(f"Expected unambiguous part/video/frame path: {relative}")
    return "-".join(relative.with_suffix(".jpg").parts)


def save_image(path, image, quality):
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="JPEG", quality=quality, subsampling=0)
    temporary.replace(path)


def sample_frames(frames, count):
    valid = sorted((frame for frame in frames if frame["status"] == "ok"), key=lambda frame: frame["frame_index"])
    if count is None or len(valid) <= count:
        return valid
    return [valid[i] for i in np.linspace(0, len(valid) - 1, count, dtype=int)]


def collect_manifests(args):
    root = Path(args.input_dir)
    if args.parts is None:
        return sorted(root.glob("*/*/metadata.json"))
    manifests = []
    for number in sorted(set(args.parts)):
        directory = root / f"dfdc_train_part_{number}"
        if not directory.is_dir():
            raise FileNotFoundError(f"Requested data part missing: {directory}")
        manifests.extend(sorted(directory.glob("*/metadata.json")))
    return manifests


def select_manifests(args):
    """Apply class limits before distributing jobs, so worker count cannot change the sample."""
    counts, selected = {0: 0, 1: 0}, []
    for manifest in collect_manifests(args):
        label = read_json(manifest)["label"]
        if label not in (0, 1):
            raise ValueError(f"Invalid real/fake label: {manifest}")
        if args.labels != "all" and label != (0 if args.labels == "real" else 1):
            continue
        if args.limit is not None and counts[label] >= args.limit:
            continue
        selected.append(manifest)
        counts[label] += 1
    return selected


def output_directories(args):
    outputs = {0: Path(args.normal_output_dir), 1: Path(args.anomaly_output_dir)}
    if args.labels != "all":
        label = 0 if args.labels == "real" else 1
        outputs = {label: outputs[label]}
    return outputs


def lock_outputs(stack, outputs, inherited=()):
    locks = []
    for output in sorted(outputs.values()):
        output.mkdir(parents=True, exist_ok=True)
        info = output.stat()
        matching = [fd for fd in inherited if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (info.st_dev, info.st_ino)]
        if inherited and not matching:
            raise ValueError(f"Missing inherited output lock for {output}")
        lock = matching[0] if matching else os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        stack.callback(os.close, lock)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Segmentation is already writing {output}") from None
        locks.append(lock)
    return locks


def segment_video(manifest, data, args, detector):
    source_root, output = Path(args.input_dir), Path(args.output_dir)
    relative = manifest.parent.relative_to(source_root)
    if len(relative.parts) != 2:
        raise ValueError(f"Expected part/video/metadata.json: {manifest}")
    frames = sample_frames(data["frames"], args.frames_per_video)
    paths, saved = [], 0
    for frame in frames:
        path = (manifest.parent / frame["file"]).resolve()
        if path.parent != manifest.parent.resolve():
            raise ValueError(f"Crop escapes source directory: {path}")
        destination = output / flat_name(path.relative_to(source_root))
        if destination.is_file() and not args.overwrite:
            saved += 1
        else:
            paths.append(path)
    no_face = 0
    for start in range(0, len(paths), args.batch_size):
        images = []
        for path in paths[start:start + args.batch_size]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        masks = detector(images)
        for path, image, raw in zip(paths[start:start + args.batch_size], images, masks):
            filename = flat_name(path.relative_to(source_root))
            expanded = expand_mask(raw, args.dilation_ratio, args.closing_ratio, args.min_face_area)
            if expanded is None:
                no_face += 1
                if args.overwrite:
                    (output / filename).unlink(missing_ok=True)
            else:
                pixels, _ = crop_masked_face(np.array(image), expanded, args.background)
                save_image(output / filename, Image.fromarray(pixels), args.jpeg_quality)
                saved += 1
    return {"saved": saved, "no_face": no_face, "skipped": not paths}


def main(argv=None):
    args = parse_args(argv)
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)
    checkpoint = Path(args.model_dir) / args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError("SegFace weights missing; run: conda run -n pt230 python -m script.setup_segface")
    outputs = output_directories(args)
    manifests = [Path(p) for p in read_json(args.manifest_list)] if args.manifest_list else select_manifests(args)
    with ExitStack() as stack:
        lock_outputs(stack, outputs, args.lock_fds)
        # Lazy model initialization allows completed runs to resume without loading weights.
        parser = None

        def detect(images):
            nonlocal parser
            if parser is None:
                parser = SegFaceParser(vars(args), args.device)
            return parser(images)

        totals = {"videos": 0, "saved": 0, "no_face": 0, "skipped_videos": 0}
        with tqdm(total=len(manifests), desc=f"SegFace {args.device}", unit="video",
                  position=args.progress_position, dynamic_ncols=True, mininterval=.5) as progress:
            for manifest in manifests:
                data = read_json(manifest)
                if data["label"] not in (0, 1):
                    raise ValueError(f"Invalid real/fake label: {manifest}")
                if args.labels != "all" and data["label"] != (0 if args.labels == "real" else 1):
                    progress.update()
                    continue
                args.output_dir = str(outputs[data["label"]])
                result = segment_video(manifest, data, args, detect)
                totals["videos"] += 1
                for key in ("saved", "no_face"):
                    totals[key] += result[key]
                totals["skipped_videos"] += int(result["skipped"])
                progress.update()
                progress.set_postfix(saved=totals["saved"], no_face=totals["no_face"],
                                     skipped=totals["skipped_videos"], refresh=False)
        if not totals["videos"]:
            raise ValueError("No matching input metadata found")
        print(f"Segmentation complete: {totals}; outputs={outputs}", flush=True)
