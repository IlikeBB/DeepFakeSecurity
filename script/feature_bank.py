"""Real-only DINOv3 patch bank: prepare, extract, calibrate, evaluate, predict."""

import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import yaml

from script.bank_data import image_records, prepare_plan, read_json, save_array, sha256, write_json
from script.bank_encoder import encode, load_encoder
from script.bank_search import PatchBank


def parse_args(argv=None, profile="feature_bank"):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "utils/config.yaml").read_text())
    defaults = dict(config["feature_bank"], model_path=config["model_path"])
    if profile == "patch_bank":
        defaults.update(config[profile])
    parser = argparse.ArgumentParser(description=__doc__)
    if profile == "patch_bank":
        for key in ("normal_dir", "anomaly_dir"):
            parser.add_argument("--" + key.replace("_", "-"))
        parser.add_argument("--train-adapter", action=argparse.BooleanOptionalAction)
        parser.add_argument("--full-data", action="store_true", help="Use all segmented images with family-disjoint splits")
        for key in ("train_fake_videos", "train_epochs", "train_batch_size", "adapter_hidden", "reference_patches",
                    "fake_interval", "fake_batch_size"):
            parser.add_argument("--" + key.replace("_", "-"), type=int)
        for key in ("learning_rate", "anchor_weight", "fake_weight", "fake_margin"):
            parser.add_argument("--" + key.replace("_", "-"), type=float)
    parser.add_argument("--stage", choices=("build", "visualize", "all", "prepare", "extract", "train", "evaluate", "predict", "stage1", "stage2"),
                        default=defaults.get("stage", "visualize" if defaults["storage_format"] == "jpg" else "all"))
    parser.add_argument("--image", type=Path, help="Already-cropped face image; required for predict")
    for key in ("experiment", "device", "faces_dir", "source_root", "label_csv", "model_path", "output_dir", "results_dir"):
        parser.add_argument("--" + key.replace("_", "-"))
    for key in ("seed", "bank_videos", "calibration_videos", "eval_real_videos", "eval_fake_videos",
                "max_frames", "batch_size", "query_chunk_size", "bank_chunk_size"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    for key in ("top_fraction", "threshold_quantile"):
        parser.add_argument("--" + key.replace("_", "-"), type=float)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.stage in ("stage1", "stage2") and profile != "patch_bank":
        parser.error("stage1 and stage2 require --task patch-bank")
    if profile == "patch_bank" and args.stage in ("build", "visualize"):
        parser.error("patch-bank uses prepare, extract, train, evaluate, all, or predict")
    if args.stage == "train" and (profile != "patch_bank" or not args.train_adapter):
        parser.error("train requires patch-bank with train_adapter enabled")
    if profile == "patch_bank":
        for key in ("train_epochs", "train_batch_size", "adapter_hidden", "reference_patches", "fake_interval", "fake_batch_size"):
            if type(getattr(args, key)) is not int or getattr(args, key) < 1:
                parser.error(f"{key} must be a positive integer")
        if type(args.train_fake_videos) is not int or args.train_fake_videos < 0:
            parser.error("train_fake_videos must be nonnegative")
        for key in ("learning_rate", "anchor_weight", "fake_weight", "fake_margin"):
            if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
                parser.error(f"{key} must be finite and positive")
        if args.train_adapter and args.bank_videos < 2:
            parser.error("Adapter training needs at least two real bank videos")
        if args.reference_patches < args.bank_videos:
            parser.error("reference_patches must cover at least one patch per bank video")
    for key in ("bank_videos", "calibration_videos", "eval_real_videos", "eval_fake_videos", "max_frames",
                "batch_size", "query_chunk_size", "bank_chunk_size"):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            parser.error(f"{key} must be a positive integer")
    if not 0 < args.top_fraction <= 1 or not 0 < args.threshold_quantile < 1:
        parser.error("top_fraction must be in (0,1]; threshold_quantile must be in (0,1)")
    if args.dtype not in ("float16", "float32") or args.storage_format not in ("npy", "jpg"):
        parser.error("Use npy or jpg storage and float16 or float32 dtype")
    if args.storage_format == "jpg" and args.stage not in ("visualize", "prepare"):
        parser.error("JPG is a visualization; set storage_format: npy and use a new experiment for numeric search")
    if not 1 <= args.jpeg_quality <= 100 or args.visualization_size < 1 or args.pca_images < 1:
        parser.error("Invalid JPEG quality, visualization size, or PCA image count")
    if not args.experiment or args.experiment in (".", "..") or Path(args.experiment).name != args.experiment:
        parser.error("experiment must be a single directory name")
    if args.stage == "predict" and args.image is None:
        parser.error("--stage predict requires --image (already-cropped face)")
    paths = ["faces_dir", "source_root", "label_csv", "model_path", "output_dir", "results_dir"]
    if profile == "patch_bank":
        paths.extend(("normal_dir", "anomaly_dir"))
    for key in paths:
        setattr(args, key, str((root / Path(getattr(args, key)).expanduser()).resolve()))
    for output in (args.output_dir, args.results_dir):
        sources = [args.faces_dir, args.source_root]
        if profile == "patch_bank":
            sources.extend((args.normal_dir, args.anomaly_dir))
        for source in sources:
            if Path(output) == Path(source) or Path(source) in Path(output).parents:
                parser.error("Feature outputs must be outside source/crop datasets")
    if args.output_dir == args.results_dir:
        parser.error("Bank and results directories must be separate")
    return args


def experiment_spec(args):
    runtime = {"stage", "image", "device", "batch_size", "query_chunk_size", "bank_chunk_size"}
    spec = {key: value for key, value in vars(args).items() if key not in runtime}
    if not spec.get("full_data"):
        spec.pop("full_data", None)  # Preserve existing small-experiment fingerprints.
    model = Path(args.model_path)
    spec.update(schema_version=4, representation="per-image spatial patch map [H,W,D]; registers excluded",
                layout="source folders with per-image NPY and one preview" if args.stage == "build" or
                       getattr(args, "input_layout", "nested") == "flat" else "legacy",
                stored_normalization="model output only; no additional L2",
                search="L2-normalized float32 exact cosine nearest patch",
                weights_sha256=sha256(model / "model.safetensors"),
                model_config_sha256=sha256(model / "config.json"),
                processor_sha256=sha256(model / "preprocessor_config.json"),
                labels_sha256=sha256(args.label_csv))
    return spec


def role_dir(bank_root, results_root, role):
    return bank_root if role == "bank" else results_root / role


def load_sample(directory, record):
    if "size" in record:
        stat = Path(record["image_path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"Crop changed since sampling: {record['image_path']}; use a new experiment")
    if "feature_path" in record:
        from script.bank_export import load_feature
        return None, load_feature(directory, record)
    files = directory / "features"
    index = record["sample_id"]
    metadata = read_json(files / f"image_{index:08d}.json")
    if metadata["image"] != record:
        raise ValueError("Feature provenance mismatch; use another experiment name")
    cls = np.load(files / f"cls_{index:08d}.npy", mmap_mode="r", allow_pickle=False)
    patches = np.load(files / f"patches_{index:08d}.npy", mmap_mode="r", allow_pickle=False)
    if (list(cls.shape) != metadata["cls_shape"] or list(patches.shape) != metadata["patch_shape"]
            or cls.ndim != 1 or patches.ndim != 3 or patches.shape[-1] != len(cls)
            or patches.dtype != metadata["dtype"]
            or cls.dtype != metadata["dtype"] or not np.isfinite(patches).all() or not np.isfinite(cls).all()):
        raise ValueError(f"Invalid image feature: {files}, sample_id={index}")
    return cls, patches


def extract_features(plan, args, bank_root, results_root, roles=None):
    flat = getattr(args, "input_layout", "nested") == "flat"
    if flat and (roles is None or "bank" in roles):
        from script.bank_export import build_bank
        build_bank(plan, args, bank_root)
    model, processor = load_encoder(args)
    for role, videos in plan["groups"].items():
        if roles is not None and role not in roles:
            continue
        if flat and role == "bank":
            continue
        if role == "train_fake" and (not getattr(args, "train_adapter", False) or not args.train_fake_videos):
            continue
        directory = role_dir(bank_root, results_root, role)
        files = directory / "features"
        (directory if flat else files).mkdir(parents=True, exist_ok=True)
        manifest = image_records(videos, role)
        for start in range(0, len(manifest), args.batch_size):
            pending = []
            for record in manifest[start:start + args.batch_size]:
                stat = Path(record["image_path"]).stat()
                if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
                    raise ValueError(f"Crop changed since sampling: {record['image_path']}; use a new experiment")
                metadata_path = ((directory / record["feature_path"]).with_suffix(".json") if flat else
                                 files / f"image_{record['sample_id']:08d}.json")
                if metadata_path.exists():
                    load_sample(directory, record)
                else:
                    pending.append(record)
            if pending:
                cls, patches = encode([r["image_path"] for r in pending], model, processor, args)
                for row, record in enumerate(pending):
                    if flat:
                        destination = directory / record["feature_path"]
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        save_array(destination, patches[row])
                        write_json(destination.with_suffix(".json"), {"image": record,
                                   "shape": list(patches[row].shape), "dtype": args.dtype})
                        continue
                    index = record["sample_id"]
                    save_array(files / f"cls_{index:08d}.npy", cls[row])
                    save_array(files / f"patches_{index:08d}.npy", patches[row])
                    write_json(files / f"image_{index:08d}.json", {"image": record,
                               "cls_shape": list(cls[row].shape), "patch_shape": list(patches[row].shape), "dtype": args.dtype})
            print(f"[{role}] {min(start + args.batch_size, len(manifest))}/{len(manifest)} images ready", flush=True)
        temporary = directory / "manifest.jsonl.tmp"
        temporary.write_text("".join(json.dumps(row) + "\n" for row in manifest))
        temporary.replace(directory / "manifest.jsonl")


def visualize_features(plan, args, bank_root):
    """Render independent image feature maps using one shared real-bank PCA basis."""
    records = image_records(plan["groups"]["bank"], "bank")
    for record in records:
        if record["label"] != 0:
            raise ValueError("Only real images may enter this bank")
        stat = Path(record["image_path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"Crop changed since sampling: {record['image_path']}")
    model, processor = load_encoder(args)
    projection_file = bank_root / "visualization.json"
    if not projection_file.exists():
        indices = np.linspace(0, len(records) - 1, min(args.pca_images, len(records)), dtype=int)
        _, features = encode([records[i]["image_path"] for i in indices], model, processor, args)
        values = torch.from_numpy(features.reshape(-1, features.shape[-1]).astype(np.float32))
        mean = values.mean(0)
        torch.manual_seed(args.seed)
        _, _, components = torch.pca_lowrank(values - mean, q=3, center=False, niter=4)
        colors = ((values - mean) @ components).numpy()
        write_json(projection_file, {"method": "shared real-bank PCA to RGB; display only",
                   "fit_image_paths": [records[i]["image_path"] for i in indices],
                   "mean": mean.tolist(), "components": components.tolist(),
                   "low": np.quantile(colors, .01, axis=0).tolist(),
                   "high": np.quantile(colors, .99, axis=0).tolist()})
    projection = read_json(projection_file)
    mean = np.asarray(projection["mean"], dtype=np.float32)
    components = np.asarray(projection["components"], dtype=np.float32)
    low, high = np.array(projection["low"]), np.array(projection["high"])
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        pending = []
        for record in batch:
            relative = Path(record["image_path"]).relative_to(args.faces_dir).with_suffix(".jpg")
            record["feature_image"] = relative.as_posix()
            if not (bank_root / relative).exists():
                pending.append(record)
        if pending:
            _, features = encode([r["image_path"] for r in pending], model, processor, args)
            for record, feature in zip(pending, features):
                colors = (feature.astype(np.float32) - mean) @ components
                rgb = np.rint(np.clip((colors - low) / np.maximum(high - low, 1e-6), 0, 1) * 255).astype(np.uint8)
                display = Image.fromarray(rgb).resize((args.visualization_size, args.visualization_size), Image.Resampling.NEAREST)
                destination = bank_root / record["feature_image"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".jpg.tmp")
                display.save(temporary, format="JPEG", quality=args.jpeg_quality, subsampling=0)
                temporary.replace(destination)
        print(f"[real feature JPG] {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)
    temporary = bank_root / "manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(record) + "\n" for record in records))
    temporary.replace(bank_root / "manifest.jsonl")


def load_bank(plan, args, bank_root, adapter=None):
    from script.bank_adapter import adapt_features
    arrays, sources = [], []
    start = 0
    for record in image_records(plan["groups"]["bank"], "bank"):
        if record["label"] != 0:
            raise ValueError("Only real videos may enter the bank")
        _, patches = load_sample(bank_root, record)
        patches = adapt_features(patches, adapter)
        if adapter is not None:
            destination = bank_root / "adapted" / record["feature_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_array(destination, patches.astype(args.dtype))
        arrays.append(patches.reshape(-1, patches.shape[-1]).copy())
        sources.append(dict(record, patch_id_start=start, patch_id_stop=start + len(arrays[-1]),
                            grid_shape=list(patches.shape[:2])))
        start += len(arrays[-1])
    write_json(bank_root / "patch_index.json", sources)
    return PatchBank(np.concatenate(arrays), args.device, args.query_chunk_size, args.bank_chunk_size)


def score_group(videos, directory, bank, args, adapter=None):
    from script.bank_adapter import adapt_features
    records = image_records(videos, directory.name)
    scores_dir = directory / "scores"
    scores_dir.mkdir(exist_ok=True)
    for number, record in enumerate(records, 1):
        _, patches = load_sample(directory, record)
        patches = adapt_features(patches, adapter)
        scores, distances, neighbors = bank.score(patches.reshape(1, -1, patches.shape[-1]), args.top_fraction)
        index = record["sample_id"]
        save_array(scores_dir / f"distances_{index:08d}.npy", distances[0].reshape(patches.shape[:2]))
        save_array(scores_dir / f"neighbors_{index:08d}.npy", neighbors[0].reshape(patches.shape[:2]))
        record["score"] = float(scores[0])
        if number % args.batch_size == 0 or number == len(records):
            print(f"[score] {number}/{len(records)} images", flush=True)
    return records


def metrics(records, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.array([row["label"] for row in records])
    scores = np.array([row["score"] for row in records])
    predictions = scores > threshold
    return {"count": len(records), "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "false_positive_rate": float(predictions[labels == 0].mean()),
            "true_positive_rate": float(predictions[labels == 1].mean()), "threshold": threshold}


def evaluate(plan, args, bank_root, results_root):
    from script.bank_adapter import load_adapter
    adapter = load_adapter(args, bank_root, results_root)
    bank = load_bank(plan, args, bank_root, adapter)
    calibration = score_group(plan["groups"]["calibration"], results_root / "calibration", bank, args, adapter)
    threshold = float(np.quantile([row["score"] for row in calibration], args.threshold_quantile))
    patch_values = np.concatenate([np.load(results_root / "calibration/scores" / f"distances_{r['sample_id']:08d}.npy",
                                           allow_pickle=False).ravel() for r in calibration])
    heatmap_vmax = max(float(np.quantile(patch_values, .995)), 1e-6)
    write_json(results_root / "thresholds.json", {"image_threshold": threshold, "quantile": args.threshold_quantile,
               "top_fraction": args.top_fraction,
               "bank_config_sha256": sha256(bank_root / "bank_config.json"),
               "plan_sha256": sha256(bank_root / "splits.json"),
               "calibration_images": len(calibration), "heatmap_vmax": heatmap_vmax,
               "adapter_sha256": sha256(results_root / "training/adapter.safetensors") if adapter is not None else None})
    write_json(results_root / "calibration_scores.json", {"images": calibration})
    evaluated = score_group(plan["groups"]["evaluation"], results_root / "evaluation", bank, args, adapter)
    report = {"protocol": plan["protocol"], "image": metrics(evaluated, threshold)}
    for record in evaluated:
        record["is_anomalous"] = bool(record["score"] > threshold)
    write_json(results_root / "evaluation_scores.json", {"images": evaluated})
    write_json(results_root / "metrics.json", report)
    if getattr(args, "input_layout", "nested") == "flat":
        from script.bank_heatmap import save_video_previews
        save_video_previews(evaluated, results_root / "evaluation", heatmap_vmax, threshold)
    print(json.dumps(report, indent=2), flush=True)


def predict(plan, args, bank_root, results_root):
    from script.bank_adapter import adapt_features, load_adapter
    adapter = load_adapter(args, bank_root, results_root)
    calibration = read_json(results_root / "thresholds.json")
    if (calibration["bank_config_sha256"] != sha256(bank_root / "bank_config.json")
            or calibration["plan_sha256"] != sha256(bank_root / "splits.json")):
        raise ValueError("Calibration does not match this bank; run evaluate again")
    adapter_hash = sha256(results_root / "training/adapter.safetensors") if adapter is not None else None
    if calibration.get("adapter_sha256") != adapter_hash:
        raise ValueError("Calibration does not match adapter weights; run evaluate again")
    model, processor = load_encoder(args)
    _, patches = encode([args.image], model, processor, args)
    patches = adapt_features(patches, adapter)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    bank = load_bank(plan, args, bank_root, adapter)
    scores, distances, neighbors = bank.score(patches.reshape(1, -1, patches.shape[-1]), args.top_fraction)
    threshold = calibration["image_threshold"]
    directory = results_root / "predictions" / sha256(args.image)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    save_array(directory / "patch_distances.npy", distances[0].reshape(patches.shape[1:3]))
    save_array(directory / "neighbor_ids.npy", neighbors[0].reshape(patches.shape[1:3]))
    result = {"image": str(args.image.resolve()), "score": float(scores[0]), "threshold": threshold,
              "is_anomalous": bool(scores[0] > threshold),
              "note": "Anomaly score measures deviation from the real bank, not fake probability"}
    write_json(directory / "prediction.json", result)
    if getattr(args, "input_layout", "nested") == "flat":
        from script.bank_heatmap import save_heatmap
        save_heatmap(args.image, distances[0].reshape(patches.shape[1:3]), directory / "preview.jpg",
                     calibration["heatmap_vmax"], result["score"], threshold)
    print(json.dumps(result, indent=2), flush=True)


@contextmanager
def experiment_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("w") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This experiment is already running") from None
        yield


def main(argv=None, profile="feature_bank"):
    args = parse_args(argv, profile)
    bank_root = Path(args.output_dir) / args.experiment
    results_root = Path(args.results_dir) / args.experiment
    if args.device == "auto":
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable; use --device cpu")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    with experiment_lock(bank_root):
        spec = experiment_spec(args)
        config_file = bank_root / "bank_config.json"
        if config_file.exists() and read_json(config_file) != spec:
            raise ValueError("Experiment settings changed; use a new --experiment name")
        write_json(config_file, spec)
        plan_file = bank_root / "splits.json"
        if plan_file.exists():
            plan = read_json(plan_file)
        elif args.stage in ("prepare", "all", "visualize", "build", "stage1"):
            if getattr(args, "input_layout", "nested") == "flat":
                from script.bank_flat import prepare_flat_plan
                plan = prepare_flat_plan(vars(args))
            else:
                plan = prepare_plan(vars(args), bank_only=args.stage == "build")
            write_json(plan_file, plan)
        else:
            raise ValueError("Run --stage prepare first")
        if args.stage not in ("build", "visualize", "prepare"):
            results_root.mkdir(parents=True, exist_ok=True)
        print(f"Device: {args.device}; bank: {bank_root}", flush=True)
        print(json.dumps({role: len(videos) for role, videos in plan["groups"].items()}), flush=True)
        print(json.dumps({role: sum(len(v["frames"]) for v in videos) for role, videos in plan["groups"].items()}), flush=True)
        if args.stage == "stage1":
            from script.bank_adapter import load_adapter, train_adapter
            extract_features(plan, args, bank_root, results_root, roles={"bank", "train_fake"})
            if args.train_adapter:
                train_adapter(plan, args, bank_root, results_root)
            load_bank(plan, args, bank_root, load_adapter(args, bank_root, results_root))
            write_json(bank_root / "stage1_complete.json", {"config_sha256": sha256(config_file),
                       "plan_sha256": sha256(plan_file)})
            print("Stage 1 complete: real bank and adapter ready. Run stage2 for held-out evaluation.", flush=True)
            return
        if args.stage == "stage2":
            from script.bank_adapter import load_adapter
            complete = bank_root / "stage1_complete.json"
            if not complete.exists() or read_json(complete) != {"config_sha256": sha256(config_file),
                                                               "plan_sha256": sha256(plan_file)}:
                raise ValueError("Complete stage1 with the same experiment settings before running stage2")
            load_adapter(args, bank_root, results_root)
            extract_features(plan, args, bank_root, results_root, roles={"calibration", "evaluation"})
            evaluate(plan, args, bank_root, results_root)
            return
        if args.stage == "build":
            from script.bank_export import build_bank
            build_bank(plan, args, bank_root)
        if args.stage == "visualize":
            visualize_features(plan, args, bank_root)
        if args.stage in ("extract", "all"):
            extract_features(plan, args, bank_root, results_root)
        if args.stage in ("train", "all") and getattr(args, "train_adapter", False):
            from script.bank_adapter import train_adapter
            train_adapter(plan, args, bank_root, results_root)
        if args.stage in ("evaluate", "all"):
            evaluate(plan, args, bank_root, results_root)
        if args.stage == "predict":
            predict(plan, args, bank_root, results_root)
