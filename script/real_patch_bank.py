"""Fit and evaluate a position-aware anomaly model using real patch features only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
from tqdm.auto import tqdm
import yaml

from models.real_patch_bank import COMPONENTS, RealPatchBank, spatial_region_ids
from script.bank_augmentation import load_bank_rows
from script.bank_data import image_records, read_json, sha256, write_json
from script.retrieval_io import ensure_experiment_config, experiment_lock, extract_roles, load_sample, metrics


def parse_args(argv=None):
    root = Path(__file__).resolve().parents[1]
    full_config = yaml.safe_load((root / "utils/config.yaml").read_text(encoding="utf-8"))
    config = full_config["real_patch_bank"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "evaluate", "all"), default="fit")
    parser.add_argument("--exper", "--experiment", dest="experiment")
    parser.add_argument("--gpus", dest="gpu_ids", type=int, nargs="*",
                        help="GPU IDs used for validation/evaluation; empty list uses CPU")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--spatial-restriction", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip-scores", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-weights", type=float, nargs=3)
    parser.add_argument("--replace", action="store_true", help="Replace an existing real-only model")
    parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if not isinstance(args.experiment, str) or not args.experiment.strip():
        parser.error("請以 --exper 指定已完成 Stage 1 的 feature-bank 實驗名稱")
    if args.gpu_ids is None:
        args.gpu_ids = list(config["gpu_ids"])
    if len(set(args.gpu_ids)) != len(args.gpu_ids) or any(
            gpu < 0 or gpu >= torch.cuda.device_count() for gpu in args.gpu_ids):
        parser.error("--gpus 包含不存在或重複的 GPU")
    args.devices = [f"cuda:{gpu}" for gpu in args.gpu_ids] or ["cpu"]
    integers = ("batch_size", "workers", "extract_batch_size", "pca_dim",
                "patch_samples_per_region", "prototypes_per_region", "minimum_relation_samples",
                "top_patch_count")
    if any(type(getattr(args, key)) is not int or getattr(args, key) < 1 for key in integers):
        parser.error("batch、worker、PCA、sample、prototype 與 relation 數量必須是正整數")
    if (not 0 < args.validation_fraction < 1 or not 0 < args.foreground_minimum <= 1
            or not 0 < args.top_fraction <= 1 or not 0 < args.threshold_quantile < 1
            or not 0 < args.normalization_quantile < 1
            or not 0 <= args.covariance_shrinkage <= 1 or args.covariance_ridge <= 0
            or args.relation_std_floor <= 0 or type(args.neighbor_radius) is not int
            or args.neighbor_radius < 0):
        parser.error("real_patch_bank 的比例或 covariance 設定不正確")
    if (not isinstance(args.spatial_grid, list) or len(args.spatial_grid) != 2
            or any(type(value) is not int or value < 1 for value in args.spatial_grid)):
        parser.error("spatial_grid 必須是兩個正整數")
    if (not isinstance(args.score_weights, list) or len(args.score_weights) != 3
            or any(not math.isfinite(value) or value < 0 for value in args.score_weights)
            or sum(args.score_weights) <= 0):
        parser.error("score_weights 必須依序提供三個非負權重")
    if type(args.spatial_restriction) is not bool or type(args.clip_scores) is not bool:
        parser.error("spatial_restriction / clip_scores 必須是 bool")
    for name in ("bank_dir", "source_results_dir", "model_dir", "results_dir"):
        path = Path(getattr(args, name)).expanduser()
        setattr(args, name, str((root / path).resolve() if not path.is_absolute() else path.resolve()))
    return args


def _paths(args):
    bank = Path(args.bank_dir) / args.experiment
    return (bank, bank / "cache", Path(args.model_dir) / args.experiment,
            Path(args.results_dir) / args.experiment)


def _source_spec(bank):
    spec = read_json(bank / "bank_config.json")
    if spec.get("schema") != "retrieval-v1":
        raise ValueError("RealPatchBank 需要 retrieval-v1 Stage 1 feature bank")
    return spec


def _extract_args(args, spec):
    tuning = spec["encoder_tuning"]
    checkpoint = None
    if tuning["enabled"]:
        info = read_json(Path(args.source_results_dir) / args.experiment / "stage1/dino_lora.json")
        checkpoint = info["checkpoint"]
        if info["config"] != tuning or not Path(checkpoint).is_file():
            raise ValueError("Stage 1 LoRA checkpoint 與 feature bank 不一致")
    return SimpleNamespace(devices=args.devices, workers=args.workers,
                           batch_size=args.extract_batch_size, dtype=spec["dtype"],
                           model_path=spec["model_path"], encoder_tuning=tuning,
                           encoder_checkpoint=checkpoint)


def ensure_cache(plan, args, bank, cache, roles):
    rows = [row for role in roles for row in image_records(plan["groups"][role], role)]
    missing = sum(not (cache / row["role"] / row["feature_path"]).is_file()
                  or not (cache / row["role"] / row["feature_path"]).with_suffix(".json").is_file()
                  for row in rows)
    if missing:
        print(f"RealPatchBank：需要快取 {missing}/{len(rows)} 張 DINO features", flush=True)
        extract_roles(plan, roles, bank, cache, _extract_args(args, _source_spec(bank)))
    else:
        print(f"RealPatchBank：沿用 {len(rows)} 張 feature cache", flush=True)


def _split_real_rows(rows, fraction, seed):
    if not rows or any(row["label"] != 0 for row in rows):
        raise ValueError("RealPatchBank fitting data must contain real images only")
    families = sorted({row["group_id"] for row in rows})
    if len(families) < 2:
        raise ValueError("RealPatchBank needs at least two independent real source families")
    random.Random(seed).shuffle(families)
    count = min(len(families) - 1, max(1, round(len(families) * fraction)))
    validation_families = set(families[:count])
    fit = [row for row in rows if row["group_id"] not in validation_families]
    validation = [row for row in rows if row["group_id"] in validation_families]
    return fit, validation


def _foreground_mask(image_path, grid_size, minimum):
    with Image.open(image_path) as image:
        mask = (np.asarray(image.convert("RGB")).max(axis=-1) > 16).astype(np.uint8) * 255
    occupancy = np.asarray(Image.fromarray(mask).resize(
        (grid_size[1], grid_size[0]), Image.Resampling.BOX), dtype=np.float32) / 255
    result = occupancy >= minimum
    if not result.any():
        raise ValueError(f"No foreground patches: {image_path}")
    return result


def _feature_directory(bank, cache, row):
    return bank if row["role"] == "bank" else cache / row["role"]


def _load_observation(row, bank, cache, foreground_minimum, expected_grid=None):
    _, feature = load_sample(_feature_directory(bank, cache, row), row)
    if feature.ndim != 3 or not np.isfinite(feature).all():
        raise ValueError(f"Invalid patch feature: {row['feature_path']}")
    if expected_grid is not None and tuple(feature.shape[:2]) != tuple(expected_grid):
        raise ValueError(f"Feature grid changed: {row['feature_path']}")
    mask = _foreground_mask(row["image_path"], feature.shape[:2], foreground_minimum)
    return feature.astype(np.float32, copy=False), mask


class _RegionReservoir:
    """Keep a deterministic uniform random-key sample for every spatial region."""

    def __init__(self, regions, capacity, dimensions, seed):
        self.capacity = capacity
        self.samples = [np.empty((0, dimensions), dtype=np.float32) for _ in range(regions)]
        self.priorities = [np.empty(0, dtype=np.float64) for _ in range(regions)]
        self.buffers = [[] for _ in range(regions)]
        self.rng = np.random.default_rng(seed)

    def buffer(self, region, values):
        if len(values):
            self.buffers[region].append(np.asarray(values, dtype=np.float32))

    def flush(self):
        for region, chunks in enumerate(self.buffers):
            if not chunks:
                continue
            incoming = np.concatenate(chunks)
            values = np.concatenate((self.samples[region], incoming))
            priorities = np.concatenate((self.priorities[region], self.rng.random(len(incoming))))
            if len(values) > self.capacity:
                keep = np.argpartition(priorities, self.capacity - 1)[:self.capacity]
                values, priorities = values[keep], priorities[keep]
            self.samples[region], self.priorities[region] = values, priorities
            chunks.clear()


def _farthest_indices(values, count):
    """Greedy coreset selection in the compact PCA space."""
    count = min(int(count), len(values))
    if count == len(values):
        return np.arange(len(values))
    selected = np.empty(count, dtype=np.int64)
    center = values.mean(0)
    selected[0] = int(np.square(values - center).sum(1).argmax())
    minimum = np.square(values - values[selected[0]]).sum(1)
    for index in range(1, count):
        selected[index] = int(minimum.argmax())
        distance = np.square(values - values[selected[index]]).sum(1)
        minimum = np.minimum(minimum, distance)
    return selected


def _fit_statistics(rows, args, bank, cache):
    first, _ = _load_observation(rows[0], bank, cache, args.foreground_minimum)
    grid_size, input_dim = tuple(first.shape[:2]), int(first.shape[-1])
    spatial_grid = tuple(args.spatial_grid)
    region_ids = spatial_region_ids(grid_size, spatial_grid).numpy()
    region_count = spatial_grid[0] * spatial_grid[1]
    reservoir = _RegionReservoir(region_count, args.patch_samples_per_region, input_dim, args.seed)
    relation_sum = np.zeros((region_count, region_count), dtype=np.float64)
    relation_square = np.zeros_like(relation_sum)
    relation_count = np.zeros((region_count, region_count), dtype=np.int64)
    reader = partial(_load_observation, bank=bank, cache=cache,
                     foreground_minimum=args.foreground_minimum, expected_grid=grid_size)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        iterator = pool.map(reader, rows)
        for image_index, (feature, foreground) in enumerate(tqdm(
                iterator, total=len(rows), desc="Real-only fitting：讀取 real patch", unit="image",
                dynamic_ncols=True)):
            flat = feature.reshape(-1, input_dim)
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            valid = foreground.reshape(-1) & np.isfinite(norms[:, 0]) & (norms[:, 0] > 0)
            normalized = flat / np.maximum(norms, 1e-12)
            pooled = np.zeros((region_count, input_dim), dtype=np.float32)
            region_valid = np.zeros(region_count, dtype=bool)
            for region in range(region_count):
                selected = valid & (region_ids == region)
                if selected.any():
                    values = normalized[selected]
                    reservoir.buffer(region, values)
                    pooled[region] = values.mean(0)
                    pooled[region] /= max(np.linalg.norm(pooled[region]), 1e-12)
                    region_valid[region] = True
            similarity = pooled @ pooled.T
            pair_valid = region_valid[:, None] & region_valid[None, :]
            relation_sum[pair_valid] += similarity[pair_valid]
            relation_square[pair_valid] += np.square(similarity[pair_valid])
            relation_count[pair_valid] += 1
            if (image_index + 1) % 256 == 0:
                reservoir.flush()
    reservoir.flush()
    minimum_samples = max(args.pca_dim + 2, args.prototypes_per_region)
    region_samples = []
    for region in range(region_count):
        row, column = divmod(region, spatial_grid[1])
        neighboring = []
        for other_row in range(max(0, row - args.neighbor_radius),
                               min(spatial_grid[0], row + args.neighbor_radius + 1)):
            for other_column in range(max(0, column - args.neighbor_radius),
                                      min(spatial_grid[1], column + args.neighbor_radius + 1)):
                values = reservoir.samples[other_row * spatial_grid[1] + other_column]
                if len(values):
                    neighboring.append(values)
        region_samples.append(np.concatenate(neighboring) if neighboring
                              else np.empty((0, input_dim), dtype=np.float32))
    too_small = [region for region, values in enumerate(region_samples) if len(values) < minimum_samples]
    if too_small:
        raise ValueError(f"Insufficient real foreground patches in spatial neighborhoods: {too_small}")

    all_samples = np.concatenate([values for values in reservoir.samples if len(values)])
    pca_dim = min(args.pca_dim, input_dim, len(all_samples) - 1)
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=args.seed)
    pca.fit(all_samples)
    prototypes = np.zeros((region_count, args.prototypes_per_region, input_dim), dtype=np.float32)
    prototype_counts = np.zeros(region_count, dtype=np.int64)
    gaussian_means = np.zeros((region_count, pca_dim), dtype=np.float32)
    gaussian_precisions = np.zeros((region_count, pca_dim, pca_dim), dtype=np.float32)
    gaussian_valid = np.ones(region_count, dtype=bool)
    identity = np.eye(pca_dim, dtype=np.float64)
    for region, values in enumerate(tqdm(region_samples, desc="Real-only fitting：prototype / PaDiM",
                                         unit="region", dynamic_ncols=True)):
        projected = pca.transform(values).astype(np.float32, copy=False)
        selected = _farthest_indices(projected, args.prototypes_per_region)
        prototype_counts[region] = len(selected)
        prototypes[region, :len(selected)] = values[selected]
        gaussian_means[region] = projected.mean(0)
        covariance = np.cov(projected, rowvar=False).astype(np.float64, copy=False)
        diagonal = np.diag(np.diag(covariance))
        covariance = ((1 - args.covariance_shrinkage) * covariance
                      + args.covariance_shrinkage * diagonal
                      + args.covariance_ridge * identity)
        gaussian_precisions[region] = np.linalg.inv(covariance).astype(np.float32)

    valid_relations = relation_count >= args.minimum_relation_samples
    relation_means = np.divide(relation_sum, relation_count, out=np.zeros_like(relation_sum),
                               where=relation_count > 0)
    relation_variance = np.divide(relation_square, relation_count, out=np.zeros_like(relation_square),
                                  where=relation_count > 0) - np.square(relation_means)
    relation_stds = np.sqrt(np.maximum(relation_variance, args.relation_std_floor ** 2))
    return RealPatchBank(
        prototypes=prototypes, prototype_counts=prototype_counts,
        pca_mean=pca.mean_.astype(np.float32), pca_components=pca.components_.astype(np.float32),
        gaussian_means=gaussian_means, gaussian_precisions=gaussian_precisions,
        gaussian_valid=gaussian_valid, relation_means=relation_means.astype(np.float32),
        relation_stds=relation_stds.astype(np.float32), relation_valid=valid_relations,
        score_centers=np.zeros(3, dtype=np.float32), score_scales=np.ones(3, dtype=np.float32),
        score_weights=np.asarray(args.score_weights, dtype=np.float32),
        grid_size=grid_size, spatial_grid=spatial_grid, top_fraction=args.top_fraction,
        spatial_restriction=args.spatial_restriction, clip_scores=args.clip_scores), relation_count


def _result_row(row, output, top_patch_count):
    components = output["components"]
    normalized = output["normalized_components"]
    anomaly_map = output["anomaly_map"]
    valid = anomaly_map >= 0
    coordinates = np.argwhere(valid)
    scores = anomaly_map[valid]
    count = min(top_patch_count, len(scores))
    chosen = np.argsort(-scores, kind="stable")[:count]
    nearest_map, mahalanobis_map = output["nearest_map"], output["mahalanobis_map"]
    top_patches = [{"patch": [int(coordinates[index, 0]), int(coordinates[index, 1])],
                    "anomaly": float(scores[index]),
                    "nearest": float(nearest_map[tuple(coordinates[index])]),
                    "mahalanobis": float(mahalanobis_map[tuple(coordinates[index])])}
                   for index in chosen]
    return {key: row[key] for key in ("sample_id", "role", "video_id", "group_id", "label", "image_path")} | {
        "score": float(output["score"]),
        "components": {name: float(components[index]) for index, name in enumerate(COMPONENTS)},
        "normalized_components": {name: float(normalized[index]) for index, name in enumerate(COMPONENTS)},
        "top_patches": top_patches,
    }


def _score_batches(rows, model, args, bank, cache, description, progress=None, details=True, io_pool=None):
    device = next(model.buffers()).device
    own_pool = io_pool is None
    io_pool = io_pool or ThreadPoolExecutor(max_workers=args.workers)
    reader = partial(_load_observation, bank=bank, cache=cache,
                     foreground_minimum=args.foreground_minimum, expected_grid=model.grid_size)
    results = []
    local_progress = None
    if progress is None:
        local_progress = tqdm(total=len(rows), desc=description, unit="image", dynamic_ncols=True)
        progress = local_progress
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                batch_rows = rows[start:start + args.batch_size]
                observations = list(io_pool.map(reader, batch_rows))
                features = torch.from_numpy(np.stack([item[0] for item in observations])).to(
                    device, non_blocking=True)
                foreground = torch.from_numpy(np.stack([item[1] for item in observations])).to(
                    device, non_blocking=True)
                prediction = model(features, foreground)
                arrays = {key: value.detach().cpu().numpy() for key, value in prediction.items()}
                for index, row in enumerate(batch_rows):
                    sample = {key: value[index] for key, value in arrays.items()}
                    results.append(_result_row(row, sample, args.top_patch_count) if details else {
                        "sample_id": row["sample_id"], "role": row["role"], "group_id": row["group_id"],
                        "label": row["label"], "score": float(sample["score"]),
                        "components": {name: float(sample["components"][i])
                                       for i, name in enumerate(COMPONENTS)}})
                with progress.get_lock():
                    progress.update(len(batch_rows))
    finally:
        if local_progress is not None:
            local_progress.close()
        if own_pool:
            io_pool.shutdown()
    return results


def _load_model(model_dir, bank, args, device):
    metadata = read_json(model_dir / "model.json")
    if (metadata["experiment"] != args.experiment
            or metadata["plan_sha256"] != sha256(bank / "splits.json")
            or metadata["bank_config_sha256"] != sha256(bank / "bank_config.json")):
        raise ValueError("RealPatchBank checkpoint 不屬於目前的 feature-bank split")
    checkpoint = Path(metadata["checkpoint"])
    if sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("RealPatchBank checkpoint 已被修改")
    return RealPatchBank.load(checkpoint, metadata, device=device).eval(), metadata


def _score_parallel(rows, args, bank, cache, model_dir, description):
    devices = args.devices
    shards = [rows[index::len(devices)] for index in range(len(devices))]
    with ThreadPoolExecutor(max_workers=args.workers) as io_pool, tqdm(
            total=len(rows), desc=description, unit="image", dynamic_ncols=True, mininterval=.5) as progress:
        def lane(shard, device):
            if not shard:
                return []
            model, _ = _load_model(model_dir, bank, args, device)
            return _score_batches(shard, model, args, bank, cache, description,
                                  progress=progress, details=True, io_pool=io_pool)

        with ThreadPoolExecutor(max_workers=len(devices)) as device_pool:
            lanes = list(device_pool.map(lane, shards, devices))
    return sorted((row for lane_rows in lanes for row in lane_rows), key=lambda row: row["sample_id"])


def fit(plan, args, bank, cache, model_dir, output):
    checkpoint, metadata_path = model_dir / "model.safetensors", model_dir / "model.json"
    if (checkpoint.exists() or metadata_path.exists()) and not args.replace:
        raise ValueError(f"模型已存在：{model_dir}；若要重新建立請加 --replace")
    rows = load_bank_rows(plan, bank)
    fit_rows, validation_rows = _split_real_rows(rows, args.validation_fraction, args.seed)
    model, relation_counts = _fit_statistics(fit_rows, args, bank, cache)
    validation_device = args.devices[0]
    model = model.to(validation_device).eval()
    validation = _score_batches(validation_rows, model, args, bank, cache,
                                "Real-only fitting：held-out real 校準", details=False)
    raw = np.asarray([[row["components"][name] for name in COMPONENTS] for row in validation])
    centers = np.median(raw, axis=0)
    upper = np.quantile(raw, args.normalization_quantile, axis=0)
    scales = np.maximum(upper - centers, 1e-6)
    model.set_score_calibration(centers, scales)
    weights = np.asarray(args.score_weights, dtype=np.float64)
    normalized = (raw - centers) / scales
    if args.clip_scores:
        normalized = np.maximum(normalized, 0)
    combined = (normalized * weights).sum(1) / weights.sum()
    for row, score, components in zip(validation, combined, normalized):
        row["score"] = float(score)
        row["normalized_components"] = {
            name: float(components[index])
            for index, name in enumerate(COMPONENTS)}
    model = model.cpu()
    model.save(checkpoint)
    metadata = {
        "method": "real-only prototype / Mahalanobis / relation scoring",
        "experiment": args.experiment,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "grid_size": list(model.grid_size), "spatial_grid": list(model.spatial_grid),
        "spatial_restriction": args.spatial_restriction,
        "clip_scores": args.clip_scores,
        "input_dim": model.input_dim, "pca_dim": int(model.pca_components.shape[0]),
        "top_fraction": args.top_fraction, "score_components": list(COMPONENTS),
        "score_weights": args.score_weights, "score_centers": centers.tolist(),
        "score_scales": scales.tolist(), "normalization_quantile": args.normalization_quantile,
        "foreground_minimum": args.foreground_minimum,
        "neighbor_radius": args.neighbor_radius,
        "patch_samples_per_region": args.patch_samples_per_region,
        "prototypes_per_region": args.prototypes_per_region,
        "covariance_shrinkage": args.covariance_shrinkage,
        "covariance_ridge": args.covariance_ridge,
        "relation_std_floor": args.relation_std_floor,
        "minimum_relation_samples": args.minimum_relation_samples,
        "minimum_observed_relation_count": int(relation_counts[relation_counts > 0].min()),
        "fit_images": len(fit_rows), "validation_images": len(validation_rows),
        "original_bank_images": sum('augmentation' not in row for row in rows),
        "augmented_bank_images": sum('augmentation' in row for row in rows),
        "fit_augmented_images": sum('augmentation' in row for row in fit_rows),
        "validation_augmented_images": sum('augmentation' in row for row in validation_rows),
        "fit_families": sorted({row["group_id"] for row in fit_rows}),
        "validation_families": sorted({row["group_id"] for row in validation_rows}),
        "fake_training_images": 0,
        "plan_sha256": sha256(bank / "splits.json"),
        "bank_config_sha256": sha256(bank / "bank_config.json"),
    }
    write_json(metadata_path, metadata)
    write_json(output / "fit/validation_scores.json", validation)
    write_json(output / "fit/model.json", metadata)
    print(f"RealPatchBank fitting 完成：{len(fit_rows)} 張 real 建模，"
          f"{len(validation_rows)} 張 held-out real 校準\n模型：{checkpoint}", flush=True)
    return metadata


def evaluate(plan, args, bank, cache, model_dir, output):
    _, metadata = _load_model(model_dir, bank, args, "cpu")
    calibration_rows = image_records(plan["groups"]["calibration"], "calibration")
    evaluation_rows = image_records(plan["groups"]["evaluation"], "evaluation")
    if not calibration_rows or any(row["label"] for row in calibration_rows):
        raise ValueError("Calibration must contain held-out real images only")
    calibration = _score_parallel(calibration_rows, args, bank, cache, model_dir,
                                  "RealPatchBank calibration")
    tested = _score_parallel(evaluation_rows, args, bank, cache, model_dir,
                             "RealPatchBank evaluation")
    threshold = float(np.quantile([row["score"] for row in calibration], args.threshold_quantile))
    for row in calibration + tested:
        row["prediction"] = "anomaly" if row["score"] > threshold else "normal"
    component_thresholds = {
        name: float(np.quantile([row["components"][name] for row in calibration], args.threshold_quantile))
        for name in COMPONENTS}
    component_metrics = {
        name: metrics([dict(row, score=row["components"][name]) for row in tested], component_thresholds[name])
        for name in COMPONENTS}
    stage = output / "stage2"
    write_json(stage / "calibration_scores.json", calibration)
    write_json(stage / "evaluation_scores.json", tested)
    write_json(stage / "thresholds.json", {
        "threshold": threshold, "quantile": args.threshold_quantile,
        "calibration_count": len(calibration), "component_thresholds": component_thresholds,
        "model_sha256": metadata["checkpoint_sha256"],
    })
    report = {
        "protocol": plan["protocol"], "method": metadata["method"],
        "training": {"real_only": True, "fake_images": 0,
                     "fit_images": metadata["fit_images"],
                     "validation_images": metadata["validation_images"]},
        "image": metrics(tested, threshold), "components": component_metrics,
        "spatial_restriction": metadata.get("spatial_restriction", True),
        "clip_scores": metadata.get("clip_scores", True),
        "score_weights": metadata["score_weights"],
        "checkpoint": metadata["checkpoint"], "checkpoint_sha256": metadata["checkpoint_sha256"],
    }
    write_json(stage / "metrics.json", report)
    image = report["image"]
    print(f"RealPatchBank Stage 2 完成：AUROC={image['auroc']:.4f} "
          f"AP={image['average_precision']:.4f} FPR={image['false_positive_rate']:.4f} "
          f"TPR={image['true_positive_rate']:.4f}\n完整結果：{stage / 'metrics.json'}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    bank, cache, model_dir, output = _paths(args)
    if not (bank / "splits.json").is_file() or not (bank / "bank_config.json").is_file():
        raise ValueError("請先完成對應實驗的 Stage 1 feature bank")
    plan = read_json(bank / "splits.json")
    family_sets = [{video["group_id"] for role in roles if role in plan["groups"]
                    for video in plan["groups"][role]}
                   for roles in (("bank", "train_fake"), ("calibration",), ("evaluation",))]
    if any(family_sets[left] & family_sets[right]
           for left in range(3) for right in range(left)):
        raise ValueError("Source families overlap between training, calibration and evaluation")
    spec = _source_spec(bank)
    ensure_experiment_config(bank, Path(args.source_results_dir) / args.experiment, spec)
    with experiment_lock(output):
        if args.stage in ("fit", "all"):
            fit(plan, args, bank, cache, model_dir, output)
        if args.stage in ("evaluate", "all"):
            ensure_cache(plan, args, bank, cache, ("calibration", "evaluation"))
            evaluate(plan, args, bank, cache, model_dir, output)


if __name__ == "__main__":
    main()
