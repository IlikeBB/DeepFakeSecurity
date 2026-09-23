"""Real foreground patch retrieval with source-traceable cosine nearest neighbors."""

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from script.bank_data import image_records, read_json, save_array, sha256, write_json
from script.bank_search import PatchBank
from script.retrieval_io import load_sample, map_devices, metrics


def foreground_patches(features, image_path, minimum):
    """Approximate occupancy of already segmented, black-background images."""
    h, w, _ = features.shape
    with Image.open(image_path) as image:
        mask = (np.asarray(image.convert("RGB")).max(axis=-1) > 16).astype(np.uint8) * 255
    occupancy = np.asarray(Image.fromarray(mask).resize((w, h), Image.Resampling.BOX), dtype=np.float32) / 255
    ids = np.flatnonzero(occupancy.reshape(-1) >= minimum)
    if not len(ids):
        raise ValueError(f"No foreground patches: {image_path}")
    return features.reshape(-1, features.shape[-1])[ids], ids, (h, w)


def aggregate_patch_score(distances, ids, grid, fraction, boundary_weight=1.):
    """Average the largest weighted patch errors; boundary is the foreground-mask edge."""
    mask = np.zeros(grid, dtype=bool)
    mask.reshape(-1)[ids] = True
    padded = np.pad(mask, 1, constant_values=False)
    interior = (mask & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    weights = np.where(interior.reshape(-1)[ids], 1., boundary_weight)
    evidence = np.asarray(distances) * weights
    count = max(1, math.ceil(len(evidence) * fraction))
    return float(np.sort(evidence)[-count:].mean())


def build(plan, args, bank, output):
    stage1 = output / "stage1"
    stage1.mkdir(parents=True, exist_ok=True)
    completion = stage1 / "retrieval.json"
    if completion.exists():
        load_index(plan, bank, output)
        print("Stage 1 已完成，沿用 real patch bank 索引。", flush=True)
        return
    rows = image_records(plan["groups"]["bank"], "bank")
    if not rows or any(row["label"] != 0 for row in rows):
        raise ValueError("Retrieval bank must contain real training images only")

    def read(row):
        _, features = load_sample(bank, row)
        return foreground_patches(features, row["image_path"], args.foreground_minimum)

    vectors, origins, patches = [], [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (values, ids, grid) in enumerate(tqdm(pool.map(read, rows), total=len(rows),
                                                   desc="Stage 1：建立 real 檢索索引", unit="image")):
            rows[i]["grid"] = list(grid)
            vectors.append(values)
            origins.append(np.full(len(ids), i, dtype=np.int32))
            patches.append(ids.astype(np.int32))
    directory = Path(__file__).resolve().parents[1] / args.bank_dir / args.experiment / "retrieval"
    # 相對路徑固定以專案根目錄解析，避免從其他目錄啟動時寫到錯誤位置。
    directory.mkdir(parents=True, exist_ok=True)
    files = {"features": directory / "features.npy", "origins": directory / "origins.npy",
             "patch_ids": directory / "patch_ids.npy", "sources": stage1 / "sources.json"}
    for key, values in (("features", vectors), ("origins", origins), ("patch_ids", patches)):
        save_array(files[key], np.concatenate(values))
    write_json(files["sources"], rows)
    write_json(completion, {
        "method": "real foreground patches; exact cosine 1-NN; highest-distance patch mean",
        "image_count": len(rows), "patch_count": sum(len(values) for values in vectors),
        "foreground_minimum": args.foreground_minimum, "top_fraction": args.top_fraction,
        "plan_sha256": sha256(bank / "splits.json"),
        "source_config_sha256": sha256(bank / "bank_config.json"),
        "files": {key: {"path": str(path.resolve()), "sha256": sha256(path)} for key, path in files.items()},
    })


def load_index(plan, bank, output):
    info = read_json(output / "stage1/retrieval.json")
    if "alignment" in info:
        raise ValueError("此索引使用已移除的文字對齊分支；請另取 experiment 名稱，重新建立原始 DINO bank")
    if (info["plan_sha256"] != sha256(bank / "splits.json")
            or info["source_config_sha256"] != sha256(bank / "bank_config.json")):
        raise ValueError("Retrieval index does not match source configuration/split")
    for item in tqdm(info["files"].values(), desc="驗證 bank 檔案", unit="file", dynamic_ncols=True):
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Retrieval index changed: {item['path']}")
    rows = read_json(info["files"]["sources"]["path"])
    expected = image_records(plan["groups"]["bank"], "bank")
    if [{k: v for k, v in row.items() if k != "grid"} for row in rows] != expected:
        raise ValueError("Bank contains unexpected sources")
    arrays = {key: np.load(info["files"][key]["path"], mmap_mode="r", allow_pickle=False)
              for key in ("features", "origins", "patch_ids")}
    return info, rows, arrays


def match_image(features, image_path, search, info, sources, arrays, match_count):
    selected, ids, grid = foreground_patches(features, image_path, info["foreground_minimum"])
    scores, distances, neighbors = search.score(selected[None], info["top_fraction"])
    distances, neighbors = distances[0], neighbors[0]
    count = max(1, math.ceil(len(ids) * info["top_fraction"]))
    details = getattr(search, "details", None)
    scoring = np.argsort(-distances, kind="stable")[:count]
    matches = []
    # These are the patches contributing most to the image's anomaly score.
    for position in scoring[:match_count]:
        nearest = int(neighbors[position])
        source = sources[int(arrays["origins"][nearest])]
        source_patch = int(arrays["patch_ids"][nearest])
        matches.append({"query_patch": [int(ids[position] // grid[1]), int(ids[position] % grid[1])],
                        "bank_patch_id": nearest, "source_image": source["image_path"],
                        "source_video": source["video_id"], "source_family": source["group_id"],
                        "source_patch": [source_patch // source["grid"][1], source_patch % source["grid"][1]],
                        "cosine_similarity": float(1 - (details["nearest_distances"][position] if details else distances[position])),
                        "distance": float(details["nearest_distances"][position] if details else distances[position]),
                        "anomaly_distance": float(distances[position])})
        if details:
            matches[-1]["reference_patch_ids"] = details["candidate_ids"][position].tolist()
            matches[-1]["reference_weights"] = details["weights"][position].tolist()
    distance_map = np.full(grid, -1., dtype=np.float32)
    neighbor_map = np.full(grid, -1, dtype=np.int64)
    distance_map.reshape(-1)[ids] = distances
    neighbor_map.reshape(-1)[ids] = neighbors
    if details:
        details["query_patch_ids"] = ids
    return {"score": float(scores[0]), "foreground_patches": len(ids), "scoring_patches": count,
            "matches": matches, **({"comparison_scores": details["scores"]} if details else {})}, distance_map, neighbor_map


def evaluate(plan, args, bank, cache, output):
    info, sources, arrays = load_index(plan, bank, output)
    method = getattr(args, "method", "nearest")
    attention = None
    if method == "cross_attention":
        from script.bank_attention import attention_info
        attention = attention_info(output)
    stage2 = output / "stage2"
    stage2.mkdir(parents=True, exist_ok=True)
    bank_families = {row["group_id"] for row in sources}
    calibration = image_records(plan["groups"]["calibration"], "calibration")
    evaluation = image_records(plan["groups"]["evaluation"], "evaluation")
    if not calibration or any(row["label"] for row in calibration):
        raise ValueError("Calibration must contain only independent real images")
    if bank_families & {row["group_id"] for row in calibration + evaluation}:
        raise ValueError("Cannot score held-out images against their own source family")
    rows = calibration + evaluation
    devices = args.devices or ["cpu"]
    shards = [rows[i::len(devices)] for i in range(len(devices))]

    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(
            total=len(rows), desc="Stage 2 檢索", unit="image", dynamic_ncols=True, mininterval=.5) as progress:
        def worker(shard, device):
            if not shard:
                return []
            if method == "nearest":
                search = PatchBank(arrays["features"], device, args.query_chunk_size, args.bank_chunk_size)
            else:
                from script.bank_attention import ReferenceBank, load_attention
                model = load_attention(attention, device) if attention else None
                search = ReferenceBank(arrays["features"], device, args.query_chunk_size, args.bank_chunk_size,
                                       args.attention, model=model)
            result = []
            def read(row):
                return load_sample(cache / row["role"], row)[1]
            # Bound prefetched features rather than retaining the entire test set.
            for start in range(0, len(shard), 32):
                batch = shard[start:start + 32]
                for row, features in zip(batch, pool.map(read, batch)):
                    prediction, distances, neighbors = match_image(features, row["image_path"], search,
                                                                   info, sources, arrays, args.match_count)
                    folder = stage2 / row["role"] / "patch_matches"
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{row['sample_id']:08d}.npz"
                    temporary = path.with_suffix(".tmp.npz")
                    details = getattr(search, "details", {})
                    np.savez(temporary, distances=distances, neighbors=neighbors,
                             **{key: value for key, value in details.items() if key != "scores"})
                    temporary.replace(path)
                    result.append(dict(row, **prediction, patch_matches=str(path.resolve())))
                with progress.get_lock():
                    progress.update(len(batch))
            return result

        results = [row for shard in map_devices(worker, shards, devices) for row in shard]
    calibrated = sorted((r for r in results if r["role"] == "calibration"), key=lambda r: r["sample_id"])
    tested = sorted((r for r in results if r["role"] == "evaluation"), key=lambda r: r["sample_id"])
    threshold = float(np.quantile([row["score"] for row in calibrated], args.threshold_quantile))
    for row in calibrated + tested:
        row["prediction"] = "anomaly" if row["score"] > threshold else "normal"
    write_json(stage2 / "calibration_scores.json", calibrated)
    write_json(stage2 / "evaluation_scores.json", tested)
    write_json(stage2 / "thresholds.json", {"threshold": threshold, "quantile": args.threshold_quantile,
               "calibration_count": len(calibrated), "retrieval_sha256": sha256(output / "stage1/retrieval.json")})
    comparison = {}
    if method != "nearest":
        for name in tested[0]["comparison_scores"]:
            cutoff = float(np.quantile([r["comparison_scores"][name] for r in calibrated], args.threshold_quantile))
            comparison[name] = metrics([dict(r, score=r["comparison_scores"][name]) for r in tested], cutoff)
        write_json(stage2 / "comparison.json", dict(
            protocol="Development comparison on the existing evaluation split; not an untouched final test",
            methods=comparison, retrieval_sha256=sha256(output / "stage1/retrieval.json"),
            attention_sha256=sha256(output / "stage1/attention.json") if attention else None))
    report = {"protocol": plan["protocol"], "method": info["method"] if method == "nearest" else method,
              "bank_images": info["image_count"], "bank_patches": info["patch_count"],
              "image": metrics(tested, threshold), **({"comparison": comparison} if comparison else {})}
    write_json(stage2 / "metrics.json", report)
    result = report["image"]
    print(f"Stage 2 完成：AUROC={result['auroc']:.4f} AP={result['average_precision']:.4f} "
          f"FPR={result['false_positive_rate']:.4f} TPR={result['true_positive_rate']:.4f}\n"
          f"完整結果：{stage2 / 'metrics.json'}", flush=True)


def evaluate_ablation(plan, args, bank, cache, output):
    """Compare boundary weighting and source-family-diverse Top-K on the same queries."""
    info, sources, arrays = load_index(plan, bank, output)
    calibration = image_records(plan["groups"]["calibration"], "calibration")
    evaluation = image_records(plan["groups"]["evaluation"], "evaluation")
    if not calibration or any(row["label"] for row in calibration):
        raise ValueError("Calibration must contain only independent real images")
    rows = calibration + evaluation
    ablation = args.ablation
    family_names = {name: i for i, name in enumerate(sorted({row["group_id"] for row in sources}))}
    source_families = np.array([family_names[row["group_id"]] for row in sources])
    patch_families = source_families[np.asarray(arrays["origins"])]
    devices = args.devices or ["cpu"]
    shards = [rows[i::len(devices)] for i in range(len(devices))]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(
            total=len(rows), desc="邊界／家族多樣性消融", unit="image", dynamic_ncols=True, mininterval=.5) as progress:
        def worker(shard, device):
            from script.bank_attention import ReferenceBank
            reference = ReferenceBank(arrays["features"], device, args.query_chunk_size,
                                      args.bank_chunk_size, args.attention, families=patch_families)
            result = []
            def read(row):
                return load_sample(cache / row["role"], row)[1]
            for start in range(0, len(shard), 32):
                batch = shard[start:start + 32]
                for row, features in zip(batch, pool.map(read, batch)):
                    selected, ids, grid = foreground_patches(features, row["image_path"], info["foreground_minimum"])
                    match_path = output / "stage2" / row["role"] / "patch_matches" / f"{row['sample_id']:08d}.npz"
                    with np.load(match_path, allow_pickle=False) as matched:
                        if not {"candidate_ids", "query_patch_ids", "topk_distances"} <= set(matched.files):
                            raise ValueError(f"Stage 2 Top-K evidence missing: {match_path}")
                        if not np.array_equal(ids, matched["query_patch_ids"]):
                            raise ValueError(f"Stage 2 query patches changed: {match_path}")
                        candidate_ids = matched["candidate_ids"]
                        base_distance = matched["topk_distances"]
                    diverse_distance, keep = reference.score_candidates(
                        selected, candidate_ids, ablation["max_per_family"])
                    scores = {
                        "topk": aggregate_patch_score(base_distance, ids, grid, info["top_fraction"]),
                        "topk_boundary": aggregate_patch_score(base_distance, ids, grid, info["top_fraction"],
                                                               ablation["boundary_weight"]),
                        "family_cap_topk": aggregate_patch_score(diverse_distance, ids, grid, info["top_fraction"]),
                        "family_cap_topk_boundary": aggregate_patch_score(
                            diverse_distance, ids, grid, info["top_fraction"], ablation["boundary_weight"]),
                    }
                    result.append({key: row[key] for key in ("sample_id", "role", "video_id", "group_id", "label")} |
                                  {"scores": scores, "mean_kept_candidates": float(keep.sum(1).mean())})
                with progress.get_lock():
                    progress.update(len(batch))
            return result
        results = [row for shard in map_devices(worker, shards, devices) for row in shard]
    calibrated = [row for row in results if row["role"] == "calibration"]
    tested = [row for row in results if row["role"] == "evaluation"]
    methods, thresholds = {}, {}
    for name in tested[0]["scores"]:
        threshold = float(np.quantile([row["scores"][name] for row in calibrated], args.threshold_quantile))
        thresholds[name] = threshold
        methods[name] = metrics([dict(row, score=row["scores"][name]) for row in tested], threshold)
    kept = np.array([row["mean_kept_candidates"] for row in results])
    directory = output / "ablations/boundary_family"
    write_json(directory / "scores.json", results)
    write_json(directory / "report.json", {
        "protocol": "Fixed one-shot development ablation on the existing evaluation split; not final test",
        "settings": dict(ablation, candidate_scope=f"existing_top_{args.attention['neighbors']}"),
        "methods": methods, "thresholds": thresholds,
        "kept_candidates_per_patch": {"mean": float(kept.mean()), "minimum_image_mean": float(kept.min()),
                                      "maximum_image_mean": float(kept.max())},
        "retrieval_sha256": sha256(output / "stage1/retrieval.json"),
        "source_count": len(sources), "bank_patches": int(len(arrays["features"])),
    })
    print("消融完成：" + " ".join(f"{name}={value['auroc']:.4f}" for name, value in methods.items())
          + f"\n完整結果：{directory / 'report.json'}", flush=True)
