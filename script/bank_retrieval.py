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
    if (info["plan_sha256"] != sha256(bank / "splits.json")
            or info["source_config_sha256"] != sha256(bank / "bank_config.json")):
        raise ValueError("Retrieval index does not match source configuration/split")
    for item in info["files"].values():
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
                        "cosine_similarity": float(1 - distances[position]),
                        "distance": float(distances[position])})
    distance_map = np.full(grid, -1., dtype=np.float32)
    neighbor_map = np.full(grid, -1, dtype=np.int64)
    distance_map.reshape(-1)[ids] = distances
    neighbor_map.reshape(-1)[ids] = neighbors
    return {"score": float(scores[0]), "foreground_patches": len(ids), "scoring_patches": count,
            "matches": matches}, distance_map, neighbor_map


def evaluate(plan, args, bank, cache, output):
    info, sources, arrays = load_index(plan, bank, output)
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

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        def worker(shard, device):
            if not shard:
                return []
            search = PatchBank(arrays["features"], device, args.query_chunk_size, args.bank_chunk_size)
            result = []
            def read(row):
                return load_sample(cache / row["role"], row)[1]
            # Bound prefetched features rather than retaining the entire test set.
            for start in tqdm(range(0, len(shard), 32), desc=f"Stage 2 檢索 {device} ({len(shard)} images)",
                              unit="batch", dynamic_ncols=True):
                batch = shard[start:start + 32]
                for row, features in zip(batch, pool.map(read, batch)):
                    prediction, distances, neighbors = match_image(features, row["image_path"], search,
                                                                   info, sources, arrays, args.match_count)
                    folder = stage2 / row["role"] / "patch_matches"
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{row['sample_id']:08d}.npz"
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez(temporary, distances=distances, neighbors=neighbors)
                    temporary.replace(path)
                    result.append(dict(row, **prediction, patch_matches=str(path.resolve())))
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
    report = {"protocol": plan["protocol"], "method": info["method"],
              "bank_images": info["image_count"], "bank_patches": info["patch_count"],
              "image": metrics(tested, threshold)}
    write_json(stage2 / "metrics.json", report)
    print(report, flush=True)
