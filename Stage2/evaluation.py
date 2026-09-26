"""Stage 2 held-out patch retrieval, calibration, evaluation, and ablation."""

from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np
from tqdm.auto import tqdm

from Stage1.bank_builder import foreground_patches, load_index
from Stage2.bank_search import PatchBank, TopKPatchBank
from script.bank_data import image_records, sha256, write_json
from script.experiment_paths import stage1_output, stage2_output
from script.retrieval_io import load_sample, map_devices, metrics


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


def summarize_match(distances, neighbors, ids, grid, info, sources, arrays, match_count, details=None):
    """Convert patch search output into the persisted per-image result."""
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
    score = float(np.sort(distances)[-count:].mean())
    return {"score": score, "foreground_patches": len(ids), "scoring_patches": count,
            "matches": matches, **({"comparison_scores": details["scores"]} if details else {})}, distance_map, neighbor_map


def match_image(features, image_path, search, info, sources, arrays, match_count):
    selected, ids, grid = foreground_patches(features, image_path, info["foreground_minimum"])
    _, distances, neighbors = search.score(selected[None], info["top_fraction"])
    details = getattr(search, "details", None)
    if details:
        details = dict(details, scores={
            "nearest": float(np.sort(details["nearest_distances"])[
                -max(1, math.ceil(len(ids) * info["top_fraction"])):].mean()),
            "topk": float(np.sort(details["topk_distances"])[
                -max(1, math.ceil(len(ids) * info["top_fraction"])):].mean()),
        })
    return summarize_match(distances[0], neighbors[0], ids, grid, info, sources, arrays,
                           match_count, details)


def match_batch(items, search, info, sources, arrays, match_count):
    """Search several images together so exact retrieval fills query chunks."""
    prepared = [(row, *foreground_patches(features, row["image_path"], info["foreground_minimum"]))
                for row, features in items]
    merged = np.concatenate([selected for _, selected, _, _ in prepared])
    _, all_distances, all_neighbors = search.score(merged[None], info["top_fraction"])
    all_distances, all_neighbors = all_distances[0], all_neighbors[0]
    all_details = getattr(search, "details", None)
    results, offset = [], 0
    for row, selected, ids, grid in prepared:
        stop = offset + len(selected)
        details = None
        if all_details:
            details = {key: value[offset:stop] for key, value in all_details.items()
                       if key != "scores"}
            count = max(1, math.ceil(len(ids) * info["top_fraction"]))
            details["scores"] = {
                "nearest": float(np.sort(details["nearest_distances"])[-count:].mean()),
                "topk": float(np.sort(details["topk_distances"])[-count:].mean()),
            }
        prediction, distances, neighbors = summarize_match(
            all_distances[offset:stop], all_neighbors[offset:stop], ids, grid,
            info, sources, arrays, match_count, details)
        results.append((row, prediction, distances, neighbors, details or {}))
        offset = stop
    return results


def load_saved_match(path, row, features, info, sources, arrays, match_count, method):
    """Resume an atomically written match without repeating exhaustive retrieval."""
    _, ids, grid = foreground_patches(features, row["image_path"], info["foreground_minimum"])
    with np.load(path, allow_pickle=False) as stored:
        required = {"distances", "neighbors"}
        if not required <= set(stored.files):
            raise ValueError(f"Incomplete Stage 2 match: {path}")
        distance_map = stored["distances"]
        neighbor_map = stored["neighbors"]
        if distance_map.shape != grid or neighbor_map.shape != grid:
            raise ValueError(f"Stage 2 match grid changed: {path}")
        details = None
        if method == "topk":
            keys = {"candidate_ids", "weights", "nearest_distances", "topk_distances", "query_patch_ids"}
            if not keys <= set(stored.files) or not np.array_equal(stored["query_patch_ids"], ids):
                raise ValueError(f"Incomplete Stage 2 Top-K evidence: {path}")
            details = {key: stored[key] for key in keys if key != "query_patch_ids"}
            count = max(1, math.ceil(len(ids) * info["top_fraction"]))
            details["scores"] = {
                "nearest": float(np.sort(details["nearest_distances"])[-count:].mean()),
                "topk": float(np.sort(details["topk_distances"])[-count:].mean()),
            }
    prediction, _, _ = summarize_match(distance_map.reshape(-1)[ids], neighbor_map.reshape(-1)[ids],
                                       ids, grid, info, sources, arrays, match_count, details)
    return prediction


def write_evaluation_plot(calibrated, tested, threshold, destination):
    """Write ROC, precision-recall, and anomaly-score distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (average_precision_score, precision_recall_curve,
                                 roc_auc_score, roc_curve)

    labels = np.asarray([row["label"] for row in tested])
    scores = np.asarray([row["score"] for row in tested])
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].plot(fpr, tpr, label=f"AUROC={roc_auc_score(labels, scores):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate",
                xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP={average_precision_score(labels, scores):.4f}")
    axes[1].axhline(labels.mean(), linestyle="--", color="gray", linewidth=1,
                    label=f"prevalence={labels.mean():.4f}")
    axes[1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision",
                xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    axes[2].hist([scores[labels == 0], scores[labels == 1]], bins=60, density=True,
                 label=("evaluation real", "evaluation fake"), color=("#2070b4", "#d9483b"), alpha=.65)
    calibration_scores = np.asarray([row["score"] for row in calibrated])
    axes[2].hist(calibration_scores, bins=60, density=True, histtype="step", linewidth=1.5,
                 label="calibration real", color="black")
    axes[2].axvline(threshold, linestyle="--", color="black", label=f"threshold={threshold:.4f}")
    axes[2].set(title="Anomaly-score distribution", xlabel="Image anomaly score", ylabel="Density")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=.15)
    figure.savefig(destination / "evaluation_analysis.png", dpi=180)
    plt.close(figure)


def evaluate(plan, args, bank, cache, output):
    info, sources, arrays = load_index(plan, bank, output)
    method = getattr(args, "method", "nearest")
    stage1 = stage1_output(output)
    stage2 = stage2_output(output)
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
            total=len(arrays["features"]) * len(devices), desc="Stage 2：載入 FP16 bank",
            unit="patch", unit_scale=True, dynamic_ncols=True) as loading, tqdm(
            total=len(rows), desc="Stage 2 檢索", unit="image", dynamic_ncols=True, mininterval=.5) as progress:
        def loaded(count):
            with loading.get_lock():
                loading.update(count)

        def worker(shard, device):
            if not shard:
                return []
            if method == "nearest":
                search = PatchBank(arrays["features"], device, args.query_chunk_size, args.bank_chunk_size,
                                   progress=loaded)
            else:
                search = TopKPatchBank(arrays["features"], device, args.query_chunk_size,
                                       args.bank_chunk_size, args.topk, progress=loaded)
            result = []
            def read(row):
                return load_sample(cache / row["role"], row)[1]
            # Bound prefetched features rather than retaining the entire test set.
            for start in range(0, len(shard), 32):
                batch = shard[start:start + 32]
                features = list(pool.map(read, batch))
                pending = []
                for row, feature in zip(batch, features):
                    folder = stage2 / row["role"] / "patch_matches"
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / f"{row['sample_id']:08d}.npz"
                    if path.exists():
                        prediction = load_saved_match(path, row, feature, info, sources, arrays,
                                                      args.match_count, method)
                        result.append(dict(row, **prediction, patch_matches=str(path.resolve())))
                    else:
                        pending.append((row, feature))
                for row, prediction, distances, neighbors, details in match_batch(
                        pending, search, info, sources, arrays, args.match_count) if pending else ():
                    folder = stage2 / row["role"] / "patch_matches"
                    path = folder / f"{row['sample_id']:08d}.npz"
                    temporary = path.with_suffix(".tmp.npz")
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
               "calibration_count": len(calibrated), "retrieval_sha256": sha256(stage1 / "retrieval.json")})
    comparison = {}
    if method != "nearest":
        for name in tested[0]["comparison_scores"]:
            cutoff = float(np.quantile([r["comparison_scores"][name] for r in calibrated], args.threshold_quantile))
            comparison[name] = metrics([dict(r, score=r["comparison_scores"][name]) for r in tested], cutoff)
        write_json(stage2 / "comparison.json", dict(
            protocol="Development comparison on the existing evaluation split; not an untouched final test",
            methods=comparison, retrieval_sha256=sha256(stage1 / "retrieval.json")))
    report = {"protocol": plan["protocol"], "method": info["method"] if method == "nearest" else method,
              "bank_images": info["image_count"], "bank_patches": info["patch_count"],
              "image": metrics(tested, threshold), **({"comparison": comparison} if comparison else {})}
    write_json(stage2 / "metrics.json", report)
    write_evaluation_plot(calibrated, tested, threshold, stage2)
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
            reference = TopKPatchBank(arrays["features"], device, args.query_chunk_size,
                                      args.bank_chunk_size, args.topk, families=patch_families)
            result = []
            def read(row):
                return load_sample(cache / row["role"], row)[1]
            for start in range(0, len(shard), 32):
                batch = shard[start:start + 32]
                for row, features in zip(batch, pool.map(read, batch)):
                    selected, ids, grid = foreground_patches(features, row["image_path"], info["foreground_minimum"])
                    match_path = stage2_output(output) / row["role"] / "patch_matches" / f"{row['sample_id']:08d}.npz"
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
    directory = stage2_output(output) / "ablations/boundary_family"
    write_json(directory / "scores.json", results)
    write_json(directory / "report.json", {
        "protocol": "Fixed one-shot development ablation on the existing evaluation split; not final test",
        "settings": dict(ablation, candidate_scope=f"existing_top_{args.topk['neighbors']}"),
        "methods": methods, "thresholds": thresholds,
        "kept_candidates_per_patch": {"mean": float(kept.mean()), "minimum_image_mean": float(kept.min()),
                                      "maximum_image_mean": float(kept.max())},
        "retrieval_sha256": sha256(stage1_output(output) / "retrieval.json"),
        "source_count": len(sources), "bank_patches": int(len(arrays["features"])),
    })
    print("消融完成：" + " ".join(f"{name}={value['auroc']:.4f}" for name, value in methods.items())
          + f"\n完整結果：{directory / 'report.json'}", flush=True)
