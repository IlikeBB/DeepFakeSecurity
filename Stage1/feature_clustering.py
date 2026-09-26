"""Post-hoc clustering diagnostic for a completed Stage 1 encoder."""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from tqdm.auto import tqdm

from Stage1.bank_builder import foreground_patches
from script.bank_data import image_records, save_array, sha256, write_json
from script.bank_encoder import encode, load_encoder
from script.experiment_paths import stage1_cluster_output
from script.retrieval_io import load_sample


def select_paired_samples(plan, count, seed):
    """Select one real/fake frame from each shared training source family."""
    real = image_records(plan["groups"]["bank"], "bank")
    fake = image_records(plan["groups"].get("train_fake", []), "train_fake")
    by_label = {}
    for label, rows in ((0, real), (1, fake)):
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
        by_label[label] = groups
    families = sorted(set(by_label[0]) & set(by_label[1]))
    if len(families) < 2:
        raise ValueError("分群診斷需要至少兩個同時具有 real 與 fake 的訓練來源家族")
    rng = random.Random(seed)
    rng.shuffle(families)
    families = families[:min(count, len(families))]
    rows = []
    for pair_id, family in enumerate(families):
        for label in (0, 1):
            row = dict(rng.choice(by_label[label][family]))
            row.update(analysis_id=len(rows), pair_id=pair_id)
            rows.append(row)
    return rows, len(set(by_label[0]) & set(by_label[1]))


def _pool(features, row, minimum):
    selected, _, _ = foreground_patches(features, row["image_path"], minimum)
    vector = selected.astype(np.float32).mean(axis=0)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Invalid pooled feature: {row['image_path']}")
    return vector / norm


def extract_embeddings(rows, args, bank):
    """Reuse real bank features and encode only the sampled fake frames."""
    real = [row for row in rows if row["label"] == 0]
    fake = [row for row in rows if row["label"] == 1]
    vectors = {}

    def load_real(row):
        source_row = {key: value for key, value in row.items() if key not in ("analysis_id", "pair_id")}
        return row["analysis_id"], _pool(load_sample(bank, source_row)[1], row, args.foreground_minimum)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sample_id, vector in tqdm(pool.map(load_real, real), total=len(real),
                                      desc="分群診斷：讀取 real 特徵", unit="image", dynamic_ncols=True):
            vectors[sample_id] = vector

    device = args.devices[0] if args.devices else "cpu"
    local = SimpleNamespace(**dict(vars(args), device=device, quiet=False))
    model, processor = load_encoder(local)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, tqdm(
            total=len(fake), desc="分群診斷：提取 fake 特徵", unit="image", dynamic_ncols=True) as progress:
        for start in range(0, len(fake), args.batch_size):
            batch = fake[start:start + args.batch_size]
            for row in batch:
                stat = Path(row["image_path"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                    raise ValueError(f"Source image changed: {row['image_path']}")
            _, patches = encode([row["image_path"] for row in batch], model, processor, local,
                                image_pool=pool)
            for row, feature in zip(batch, patches):
                vectors[row["analysis_id"]] = _pool(feature, row, args.foreground_minimum)
            progress.update(len(batch))
    del model, processor
    if device.startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
    return np.stack([vectors[index] for index in range(len(rows))])


def clustering_metrics(embeddings, labels, pair_ids, cluster_count, seed):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 silhouette_score)

    component_count = min(64, len(embeddings) - 1, embeddings.shape[1])
    reduced = PCA(n_components=component_count, random_state=seed).fit_transform(embeddings)
    count = min(cluster_count, len(embeddings) - 1)
    clusters = KMeans(n_clusters=count, n_init=20, random_state=seed).fit_predict(reduced)
    binary = KMeans(n_clusters=2, n_init=20, random_state=seed).fit_predict(reduced)
    aligned_accuracy = max(float((binary == labels).mean()), float((1 - binary == labels).mean()))
    composition = []
    for cluster in range(count):
        selected = labels[clusters == cluster]
        composition.append({"cluster": cluster, "count": int(len(selected)),
                            "real": int((selected == 0).sum()), "fake": int((selected == 1).sum()),
                            "fake_fraction": float(selected.mean())})

    real = {pair_id: embeddings[i] for i, pair_id in enumerate(pair_ids) if labels[i] == 0}
    fake = {pair_id: embeddings[i] for i, pair_id in enumerate(pair_ids) if labels[i] == 1}
    ordered = sorted(real)
    paired = np.array([np.dot(real[key], fake[key]) for key in ordered])
    shifted = np.array([np.dot(real[key], fake[ordered[(i + 1) % len(ordered)]])
                        for i, key in enumerate(ordered)])
    centroids = [embeddings[labels == label].mean(axis=0) for label in (0, 1)]
    centroid_similarity = float(np.dot(*centroids) /
                                (np.linalg.norm(centroids[0]) * np.linalg.norm(centroids[1])))
    metrics = {
        "sample_count": int(len(labels)),
        "samples_per_class": {"real": int((labels == 0).sum()), "fake": int((labels == 1).sum())},
        "embedding_dimension": int(embeddings.shape[1]),
        "pca_components_for_clustering": component_count,
        "pca_explained_variance": float(np.var(reduced, axis=0).sum() / np.var(embeddings, axis=0).sum()),
        "label_silhouette_cosine": float(silhouette_score(embeddings, labels, metric="cosine")),
        "kmeans_2": {
            "adjusted_rand_index": float(adjusted_rand_score(labels, binary)),
            "normalized_mutual_information": float(normalized_mutual_info_score(labels, binary)),
            "best_aligned_accuracy": aligned_accuracy,
        },
        "kmeans": {
            "cluster_count": count,
            "silhouette_euclidean": float(silhouette_score(reduced, clusters, metric="euclidean")),
            "composition": composition,
        },
        "cosine_similarity": {
            "real_fake_same_source_family_mean": float(paired.mean()),
            "real_fake_shifted_family_mean": float(shifted.mean()),
            "same_family_advantage": float(paired.mean() - shifted.mean()),
            "real_fake_centroid": centroid_similarity,
        },
    }
    return metrics, reduced[:, :2], clusters, binary


def _write_csv(path, rows, projection, clusters, binary):
    fields = ("analysis_id", "pair_id", "label", "cluster", "binary_cluster", "pca_x", "pca_y",
              "group_id", "video_id", "frame_index", "image_path")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, point, cluster, binary_cluster in zip(rows, projection, clusters, binary):
            writer.writerow({key: row[key] for key in ("analysis_id", "pair_id", "label", "group_id",
                                                       "video_id", "frame_index", "image_path")} |
                            {"cluster": int(cluster), "binary_cluster": int(binary_cluster),
                             "pca_x": float(point[0]), "pca_y": float(point[1])})


def _plot(output, projection, labels, clusters, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for label, name, color in ((0, "real", "#2070b4"), (1, "fake", "#d9483b")):
        selected = labels == label
        axes[0].scatter(projection[selected, 0], projection[selected, 1], s=13, alpha=.55,
                        label=name, color=color, edgecolors="none")
    axes[0].set_title("PCA projection by ground-truth label")
    axes[0].legend()
    axes[1].scatter(projection[:, 0], projection[:, 1], c=clusters, cmap="tab10", s=13,
                    alpha=.6, edgecolors="none")
    axes[1].set_title(f"PCA projection by K-Means (K={metrics['kmeans']['cluster_count']})")
    for axis in axes:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.grid(alpha=.15)
    fig.savefig(output / "pca_distribution.png", dpi=180)
    plt.close(fig)

    composition = metrics["kmeans"]["composition"]
    ids = np.array([row["cluster"] for row in composition])
    real = np.array([row["real"] for row in composition])
    fake = np.array([row["fake"] for row in composition])
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.bar(ids, real, label="real", color="#2070b4")
    axis.bar(ids, fake, bottom=real, label="fake", color="#d9483b")
    axis.set(title="Label composition of each K-Means cluster", xlabel="Cluster", ylabel="Images")
    axis.set_xticks(ids)
    axis.legend()
    axis.grid(axis="y", alpha=.15)
    fig.savefig(output / "cluster_composition.png", dpi=180)
    plt.close(fig)


def analyze(plan, args, bank, output):
    destination = stage1_cluster_output(output)
    destination.mkdir(parents=True, exist_ok=True)
    rows, available = select_paired_samples(plan, args.cluster_samples, args.seed)
    embeddings = extract_embeddings(rows, args, bank)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    pair_ids = np.asarray([row["pair_id"] for row in rows], dtype=np.int64)
    metrics, projection, clusters, binary = clustering_metrics(
        embeddings, labels, pair_ids, args.cluster_count, args.seed)
    metrics.update({
        "method": "paired-source-family foreground-patch mean pooling + PCA + K-Means",
        "purpose": "post-hoc Stage 1 representation diagnostic; no model update and not a final test metric",
        "available_paired_families": available,
        "sampled_paired_families": len(rows) // 2,
        "seed": args.seed,
        "encoder_checkpoint": getattr(args, "encoder_checkpoint", None),
        "encoder_checkpoint_sha256": (sha256(args.encoder_checkpoint)
                                      if getattr(args, "encoder_checkpoint", None) else None),
    })
    save_array(destination / "embeddings.npy", embeddings.astype(np.float32))
    _write_csv(destination / "projection.csv", rows, projection, clusters, binary)
    _plot(destination, projection, labels, clusters, metrics)
    write_json(destination / "metrics.json", metrics)
    kmeans_2 = metrics["kmeans_2"]
    (destination / "README.md").write_text(
        "# Stage 1 特徵分布診斷\n\n"
        "此結果從同一來源家族各抽一張 real 與 fake，使用目前 Stage 1 encoder 提取 patch token，"
        "只保留 SegFace 前景 patch 並做平均池化。PCA 僅供二維觀察；K-Means 在前 64 個 PCA "
        "分量上執行。分析不會更新模型或 Feature Bank，也不應作為最終 Stage 2 指標。\n\n"
        "## 本次數值\n\n"
        f"- 樣本：{metrics['samples_per_class']['real']} real + "
        f"{metrics['samples_per_class']['fake']} fake\n"
        f"- Label cosine silhouette：{metrics['label_silhouette_cosine']:.6f}\n"
        f"- K=2 ARI：{kmeans_2['adjusted_rand_index']:.6f}\n"
        f"- K=2 NMI：{kmeans_2['normalized_mutual_information']:.6f}\n"
        f"- K=2 最佳標籤對齊率：{kmeans_2['best_aligned_accuracy']:.4f}\n"
        f"- 同來源 real/fake cosine："
        f"{metrics['cosine_similarity']['real_fake_same_source_family_mean']:.6f}\n\n"
        "ARI、NMI 與 label silhouette 越接近 0，表示自然形成的群集與真假標籤關係越弱。\n\n"
        "- `metrics.json`：分群與真假分布統計。\n"
        "- `projection.csv`：每張圖的 PCA 座標、群集與來源。\n"
        "- `pca_distribution.png`：依真假標籤及 K-Means 群集著色。\n"
        "- `cluster_composition.png`：各群集的真假組成。\n"
        "- `embeddings.npy`：L2 正規化後的影像級表徵。\n",
        encoding="utf-8")
    print(f"Stage 1 分群診斷完成：{destination}", flush=True)
    return metrics
