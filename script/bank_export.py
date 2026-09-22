"""Export real image patch arrays beside one Matplotlib preview per source folder."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from script.bank_data import image_records, read_json, save_array, write_json
from script.bank_encoder import encode, load_encoder


def load_feature(root, record):
    path = root / record["feature_path"]
    metadata = read_json(path.with_suffix(".json"))
    feature = np.load(path, allow_pickle=False)
    if (metadata["image"] != record or list(feature.shape) != metadata["shape"]
            or feature.ndim != 3 or str(feature.dtype) != metadata["dtype"]
            or not np.isfinite(feature).all()):
        raise ValueError(f"Invalid or stale feature: {path}")
    return feature


def save_preview(source, feature, projection, destination, quality):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with Image.open(source) as image:
        face = image.convert("RGB")
    width, height = face.size
    colors = (feature.astype(np.float32) - np.asarray(projection["mean"])) @ np.asarray(projection["components"])
    low, high = np.asarray(projection["low"]), np.asarray(projection["high"])
    rgb = np.clip((colors - low) / np.maximum(high - low, 1e-6), 0, 1)
    figure = Figure(figsize=(8, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for axis, data, title in zip(axes, (face, rgb),
                                 (f"Real face ({width} x {height})", "Patch features (PCA RGB)")):
        axis.imshow(data, extent=(0, width, height, 0), interpolation="nearest", aspect="equal")
        axis.set(title=title, xlabel="Width (pixels)", ylabel="Height (pixels)")
    figure.suptitle(Path(source).name)
    temporary = destination.with_suffix(".jpg.tmp")
    figure.savefig(temporary, format="jpg", dpi=150, pil_kwargs={"quality": quality, "subsampling": 0})
    temporary.replace(destination)
    figure.clear()


def build_bank(plan, args, root):
    records = image_records(plan["groups"]["bank"], "bank")
    if not records:
        raise ValueError("No real face images selected")
    previews = {}
    for record in records:
        if record["label"] != 0:
            raise ValueError("Only real images may enter the bank")
        source = Path(record["image_path"])
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"Crop changed since sampling: {source}")
        relative = source.relative_to(args.faces_dir).with_suffix(".npy")
        record["feature_path"] = relative.as_posix()
        previews.setdefault(relative.parent, record)
    model = processor = None
    for start in range(0, len(records), args.batch_size):
        pending = []
        for record in records[start:start + args.batch_size]:
            path = root / record["feature_path"]
            if path.with_suffix(".json").exists():
                load_feature(root, record)
            else:
                pending.append(record)
        if pending:
            if model is None:
                model, processor = load_encoder(args)
            _, features = encode([r["image_path"] for r in pending], model, processor, args)
            for record, feature in zip(pending, features):
                path = root / record["feature_path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                save_array(path, feature)
                write_json(path.with_suffix(".json"), {"image": record, "shape": list(feature.shape),
                                                       "dtype": str(feature.dtype)})
        print(f"[real NPY] {min(start + args.batch_size, len(records))}/{len(records)}", flush=True)
    del model, processor
    projection_path = root / "visualization.json"
    if not projection_path.exists():
        indices = np.linspace(0, len(records) - 1, min(args.pca_images, len(records)), dtype=int)
        samples = [load_feature(root, records[i]) for i in indices]
        values = torch.from_numpy(np.concatenate([f.reshape(-1, f.shape[-1]) for f in samples]).astype(np.float32))
        mean = values.mean(0)
        torch.manual_seed(args.seed)
        _, _, components = torch.pca_lowrank(values - mean, q=3, center=False, niter=4)
        colors = ((values - mean) @ components).numpy()
        write_json(projection_path, {"method": "shared real-bank PCA; display only",
                   "fit_image_paths": [records[i]["image_path"] for i in indices],
                   "mean": mean.tolist(), "components": components.tolist(),
                   "low": np.quantile(colors, .01, axis=0).tolist(),
                   "high": np.quantile(colors, .99, axis=0).tolist()})
    projection = read_json(projection_path)
    for directory, record in previews.items():
        destination = root / directory / "preview.jpg"
        if not destination.exists():
            save_preview(record["image_path"], load_feature(root, record), projection, destination, args.jpeg_quality)
    temporary = root / "manifest.jsonl.tmp"
    temporary.write_text("".join(json.dumps(record) + "\n" for record in records))
    temporary.replace(root / "manifest.jsonl")
    print(f"Saved {len(records)} NPY features and {len(previews)} JPG previews: {root}", flush=True)
