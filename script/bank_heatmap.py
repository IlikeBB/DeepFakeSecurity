"""Patch-distance previews with one calibration-derived color scale."""

from pathlib import Path

import numpy as np
from PIL import Image


def save_heatmap(source, distances, destination, vmax, score, threshold):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    with Image.open(source) as image:
        face = image.convert("RGB")
    width, height = face.size
    figure = Figure(figsize=(9, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    extent = (0, width, height, 0)
    left.imshow(face, extent=extent)
    left.set_title("Input face")
    right.imshow(face, extent=extent)
    heat = right.imshow(distances, extent=extent, interpolation="nearest", alpha=.65,
                        cmap="magma", vmin=0, vmax=vmax)
    right.set_title("Patch anomaly distance (not probability)")
    for axis in (left, right):
        axis.set(xlabel="Width (pixels)", ylabel="Height (pixels)")
    figure.colorbar(heat, ax=right, shrink=.7)
    figure.suptitle(f"{Path(source).name}\nscore={score:.4f}; image threshold={threshold:.4f}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".jpg.tmp")
    figure.savefig(temporary, format="jpg", dpi=120)
    temporary.replace(destination)
    figure.clear()


def save_video_previews(records, directory, vmax, threshold):
    seen = set()
    for record in records:
        if record["video_id"] in seen:
            continue
        seen.add(record["video_id"])
        distances = np.load(directory / "scores" / f"distances_{record['sample_id']:08d}.npy", allow_pickle=False)
        save_heatmap(record["image_path"], distances, directory / "heatmaps" / record["video_id"] / "preview.jpg",
                     vmax, record["score"], threshold)
