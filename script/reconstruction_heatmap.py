"""Render reconstruction evidence at a fixed cosine-error scale (not a probability)."""
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import PowerNorm


def save_heatmap(row, destination):
    errors = np.full(np.prod(row['grid']), np.nan, dtype=np.float32)
    errors[row['patch_ids']] = row['patch_errors']
    errors = errors.reshape(row['grid'])
    with Image.open(row['image_path']) as image:
        face = np.asarray(image.convert('RGB'))
    h, w = face.shape[:2]
    figure = Figure(figsize=(10, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    norm = PowerNorm(gamma=.5, vmin=0, vmax=2)
    axes[0].imshow(face)
    axes[0].set_title('SegFace crop')
    heat = axes[1].imshow(errors, norm=norm, cmap='inferno', interpolation='nearest')
    axes[1].set_title('Raw reconstruction error')
    axes[2].imshow(face)
    axes[2].imshow(errors, norm=norm, cmap='inferno', alpha=.55,
                   extent=(0, w, h, 0), interpolation='nearest')
    axes[2].set_title('Patch evidence overlay')
    for axis in axes:
        axis.axis('off')
    figure.colorbar(heat, ax=list(axes), label='Cosine error (sqrt color scale); not probability', shrink=.8)
    figure.suptitle(Path(row['image_path']).name)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    figure.clear()


def save_evidence_heatmap(row, maps, destination):
    """Render NN, reconstruction, and fused calibration-relative evidence."""
    with Image.open(row['image_path']) as image:
        face = np.asarray(image.convert('RGB'))
    height, width = face.shape[:2]
    figure = Figure(figsize=(13, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 4)
    norm = PowerNorm(gamma=.5, vmin=0, vmax=2)
    axes[0].imshow(face)
    axes[0].set_title('SegFace crop')
    nearest = maps.get('nearest')
    if nearest is None:
        axes[1].text(.5, .5, 'No external NN bank', ha='center', va='center')
    else:
        axes[1].imshow(nearest, norm=norm, cmap='inferno', interpolation='nearest')
    axes[1].set_title('Normal-bank evidence')
    axes[2].imshow(maps['reconstruction'], norm=norm, cmap='inferno', interpolation='nearest')
    axes[2].set_title('Neighbor prediction evidence')
    axes[3].imshow(face)
    overlay = axes[3].imshow(maps['fusion'], norm=norm, cmap='inferno', alpha=.55,
                             extent=(0, width, height, 0), interpolation='nearest')
    axes[3].set_title('Fused evidence overlay')
    for axis in axes:
        axis.axis('off')
    figure.colorbar(overlay, ax=list(axes), shrink=.8,
                    label='Calibration-relative evidence (1 ≈ real-patch q99)')
    figure.suptitle(Path(row['image_path']).name)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    figure.clear()
