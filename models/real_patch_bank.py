"""Real-only position-aware patch anomaly model."""

import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F


COMPONENTS = ("nearest", "mahalanobis", "relation")


def spatial_region_ids(grid_size, spatial_grid):
    """Map every input patch to a coarse, position-aware facial region."""
    height, width = (int(value) for value in grid_size)
    region_height, region_width = (int(value) for value in spatial_grid)
    if min(height, width, region_height, region_width) < 1:
        raise ValueError("Grid dimensions must be positive")
    rows = torch.div(torch.arange(height) * region_height, height, rounding_mode="floor")
    cols = torch.div(torch.arange(width) * region_width, width, rounding_mode="floor")
    return (rows[:, None] * region_width + cols[None, :]).flatten()


class RealPatchBank(nn.Module):
    """Score deviations from real patch prototypes, distributions and relations.

    All tensors are fixed statistics estimated from real images.  This module
    has no trainable parameters and accepts cached patch maps in ``[B,H,W,D]``.
    """

    def __init__(self, *, prototypes, prototype_counts, pca_mean, pca_components,
                 gaussian_means, gaussian_precisions, gaussian_valid,
                 relation_means, relation_stds, relation_valid,
                 score_centers, score_scales, score_weights,
                 grid_size, spatial_grid, top_fraction, spatial_restriction=True, clip_scores=True):
        super().__init__()
        self.grid_size = tuple(int(value) for value in grid_size)
        self.spatial_grid = tuple(int(value) for value in spatial_grid)
        self.top_fraction = float(top_fraction)
        self.spatial_restriction = bool(spatial_restriction)
        self.clip_scores = bool(clip_scores)
        if not 0 < self.top_fraction <= 1:
            raise ValueError("top_fraction must be in (0, 1]")
        expected_regions = self.spatial_grid[0] * self.spatial_grid[1]
        tensors = {
            "prototypes": prototypes, "prototype_counts": prototype_counts,
            "pca_mean": pca_mean, "pca_components": pca_components,
            "gaussian_means": gaussian_means, "gaussian_precisions": gaussian_precisions,
            "gaussian_valid": gaussian_valid, "relation_means": relation_means,
            "relation_stds": relation_stds, "relation_valid": relation_valid,
            "score_centers": score_centers, "score_scales": score_scales,
            "score_weights": score_weights,
        }
        tensors = {name: torch.as_tensor(value) for name, value in tensors.items()}
        if tensors["prototypes"].ndim != 3 or tensors["prototypes"].shape[0] != expected_regions:
            raise ValueError("prototypes must be [regions, prototypes, dimensions]")
        if tensors["prototype_counts"].shape != (expected_regions,):
            raise ValueError("prototype_counts must contain one value per region")
        if tensors["pca_components"].shape[1] != tensors["prototypes"].shape[2]:
            raise ValueError("PCA and prototype dimensions differ")
        projected_dim = tensors["pca_components"].shape[0]
        if tensors["gaussian_means"].shape != (expected_regions, projected_dim):
            raise ValueError("Invalid Gaussian means")
        if tensors["gaussian_precisions"].shape != (expected_regions, projected_dim, projected_dim):
            raise ValueError("Invalid Gaussian precisions")
        if tensors["relation_means"].shape != (expected_regions, expected_regions):
            raise ValueError("Invalid relation statistics")
        if any(tensors[name].numel() != 3 for name in ("score_centers", "score_scales", "score_weights")):
            raise ValueError("Score calibration requires three components")
        if torch.any(tensors["score_scales"] <= 0) or torch.any(tensors["score_weights"] < 0):
            raise ValueError("Score scales must be positive and weights non-negative")
        if tensors["score_weights"].sum() <= 0:
            raise ValueError("At least one score weight must be positive")

        float_names = {"prototypes", "pca_mean", "pca_components", "gaussian_means",
                       "gaussian_precisions", "relation_means", "relation_stds",
                       "score_centers", "score_scales", "score_weights"}
        bool_names = {"gaussian_valid", "relation_valid"}
        for name, value in tensors.items():
            if name in float_names:
                value = value.float()
            elif name in bool_names:
                value = value.bool()
            else:
                value = value.long()
            self.register_buffer(name, value.contiguous())
        self.register_buffer("region_ids", spatial_region_ids(self.grid_size, self.spatial_grid))

    @property
    def input_dim(self):
        return int(self.prototypes.shape[-1])

    def set_score_calibration(self, centers, scales):
        centers = torch.as_tensor(centers, dtype=self.score_centers.dtype,
                                  device=self.score_centers.device)
        scales = torch.as_tensor(scales, dtype=self.score_scales.dtype,
                                 device=self.score_scales.device)
        if centers.shape != (3,) or scales.shape != (3,) or torch.any(scales <= 0):
            raise ValueError("Calibration centers/scales must be positive three-vectors")
        self.score_centers.copy_(centers)
        self.score_scales.copy_(scales)

    def _top_fraction_mean(self, values, valid, allow_empty=False):
        result = []
        for image in range(len(values)):
            selected = values[image, valid[image]]
            if not len(selected):
                if allow_empty:
                    result.append(values.new_zeros(()))
                    continue
                raise ValueError("Every image must contain valid evidence")
            count = max(1, math.ceil(len(selected) * self.top_fraction))
            result.append(selected.topk(count).values.mean())
        return torch.stack(result)

    def _patch_scores(self, normalized):
        batch, patches, _ = normalized.shape
        nearest = normalized.new_zeros((batch, patches))
        projected = (normalized - self.pca_mean) @ self.pca_components.T
        mahalanobis = normalized.new_zeros((batch, patches))
        if not self.spatial_restriction:
            reference = torch.cat([self.prototypes[i, :int(count)]
                                   for i, count in enumerate(self.prototype_counts) if count > 0])
            nearest = (1 - (normalized @ reference.T).max(-1).values).clamp(0, 2)
        for region in range(len(self.prototype_counts)):
            positions = self.region_ids == region
            count = int(self.prototype_counts[region])
            if count and self.spatial_restriction:
                reference = self.prototypes[region, :count]
                similarity = normalized[:, positions] @ reference.T
                nearest[:, positions] = (1 - similarity.max(-1).values).clamp(0, 2)
            if bool(self.gaussian_valid[region]):
                difference = projected[:, positions] - self.gaussian_means[region]
                squared = torch.einsum(
                    "bnd,de,bne->bn", difference, self.gaussian_precisions[region], difference)
                mahalanobis[:, positions] = squared.clamp_min(0).sqrt()
        return nearest, mahalanobis

    def _relation_score(self, normalized, foreground):
        batch = len(normalized)
        regions = len(self.prototype_counts)
        pooled = normalized.new_zeros((batch, regions, self.input_dim))
        region_mask = torch.zeros((batch, regions), dtype=torch.bool, device=normalized.device)
        for region in range(regions):
            positions = self.region_ids == region
            weights = foreground[:, positions]
            count = weights.sum(1)
            valid = count > 0
            region_mask[:, region] = valid
            pooled[:, region] = ((normalized[:, positions] * weights[..., None]).sum(1)
                                 / count.clamp_min(1)[..., None])
        pooled = F.normalize(pooled, dim=-1)
        similarity = pooled @ pooled.transpose(1, 2)
        pair_mask = region_mask[:, :, None] & region_mask[:, None, :] & self.relation_valid
        triangle = torch.triu(torch.ones_like(self.relation_valid), diagonal=1)
        pair_mask &= triangle
        deviations = (similarity - self.relation_means).abs() / self.relation_stds.clamp_min(1e-6)
        return self._top_fraction_mean(deviations, pair_mask, allow_empty=True)

    def forward(self, features, foreground):
        if features.ndim != 4 or tuple(features.shape[1:3]) != self.grid_size:
            raise ValueError(f"Expected features [B, {self.grid_size[0]}, {self.grid_size[1]}, D]")
        if features.shape[-1] != self.input_dim or foreground.shape != features.shape[:3]:
            raise ValueError("Feature dimensions or foreground mask do not match the model")
        flat_mask = foreground.flatten(1).bool()
        if not flat_mask.any(1).all():
            raise ValueError("Every image requires at least one foreground patch")
        flat = features.flatten(1, 2).masked_fill(~flat_mask[..., None], 0)
        normalized = F.normalize(flat, dim=-1)
        nearest_map, mahalanobis_map = self._patch_scores(normalized)
        nearest = self._top_fraction_mean(nearest_map, flat_mask)
        mahalanobis = self._top_fraction_mean(mahalanobis_map, flat_mask)
        relation = self._relation_score(normalized, flat_mask)
        raw = torch.stack((nearest, mahalanobis, relation), dim=1)
        normalized_scores = (raw - self.score_centers) / self.score_scales
        if self.clip_scores:
            normalized_scores = normalized_scores.clamp_min(0)
        image_score = (normalized_scores * self.score_weights).sum(1) / self.score_weights.sum()
        local_weights = self.score_weights[:2]
        local = (((torch.stack((nearest_map, mahalanobis_map), dim=-1) - self.score_centers[:2])
                  / self.score_scales[:2]).clamp_min(0) * local_weights).sum(-1)
        local = local / local_weights.sum().clamp_min(1e-6)
        background = normalized.new_full(local.shape, -1)
        return {
            "score": image_score,
            "components": raw,
            "normalized_components": normalized_scores,
            "nearest_map": torch.where(flat_mask, nearest_map, background).reshape(len(features), *self.grid_size),
            "mahalanobis_map": torch.where(flat_mask, mahalanobis_map, background).reshape(
                len(features), *self.grid_size),
            "anomaly_map": torch.where(flat_mask, local, background).reshape(len(features), *self.grid_size),
        }

    def checkpoint_tensors(self):
        return {name: value.detach().cpu().contiguous() for name, value in self.named_buffers()
                if name != "region_ids"}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(self.checkpoint_tensors(), str(path))

    @classmethod
    def load(cls, path, metadata, device="cpu"):
        tensors = load_file(str(path), device=device)
        return cls(**tensors, grid_size=metadata["grid_size"], spatial_grid=metadata["spatial_grid"],
                   top_fraction=metadata["top_fraction"],
                   spatial_restriction=metadata.get("spatial_restriction", True),
                   clip_scores=metadata.get("clip_scores", True)).to(device)
