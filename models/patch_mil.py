"""Patch-level relation-aware multiple-instance classifier."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PatchRelationMIL(nn.Module):
    """Classify a feature map while excluding masked background patches."""

    def __init__(self, input_dim, hidden_dim=256, heads=4, layers=2, dropout=.1,
                 top_fraction=.1, grid_size=(14, 14)):
        super().__init__()
        if hidden_dim % heads or not 0 < top_fraction <= 1:
            raise ValueError("Invalid PatchRelationMIL dimensions or top_fraction")
        self.grid_size = tuple(grid_size)
        self.top_fraction = float(top_fraction)
        patch_count = self.grid_size[0] * self.grid_size[1]
        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.positions = nn.Parameter(torch.zeros(1, patch_count, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.patch_head = nn.Linear(hidden_dim, 1)
        self.attention_head = nn.Linear(hidden_dim, 1)
        self.image_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 + 2),
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.trunc_normal_(self.positions, std=.02)

    @staticmethod
    def _relation_statistics(tokens, foreground):
        normalized = F.normalize(tokens, dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        pair_mask = foreground[:, :, None] & foreground[:, None, :]
        count = pair_mask.sum((1, 2)).clamp_min(1)
        mean = (similarity * pair_mask).sum((1, 2)) / count
        variance = (((similarity - mean[:, None, None]) ** 2) * pair_mask).sum((1, 2)) / count
        return mean, variance

    def forward(self, patches, foreground):
        if patches.ndim != 4 or tuple(patches.shape[1:3]) != self.grid_size:
            raise ValueError(f"Expected patches [B, {self.grid_size[0]}, {self.grid_size[1]}, D]")
        if foreground.shape != patches.shape[:3] or not foreground.flatten(1).any(1).all():
            raise ValueError("Every image needs at least one foreground patch")
        batch = len(patches)
        flat_mask = foreground.flatten(1).bool()
        flat = patches.flatten(1, 2)
        # Clear excluded inputs before normalization so background values cannot
        # affect valid tokens even when callers retain arbitrary pixel features.
        flat = flat.masked_fill(~flat_mask[..., None], 0)
        tokens = self.projection(self.input_norm(flat)) + self.positions
        tokens = self.encoder(tokens, src_key_padding_mask=~flat_mask)

        raw_patch_logits = self.patch_head(tokens).squeeze(-1)
        patch_logits = raw_patch_logits.masked_fill(~flat_mask, -torch.inf)
        attention_logits = self.attention_head(tokens).squeeze(-1).masked_fill(~flat_mask, -torch.inf)
        attention = torch.softmax(attention_logits, dim=1)
        attention_pool = (attention[..., None] * tokens).sum(1)
        global_pool = (tokens * flat_mask[..., None]).sum(1) / flat_mask.sum(1, keepdim=True)
        relation_mean, relation_variance = self._relation_statistics(tokens, flat_mask)
        context = torch.cat((attention_pool, global_pool,
                             relation_mean[:, None], relation_variance[:, None]), dim=1)
        image_logit = self.image_head(context).squeeze(-1)

        top_values = []
        for index in range(batch):
            valid = raw_patch_logits[index, flat_mask[index]]
            count = max(1, math.ceil(len(valid) * self.top_fraction))
            top_values.append(valid.topk(count).values.mean())
        image_logit = image_logit + torch.stack(top_values)
        return {
            "image_logit": image_logit,
            "patch_logits": patch_logits.reshape(batch, *self.grid_size),
            "attention": attention.reshape(batch, *self.grid_size),
            "relation_mean": relation_mean,
            "relation_variance": relation_variance,
        }
