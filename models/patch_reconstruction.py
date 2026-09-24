"""Center-blind local Transformer feature prediction, trained on real images only."""
import torch
from torch import nn
from torch.nn import functional as F


class PatchReconstruction(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, radius=1, heads=4, layers=2):
        super().__init__()
        self.radius = radius
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        neighbors = (2 * radius + 1) ** 2 - 1
        self.relative_position = nn.Parameter(torch.randn(1, neighbors, hidden_dim) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * .02)
        layer = nn.TransformerDecoderLayer(hidden_dim, heads, hidden_dim * 2, dropout=0.,
                                           batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(hidden_dim))
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, features, foreground):
        # Tokens are independently projected before gathering. No center-token skip path.
        b, h, w, _ = features.shape
        z = self.project(F.normalize(features, dim=-1)) * foreground[..., None]
        z = torch.cat((z, foreground[..., None].to(z.dtype)), -1).permute(0, 3, 1, 2)
        r = self.radius
        patches = F.unfold(z, 2 * r + 1, padding=r)
        patches = patches.reshape(b, z.shape[1], (2 * r + 1) ** 2, h * w)
        center = (2 * r + 1) ** 2 // 2
        context = torch.cat((patches[:, :, :center], patches[:, :, center + 1:]), 2)
        context = context.permute(0, 3, 2, 1).reshape(b * h * w, -1, z.shape[1])
        neighbor_valid = context[..., -1] > 0
        has_context = neighbor_valid.any(1)
        # A harmless zero placeholder avoids all-masked softmax; these targets are omitted.
        neighbor_valid = neighbor_valid.clone()
        neighbor_valid[~has_context, 0] = True
        memory = context[..., :-1] + self.relative_position
        predicted = self.decoder(self.query.expand(b * h * w, -1, -1), memory,
                                 memory_key_padding_mask=~neighbor_valid)
        prediction = self.output(predicted[:, 0]).reshape(b, h, w, -1)
        valid = foreground & has_context.reshape(b, h, w)
        error = (1 - F.cosine_similarity(prediction, features, dim=-1)).clamp(0, 2)
        return prediction, error, valid
