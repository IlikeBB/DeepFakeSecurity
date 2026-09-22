"""Exact patch cosine nearest neighbors with bounded distance-matrix memory."""

import math

import numpy as np
import torch
from torch.nn import functional as F


class PatchBank:
    def __init__(self, bank, device, query_chunk_size=1024, bank_chunk_size=16384):
        bank = np.asarray(bank, dtype=np.float32)
        if bank.ndim != 2 or not len(bank) or not np.isfinite(bank).all():
            raise ValueError("Bank must be a nonempty finite [patches, dimensions] matrix")
        if np.any(np.linalg.norm(bank, axis=1) == 0):
            raise ValueError("Bank contains zero-norm features")
        self.bank = F.normalize(torch.from_numpy(bank).to(device), dim=-1)
        self.query_chunk_size = query_chunk_size
        self.bank_chunk_size = bank_chunk_size

    @torch.inference_mode()
    def score(self, patches, top_fraction):
        patches = np.asarray(patches, dtype=np.float32)
        if patches.ndim != 3 or patches.shape[-1] != self.bank.shape[-1]:
            raise ValueError("Query must have shape [independent_images, patches, bank_dimensions]")
        if not np.isfinite(patches).all() or np.any(np.linalg.norm(patches, axis=-1) == 0):
            raise ValueError("Query contains invalid features")
        flat = patches.reshape(-1, patches.shape[-1])
        distances, neighbor_ids = [], []
        for start in range(0, len(flat), self.query_chunk_size):
            query = F.normalize(torch.from_numpy(flat[start:start + self.query_chunk_size]).to(self.bank.device), dim=-1)
            best = torch.full((len(query),), -float("inf"), device=query.device)
            nearest = torch.zeros(len(query), dtype=torch.long, device=query.device)
            for offset in range(0, len(self.bank), self.bank_chunk_size):
                similarities = query @ self.bank[offset:offset + self.bank_chunk_size].T
                values, ids = similarities.max(dim=1)
                better = values > best
                nearest = torch.where(better, ids + offset, nearest)
                best = torch.maximum(best, values)
            distances.append((1 - best).clamp(0, 2).cpu().numpy())
            neighbor_ids.append(nearest.cpu().numpy())
        patch_distances = np.concatenate(distances).reshape(patches.shape[:2])
        neighbors = np.concatenate(neighbor_ids).reshape(patches.shape[:2])
        count = max(1, math.ceil(patches.shape[1] * top_fraction))
        scores = np.sort(patch_distances, axis=1)[:, -count:].mean(axis=1)
        return scores, patch_distances, neighbors
