"""Exact nearest-neighbor and weighted Top-K patch search."""

import math

import numpy as np
import torch
from torch.nn import functional as F


def _storage_dtype(bank, device):
    """Keep the large bank in FP16 on CUDA; CPU matmul uses FP32."""
    return torch.float16 if torch.device(device).type == "cuda" and bank.dtype == np.float16 else torch.float32


def required_device_bytes(bank, device, query_chunk_size=1024, bank_chunk_size=16384):
    """Conservative memory needed by one exhaustive-search worker."""
    dtype = _storage_dtype(bank, device)
    element_size = torch.empty((), dtype=dtype).element_size()
    stored = int(bank.size) * element_size
    similarities = int(query_chunk_size) * int(bank_chunk_size) * element_size
    normalization = int(bank_chunk_size) * int(bank.shape[1]) * (element_size + 4)
    return stored + similarities + normalization + 512 * 1024**2


def check_device_memory(bank, devices, query_chunk_size=1024, bank_chunk_size=16384):
    """Fail before feature extraction when a selected GPU cannot hold the bank."""
    failures = []
    for name in devices:
        device = torch.device(name)
        if device.type != "cuda":
            continue
        free, _ = torch.cuda.mem_get_info(device)
        required = required_device_bytes(bank, device, query_chunk_size, bank_chunk_size)
        if free < required:
            failures.append(f"{name} 可用 {free / 1024**3:.1f} GiB、至少需要 {required / 1024**3:.1f} GiB")
    if failures:
        raise RuntimeError(
            "Stage 2 GPU 顯存不足；每張指定 GPU 都要保存一份完整 FP16 feature bank："
            + "；".join(failures)
            + "。請等待 GPU 空出，或只用顯存足夠的 GPU。"
        )


class PatchBank:
    def __init__(self, bank, device, query_chunk_size=1024, bank_chunk_size=16384, progress=None):
        bank = np.asarray(bank)
        if bank.ndim != 2 or not len(bank) or bank.dtype not in (np.float16, np.float32):
            raise ValueError("Bank must be a nonempty float16/float32 [patches, dimensions] matrix")
        check_device_memory(bank, [device], query_chunk_size, bank_chunk_size)
        target = torch.device(device)
        dtype = _storage_dtype(bank, target)
        self.bank = torch.empty(bank.shape, dtype=dtype, device=target)
        # Normalize in bounded chunks.  The previous whole-bank FP32 conversion
        # allocated about 18.5 GB per worker before the GPU copy even started.
        for offset in range(0, len(bank), bank_chunk_size):
            source = torch.tensor(bank[offset:offset + bank_chunk_size], device=target)
            norms = torch.linalg.vector_norm(source, dim=-1, keepdim=True, dtype=torch.float32)
            if not torch.isfinite(source).all() or not torch.isfinite(norms).all() or torch.any(norms == 0):
                raise ValueError("Bank contains non-finite or zero-norm features")
            self.bank[offset:offset + len(source)].copy_((source / norms).to(dtype))
            if progress is not None:
                progress(len(source))
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
            query = torch.from_numpy(flat[start:start + self.query_chunk_size]).to(
                self.bank.device, dtype=self.bank.dtype)
            query = F.normalize(query, dim=-1)
            best = torch.full((len(query),), -float("inf"), device=query.device, dtype=self.bank.dtype)
            nearest = torch.zeros(len(query), dtype=torch.long, device=query.device)
            for offset in range(0, len(self.bank), self.bank_chunk_size):
                similarities = query @ self.bank[offset:offset + self.bank_chunk_size].T
                values, ids = similarities.max(dim=1)
                better = values > best
                nearest = torch.where(better, ids + offset, nearest)
                best = torch.maximum(best, values)
            distances.append((1 - best).clamp(0, 2).float().cpu().numpy())
            neighbor_ids.append(nearest.cpu().numpy())
        patch_distances = np.concatenate(distances).reshape(patches.shape[:2])
        neighbors = np.concatenate(neighbor_ids).reshape(patches.shape[:2])
        count = max(1, math.ceil(patches.shape[1] * top_fraction))
        scores = np.sort(patch_distances, axis=1)[:, -count:].mean(axis=1)
        return scores, patch_distances, neighbors


class TopKPatchBank(PatchBank):
    """Reconstruct each query patch from its closest normal-bank references."""

    def __init__(self, bank, device, query_chunk_size, bank_chunk_size, config, families=None, progress=None):
        super().__init__(bank, device, query_chunk_size, bank_chunk_size, progress=progress)
        self.config = config
        self.families = None if families is None else torch.as_tensor(families, device=device)

    @torch.no_grad()
    def nearest(self, query):
        query = F.normalize(torch.as_tensor(query, dtype=self.bank.dtype, device=self.bank.device), dim=-1)
        if query.ndim != 2 or query.shape[1] != self.bank.shape[1] or not torch.isfinite(query).all():
            raise ValueError("Invalid retrieval query")
        if torch.any(query.norm(dim=-1) == 0):
            raise ValueError("Query contains zero-norm features")
        count = self.config["neighbors"]
        if count > len(self.bank):
            raise ValueError("Top-K exceeds reference bank size")
        all_values, all_ids = [], []
        for start in range(0, len(query), self.query_chunk_size):
            batch = query[start:start + self.query_chunk_size]
            best = torch.empty((len(batch), 0), device=batch.device, dtype=self.bank.dtype)
            ids = torch.empty((len(batch), 0), device=batch.device, dtype=torch.long)
            for offset in range(0, len(self.bank), self.bank_chunk_size):
                similarities = batch @ self.bank[offset:offset + self.bank_chunk_size].T
                values, local = similarities.topk(min(count, similarities.shape[1]), dim=1)
                combined = torch.cat((best, values), dim=1)
                candidates = torch.cat((ids, local + offset), dim=1)
                best, positions = combined.topk(min(count, combined.shape[1]), dim=1)
                ids = candidates.gather(1, positions)
            all_values.append(best)
            all_ids.append(ids)
        return torch.cat(all_values).float(), torch.cat(all_ids)

    @torch.inference_mode()
    def score_candidates(self, query, candidate_ids, max_per_family):
        """Reweight an existing Top-K list after limiting repeated source families."""
        if self.families is None:
            raise ValueError("Family IDs required for capped candidate scoring")
        query = F.normalize(torch.as_tensor(query, dtype=torch.float32, device=self.bank.device), dim=-1)
        ids = torch.as_tensor(candidate_ids, device=self.bank.device)
        candidates = self.bank[ids].float()
        similarities = (query[:, None] * candidates).sum(-1)
        families = self.families[ids]
        keep = torch.ones_like(similarities, dtype=torch.bool)
        for position in range(1, ids.shape[1]):
            keep[:, position] = (families[:, :position] == families[:, position, None]).sum(1) < max_per_family
        weights = (similarities / self.config["temperature"]).masked_fill(~keep, -torch.inf).softmax(-1)
        reconstructed = torch.einsum("bk,bkd->bd", weights, candidates)
        distances = (1 - F.cosine_similarity(query, reconstructed)).clamp(0, 2)
        return distances.cpu().numpy(), keep.cpu().numpy()

    @torch.inference_mode()
    def score(self, patches, top_fraction):
        if patches.ndim != 3 or patches.shape[0] != 1:
            raise ValueError("Top-K scoring expects one image at a time")
        query = F.normalize(torch.as_tensor(patches[0], dtype=torch.float32, device=self.bank.device), dim=-1)
        similarities, ids = self.nearest(query)
        candidates = self.bank[ids].float()
        weights = (similarities / self.config["temperature"]).softmax(-1)
        reconstructed = torch.einsum("bk,bkd->bd", weights, candidates)
        nearest_distances = (1 - similarities[:, 0]).clamp(0, 2)
        distances = (1 - F.cosine_similarity(query, reconstructed)).clamp(0, 2)
        nearest_distances = nearest_distances.cpu().numpy()
        distances = distances.cpu().numpy()
        count = max(1, math.ceil(len(query) * top_fraction))
        scores = {
            "nearest": float(np.sort(nearest_distances)[-count:].mean()),
            "topk": float(np.sort(distances)[-count:].mean()),
        }
        self.details = {
            "candidate_ids": ids.cpu().numpy(),
            "weights": weights.cpu().numpy(),
            "nearest_distances": nearest_distances,
            "topk_distances": distances,
            "scores": scores,
        }
        return np.array([scores["topk"]]), distances[None], ids[:, 0].cpu().numpy()[None]
