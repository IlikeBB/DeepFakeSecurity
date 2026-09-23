"""Exact patch cosine nearest neighbors with bounded distance-matrix memory."""

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
