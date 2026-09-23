"""Top-K normal references and constrained multi-head cross-attention.

Only Q/K projections are learned. Values stay in the frozen DINO space and
heads are averaged, so reconstruction remains a convex mixture of real patches.
"""

import math
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from script.bank_data import read_json, sha256, write_json
from script.bank_search import PatchBank
from script.retrieval_io import map_devices


class CrossAttention(nn.Module):
    def __init__(self, dimensions, hidden, heads):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.query = nn.Linear(dimensions, hidden, bias=False)
        self.key = nn.Linear(dimensions, hidden, bias=False)

    def forward(self, query, candidates):
        q = self.query(query).reshape(len(query), self.heads, self.width)
        k = self.key(candidates).reshape(len(query), candidates.shape[1], self.heads, self.width)
        logits = torch.einsum("bhd,bkhd->bhk", q, k) / math.sqrt(self.width)
        weights = logits.softmax(-1).mean(1)
        # No query residual, learned V or output decoder that could copy the query.
        return torch.einsum("bk,bkd->bd", weights, candidates), weights


class ReferenceBank(PatchBank):
    def __init__(self, bank, device, query_chunk_size, bank_chunk_size, config, model=None, families=None,
                 progress=None):
        super().__init__(bank, device, query_chunk_size, bank_chunk_size, progress=progress)
        self.config, self.model = config, model
        self.families = None if families is None else torch.as_tensor(families, device=device)

    @torch.no_grad()
    def nearest(self, query, excluded=None):
        """Exact chunked Top-K; excluded gives each query's source-family ID."""
        query = F.normalize(torch.as_tensor(query, dtype=self.bank.dtype, device=self.bank.device), dim=-1)
        if query.ndim != 2 or query.shape[1] != self.bank.shape[1] or not torch.isfinite(query).all():
            raise ValueError("Invalid retrieval query")
        if torch.any(query.norm(dim=-1) == 0):
            raise ValueError("Query contains zero-norm features")
        k = self.config["neighbors"]
        if k > len(self.bank):
            raise ValueError("Top-K exceeds reference bank size")
        if excluded is not None:
            if self.families is None:
                raise ValueError("Family IDs required for excluded retrieval")
            excluded = torch.as_tensor(excluded, device=self.bank.device)
        all_values, all_ids = [], []
        for start in range(0, len(query), self.query_chunk_size):
            q = query[start:start + self.query_chunk_size]
            best = torch.empty((len(q), 0), device=q.device, dtype=self.bank.dtype)
            ids = torch.empty((len(q), 0), device=q.device, dtype=torch.long)
            for offset in range(0, len(self.bank), self.bank_chunk_size):
                similarities = q @ self.bank[offset:offset + self.bank_chunk_size].T
                if excluded is not None:
                    mask = excluded[start:start + len(q), None] == self.families[None, offset:offset + similarities.shape[1]]
                    similarities.masked_fill_(mask, -torch.inf)
                values, local = similarities.topk(min(k, similarities.shape[1]), dim=1)
                combined = torch.cat((best, values), dim=1)
                candidates = torch.cat((ids, local + offset), dim=1)
                best, positions = combined.topk(min(k, combined.shape[1]), dim=1)
                ids = candidates.gather(1, positions)
            if not torch.isfinite(best).all():
                raise ValueError("Not enough cross-family patches for Top-K")
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
        return (1 - F.cosine_similarity(query, reconstructed)).clamp(0, 2).cpu().numpy(), keep.cpu().numpy()

    @torch.inference_mode()
    def score(self, patches, top_fraction):
        if patches.ndim != 3 or patches.shape[0] != 1:
            raise ValueError("Reference scoring expects one image at a time")
        query = F.normalize(torch.as_tensor(patches[0], dtype=torch.float32, device=self.bank.device), dim=-1)
        similarities, ids = self.nearest(query)
        candidates = self.bank[ids].float()
        weights = (similarities / self.config["temperature"]).softmax(-1)
        weighted = torch.einsum("bk,bkd->bd", weights, candidates)
        distances = {"nearest": (1 - similarities[:, 0]).clamp(0, 2),
                     "topk": (1 - F.cosine_similarity(query, weighted)).clamp(0, 2)}
        if self.model is not None:
            reconstructed, weights = self.model(query, candidates)
            distances["cross_attention"] = (1 - F.cosine_similarity(query, reconstructed)).clamp(0, 2)
        distances = {key: value.cpu().numpy() for key, value in distances.items()}
        count = max(1, math.ceil(len(query) * top_fraction))
        scores = {key: float(np.sort(value)[-count:].mean()) for key, value in distances.items()}
        selected = "cross_attention" if self.model is not None else "topk"
        self.details = dict(candidate_ids=ids.cpu().numpy(), weights=weights.cpu().numpy(),
                            nearest_distances=distances["nearest"], scores=scores,
                            **{key + "_distances": value for key, value in distances.items() if key != "nearest"})
        return np.array([scores[selected]]), distances[selected][None], ids[:, 0].cpu().numpy()[None]


def family_split(sources, fraction, seed):
    if any(row["label"] != 0 or row["role"] != "bank" for row in sources):
        raise ValueError("Attention training requires bank real only")
    families = sorted({row["group_id"] for row in sources})
    if len(families) < 3:
        raise ValueError("Attention needs at least three real source families")
    np.random.default_rng(seed).shuffle(families)
    count = max(1, min(len(families) - 2, int(len(families) * fraction)))
    return sorted(families[count:]), sorted(families[:count])


def attention_info(output):
    if not (output / "stage1/attention.json").is_file():
        raise ValueError("請先完成 Stage 1 cross-attention 訓練，再執行 Stage 2")
    info = read_json(output / "stage1/attention.json")
    if (info["retrieval_sha256"] != sha256(output / "stage1/retrieval.json")
            or info["checkpoint_sha256"] != sha256(info["checkpoint"])):
        raise ValueError("Attention checkpoint/index changed")
    return info


def load_attention(info, device):
    model = CrossAttention(info["dimensions"], info["config"]["hidden"], info["config"]["heads"]).to(device)
    model.load_state_dict(load_file(info["checkpoint"], device=device))
    return model.eval().requires_grad_(False)


def train_attention(sources, arrays, args, bank, output):
    if (output / "stage1/attention.json").exists():
        attention_info(output)
        return
    config = args.attention
    train_families, validation_families = family_split(sources, config["validation_fraction"], args.seed)
    family_names = sorted(train_families + validation_families)
    source_family = np.array([family_names.index(row["group_id"]) for row in sources])
    patch_family = source_family[np.asarray(arrays["origins"])]
    train_mask = np.isin(patch_family, [family_names.index(name) for name in train_families])
    values = np.asarray(arrays["features"], dtype=np.float32)
    values = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
    references = values[train_mask]
    reference_families = patch_family[train_mask]
    rng = np.random.default_rng(args.seed)
    sampled = []
    # Bound training samples per image, retaining all training-family patches as references.
    for i in range(len(sources)):
        ids = np.flatnonzero(arrays["origins"] == i)
        sampled.extend(rng.choice(ids, min(len(ids), config["patches_per_image"]), replace=False).tolist())
    sampled = np.asarray(sampled)
    query = values[sampled]
    excluded = patch_family[sampled]
    train_ids = np.flatnonzero(train_mask[sampled])
    val_ids = np.flatnonzero(~train_mask[sampled])
    candidates = np.empty((len(sampled), config["neighbors"]), dtype=np.int64)
    devices = args.devices or ["cpu"]
    shards = np.array_split(np.arange(len(sampled)), len(devices))
    with tqdm(total=len(sampled), desc="跨家族 Top-K 候選", unit="patch", dynamic_ncols=True) as progress:
        def retrieve(ids, device):
            search = ReferenceBank(references, device, args.query_chunk_size, args.bank_chunk_size, config,
                                   families=reference_families)
            for start in range(0, len(ids), args.query_chunk_size):
                batch = ids[start:start + args.query_chunk_size]
                candidates[batch] = search.nearest(query[batch], excluded[batch])[1].cpu().numpy()
                with progress.get_lock():
                    progress.update(len(batch))
        map_devices(retrieve, shards, devices)
    # Validate before any optimizer updates; validation families never supply K/V.
    if np.any(reference_families[candidates] == excluded[:, None]):
        raise ValueError("Source-family leakage in attention candidates")
    torch.manual_seed(args.seed)
    model = CrossAttention(values.shape[-1], config["hidden"], config["heads"]).to(devices[0])
    parallel = nn.DataParallel(model, device_ids=args.gpu_ids) if len(args.devices) > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    directory = bank / "attention"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.safetensors"
    history, best, stale, best_epoch = [], float("inf"), 0, 0
    batch_size = config["batch_size"]
    batches = sum(math.ceil(len(ids) / batch_size) for ids in (train_ids, val_ids))
    with tqdm(total=config["epochs"] * batches, desc="Attention 訓練", unit="batch", dynamic_ncols=True) as progress:
        for epoch in range(1, config["epochs"] + 1):
            losses = {}
            for role, ids in (("train", rng.permutation(train_ids)), ("validation", val_ids)):
                training = role == "train"
                parallel.train(training)
                total = 0.
                progress.set_description(f"Attention {epoch}/{config['epochs']} {role}", refresh=False)
                for start in range(0, len(ids), batch_size):
                    batch = ids[start:start + batch_size]
                    target = torch.from_numpy(query[batch]).to(devices[0])
                    refs = torch.from_numpy(references[candidates[batch]]).to(devices[0])
                    with torch.set_grad_enabled(training):
                        inputs = F.normalize(target + torch.randn_like(target) * config["noise_std"], dim=-1) if training else target
                        predicted, _ = parallel(inputs, refs)
                        loss = (1 - F.cosine_similarity(predicted, target)).mean()
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite attention loss")
                        if training:
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(model.parameters(), 1.)
                            optimizer.step()
                    total += float(loss.detach()) * len(batch)
                    progress.set_postfix(loss=f"{total / (start + len(batch)):.5f}", patience=f"{stale}/{config['patience']}", refresh=False)
                    progress.update(1)
                losses[role] = total / len(ids)
            if losses["validation"] < best:
                best, best_epoch, stale = losses["validation"], epoch, 0
                temporary = checkpoint.with_suffix(".tmp.safetensors")
                save_file({key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}, str(temporary))
                temporary.replace(checkpoint)
            else:
                stale += 1
            history.append(dict(epoch=epoch, **losses, epochs_without_improvement=stale))
            write_json(output / "stage1/attention_history.json", history)
            progress.set_postfix(loss=f"{losses['validation']:.5f}", best=f"{best:.5f}",
                                 patience=f"{stale}/{config['patience']}", refresh=False)
            if stale >= config["patience"]:
                progress.set_description(f"Early stop {epoch}，最佳 epoch {best_epoch}", refresh=False)
                break
    write_json(output / "stage1/attention.json", dict(
        method="constrained multi-head cross-attention; frozen values; no query residual",
        config=config, dimensions=values.shape[-1], train_families=train_families,
        validation_families=validation_families, reference_patches=len(references),
        train_patches=len(train_ids), validation_patches=len(val_ids),
        best_epoch=best_epoch, best_validation_loss=best, epochs_completed=len(history),
        stopped_early=stale >= config["patience"], checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=sha256(checkpoint), retrieval_sha256=sha256(output / "stage1/retrieval.json")))
