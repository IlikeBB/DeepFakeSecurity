"""Real-dominant metric adaptation with occasional image-level fake ranking."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from script.bank_data import image_records, read_json, sha256, write_json


class PatchAdapter(nn.Module):
    def __init__(self, dimensions, hidden):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimensions, hidden), nn.GELU(), nn.Linear(hidden, dimensions))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, values):
        values = F.normalize(values, dim=-1)
        return F.normalize(values + .1 * self.layers(values), dim=-1)


def matching_scores(queries, references, top_fraction, excluded=None):
    # excluded is [images, reference patches]; it prevents matching one's own video.
    similarities = queries @ references.T
    if excluded is not None:
        if excluded.all(-1).any():
            raise ValueError("Training needs reference patches from another real video")
        similarities = similarities.masked_fill(excluded[:, None, :], -torch.inf)
    distances = (1 - similarities.max(-1).values).clamp(0, 2)
    count = max(1, math.ceil(distances.shape[-1] * top_fraction))
    return distances.topk(count, dim=-1).values.mean(-1)


def fake_ranking_loss(fake_scores, real_scores, margin):
    # A fake label supervises the aggregate image score, never every patch independently.
    return F.relu(real_scores.detach().mean() + margin - fake_scores).mean()


def provenance(bank_root):
    return {"bank_config_sha256": sha256(bank_root / "bank_config.json"),
            "plan_sha256": sha256(bank_root / "splits.json")}


def load_adapter(args, bank_root, results_root):
    if not getattr(args, "train_adapter", False):
        return None
    from safetensors.torch import load_file

    directory = results_root / "training"
    info = read_json(directory / "adapter.json")
    if any(info.get(key) != value for key, value in provenance(bank_root).items()):
        raise ValueError("Adapter does not match this bank/split; use a new experiment")
    weights = directory / "adapter.safetensors"
    if info["weights_sha256"] != sha256(weights):
        raise ValueError("Adapter weights changed; use a new experiment")
    adapter = PatchAdapter(info["dimensions"], args.adapter_hidden).to(args.device)
    adapter.load_state_dict(load_file(str(weights), device=args.device))
    return adapter.eval().requires_grad_(False)


@torch.inference_mode()
def adapt_features(patches, adapter):
    if adapter is None:
        return patches
    shape = patches.shape
    flat = np.asarray(patches, dtype=np.float32).reshape(-1, shape[-1])
    device = next(adapter.parameters()).device
    return np.concatenate([adapter(torch.from_numpy(flat[i:i + 4096]).to(device)).cpu().numpy()
                           for i in range(0, len(flat), 4096)]).reshape(shape)


def train_adapter(plan, args, bank_root, results_root):
    from safetensors.torch import save_file
    from script.feature_bank import load_sample

    directory = results_root / "training"
    if (directory / "adapter.json").exists():
        load_adapter(args, bank_root, results_root)
        print("Adapter already trained; using the completed checkpoint.", flush=True)
        return
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    records = image_records(plan["groups"]["bank"], "bank")
    fake_videos = plan["groups"].get("train_fake", []) if getattr(args, "train_fake_videos", 1) else []
    fake_records = image_records(fake_videos, "train_fake")
    real = torch.from_numpy(np.stack([load_sample(bank_root, r)[1] for r in records]).astype(np.float32))
    real = F.normalize(real.flatten(1, 2), dim=-1).to(args.device)
    families = sorted({r["group_id"] for r in records})
    if len(families) < 2:
        raise ValueError("Adapter training requires at least two independent real videos")
    groups = torch.tensor([families.index(r["group_id"]) for r in records], device=args.device)
    flat = real.flatten(0, 1)
    flat_groups = groups.repeat_interleave(real.shape[1])
    # Stratify references by source video so leave-video-out matching remains possible.
    per_group = max(1, args.reference_patches // len(families))
    selected = []
    for group in range(len(families)):
        indices = (flat_groups == group).nonzero().flatten().cpu().numpy()
        selected.extend(rng.choice(indices, min(per_group, len(indices)), replace=False).tolist())
    ref_raw, ref_groups = flat[selected], flat_groups[selected]
    adapter = PatchAdapter(real.shape[-1], args.adapter_hidden).to(args.device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=.01)
    step, fake_updates, history = 0, 0, []
    for epoch in range(args.train_epochs):
        totals = np.zeros(3)
        batches = 0
        order = rng.permutation(len(real))
        for start in range(0, len(order), args.train_batch_size):
            ids = order[start:start + args.train_batch_size]
            raw = real[ids]
            adapted = adapter(raw)
            with torch.no_grad():
                references = adapter(ref_raw)
            real_scores = matching_scores(adapted, references, args.top_fraction,
                                          groups[ids, None] == ref_groups[None, :])
            anchor = (adapted - raw).square().sum(-1).mean()
            loss = real_scores.mean() + args.anchor_weight * anchor
            step += 1
            fake_loss = torch.zeros((), device=args.device)
            if fake_records and step % args.fake_interval == 0:
                chosen = rng.choice(len(fake_records), min(args.fake_batch_size, len(fake_records)), replace=False)
                # Load only this small batch instead of keeping the full fake pool on GPU.
                fake = torch.from_numpy(np.stack([load_sample(results_root / "train_fake", fake_records[i])[1]
                                                  for i in chosen]).astype(np.float32)).to(args.device)
                fake = F.normalize(fake.flatten(1, 2), dim=-1)
                fake_scores = matching_scores(adapter(fake), references, args.top_fraction)
                fake_loss = fake_ranking_loss(fake_scores, real_scores, args.fake_margin)
                loss = loss + args.fake_weight * fake_loss
                fake_updates += 1
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 1.)
            optimizer.step()
            totals += [real_scores.detach().mean().item(), anchor.item(), fake_loss.item()]
            batches += 1
        entry = dict(epoch=epoch + 1, real_loss=totals[0] / batches, anchor_loss=totals[1] / batches,
                     fake_loss_per_real_step=totals[2] / batches, steps=step, fake_updates=fake_updates)
        history.append(entry)
        print(f"[adapter] {entry}", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "adapter.safetensors"
    temporary = checkpoint.with_suffix(".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()}, str(temporary))
    temporary.replace(checkpoint)
    write_json(directory / "adapter.json", dict(provenance(bank_root), dimensions=real.shape[-1],
               weights_sha256=sha256(checkpoint), real_images=len(records), fake_images=len(fake_records),
               steps=step, fake_updates=fake_updates, history=history,
               objective="leave-video-out real patch matching + feature anchoring + occasional image fake ranking"))
