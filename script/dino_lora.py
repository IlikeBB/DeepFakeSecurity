"""Lightweight LoRA adaptation of the final DINOv3 attention block."""

from io import BytesIO
import math
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from script.bank_data import image_records, read_json, sha256, write_json


TARGETS = ("q_proj", "k_proj", "v_proj")


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        options = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **options))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **options))
        self.scale = alpha / rank
        self.enabled = True
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, values):
        output = self.base(values)
        if self.enabled:
            output = output + F.linear(F.linear(values, self.lora_a), self.lora_b) * self.scale
        return output


def attach_lora(model, rank, alpha):
    if not hasattr(model, "layer") or not len(model.layer):
        raise ValueError("Unsupported DINOv3 model: missing transformer layers")
    attention = model.layer[-1].attention
    for name in TARGETS:
        base = getattr(attention, name, None)
        if not isinstance(base, nn.Linear):
            raise ValueError(f"Unsupported DINOv3 attention projection: {name}")
        setattr(attention, name, LoRALinear(base, rank, alpha))
    return model


def set_lora_enabled(model, enabled):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def lora_parameters(model):
    return [parameter for module in model.modules() if isinstance(module, LoRALinear)
            for parameter in (module.lora_a, module.lora_b)]


def lora_state(model):
    return {name: parameter.detach().cpu().contiguous()
            for name, parameter in model.named_parameters() if "lora_" in name}


def load_lora(model, checkpoint, config, device):
    attach_lora(model, config["rank"], config["alpha"])
    stored = load_file(str(checkpoint), device=str(device))
    expected = {name: parameter for name, parameter in model.named_parameters() if "lora_" in name}
    if set(stored) != set(expected):
        raise ValueError("LoRA checkpoint parameters do not match DINOv3")
    with torch.no_grad():
        for name, parameter in expected.items():
            if stored[name].shape != parameter.shape:
                raise ValueError(f"LoRA checkpoint shape mismatch: {name}")
            parameter.copy_(stored[name])
    return model


def _split_real(records, fraction, seed):
    families = sorted({row["group_id"] for row in records})
    if len(families) < 2:
        raise ValueError("LoRA training needs at least two real source families")
    random.Random(seed).shuffle(families)
    count = min(len(families) - 1, max(1, round(len(families) * fraction)))
    validation = set(families[:count])
    return ([row for row in records if row["group_id"] not in validation],
            [row for row in records if row["group_id"] in validation])


def _open_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def _compress(path, seed, quality_range, scale):
    """Approximate UMCL quality branches while preserving patch alignment."""
    image = _open_image(path)
    rng = random.Random(seed)
    size = image.size
    reduced = tuple(max(1, round(value * scale)) for value in size)
    if reduced != size:
        image = image.resize(reduced, Image.Resampling.LANCZOS).resize(size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=rng.randint(*quality_range), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def _local_discrepancy(path, seed, config):
    """Create one localized soft discrepancy and its image-space mask from a real face."""
    image = _open_image(path)
    rng = random.Random(seed)
    pixels = torch.from_numpy(np.array(image, copy=True))
    foreground = (pixels.amax(dim=-1) > 16).nonzero()
    if not len(foreground):
        raise ValueError(f"No foreground pixels for local discrepancy: {path}")
    center_y, center_x = foreground[rng.randrange(len(foreground))].tolist()
    width, height = image.size
    fraction = rng.uniform(config["minimum_fraction"], config["maximum_fraction"])
    box_width = max(4, round(width * fraction))
    box_height = max(4, round(height * fraction))
    left = min(max(0, center_x - box_width // 2), width - box_width)
    top = min(max(0, center_y - box_height // 2), height - box_height)
    box = (left, top, left + box_width, top + box_height)

    crop = image.crop(box)
    reduced = tuple(max(1, round(value * config["scale"])) for value in crop.size)
    crop = crop.resize(reduced, Image.Resampling.LANCZOS).resize(crop.size, Image.Resampling.LANCZOS)
    crop = ImageEnhance.Contrast(crop).enhance(rng.uniform(.65, 1.35))
    crop = ImageEnhance.Color(crop).enhance(rng.uniform(.65, 1.35))
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=rng.randint(*config["quality"]), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        corrupted = image.copy()
        corrupted.paste(encoded.convert("RGB"), box[:2])

    mask = Image.new("L", image.size)
    drawer = ImageDraw.Draw(mask)
    (drawer.ellipse if rng.random() < .5 else drawer.rectangle)(box, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1., min(box_width, box_height) * .05)))
    return Image.composite(corrupted, image, mask), mask


def _tokens(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    hidden = model(**inputs).last_hidden_state
    return hidden[:, 1 + model.config.num_register_tokens:]


def _batch_images(rows, epoch, batch_index, pool, config):
    paths = [row["image_path"] for row in rows]
    originals = list(pool.map(_open_image, paths))
    seeds = [epoch * 1_000_003 + batch_index * 10_007 + i for i in range(len(paths))]
    compression = config["compression"]
    mild = list(pool.map(_compress, paths, seeds,
                         [compression["mild_quality"]] * len(paths), [compression["mild_scale"]] * len(paths)))
    severe = list(pool.map(_compress, paths, [seed + 97 for seed in seeds],
                           [compression["severe_quality"]] * len(paths),
                           [compression["severe_scale"]] * len(paths)))
    anomalies = list(pool.map(_local_discrepancy, paths, [seed + 193 for seed in seeds],
                              [config["local_anomaly"]] * len(paths)))
    pseudo, masks = map(list, zip(*anomalies))
    return originals, mild, severe, pseudo, masks


def cross_compression_loss(clean, mild, severe, affinity_weight):
    """Align patch identity and patch-to-patch relations across quality branches."""
    views = [F.normalize(values, dim=-1) for values in (clean, mild, severe)]
    consistency = sum((1 - (left * right).sum(-1)).mean()
                      for left, right in ((views[0], views[1]), (views[0], views[2]),
                                          (views[1], views[2]))) / 3
    affinities = [values @ values.transpose(-1, -2) for values in views]
    affinity = (F.mse_loss(affinities[0], affinities[1])
                + F.mse_loss(affinities[0], affinities[2])) / 2
    return consistency + affinity_weight * affinity, consistency, affinity


def local_anomaly_loss(clean, pseudo, masks, config):
    distance = 1 - F.cosine_similarity(clean, pseudo, dim=-1)
    side = math.isqrt(distance.shape[1])
    if side * side != distance.shape[1]:
        raise ValueError("Local anomaly mask needs a square patch grid")
    occupancy = torch.stack([
        torch.from_numpy(np.array(mask.resize((side, side), Image.Resampling.BOX), copy=True))
        for mask in masks]).flatten(1).to(distance.device, dtype=distance.dtype) / 255
    selected = occupancy >= config["mask_threshold"]
    if not selected.any() or selected.all():
        raise ValueError("Local anomaly mask must contain anomaly and background patches")
    anomaly = F.relu(config["margin"] - distance[selected]).mean()
    background = distance[~selected].mean()
    return anomaly, background


def _real_loss(model, processor, originals, mild_images, severe_images, pseudo_images, masks, device, config):
    set_lora_enabled(model, False)
    with torch.no_grad():
        anchor = _tokens(model, processor, originals, device)
    set_lora_enabled(model, True)
    current = _tokens(model, processor, originals, device)
    mild = _tokens(model, processor, mild_images, device)
    severe = _tokens(model, processor, severe_images, device)
    pseudo = _tokens(model, processor, pseudo_images, device)
    compression, consistency, affinity = cross_compression_loss(
        current, mild, severe, config["compression"]["affinity_weight"])
    anchor_loss = (1 - F.cosine_similarity(current, anchor, dim=-1)).mean()
    anomaly, background = local_anomaly_loss(current, pseudo, masks, config["local_anomaly"])
    total = (compression + config["local_anomaly"]["weight"] * anomaly
             + config["local_anomaly"]["background_weight"] * background
             + config["anchor_weight"] * anchor_loss)
    return total, current, mild, {"compression": compression, "consistency": consistency,
        "affinity": affinity, "local_anomaly": anomaly, "background": background, "anchor": anchor_loss}


def _fake_loss(model, processor, fake_rows, real, altered, device, margin, pool):
    fake = _tokens(model, processor, list(pool.map(_open_image, [r["image_path"] for r in fake_rows])), device)
    real_pool = F.normalize(real.mean(1), dim=-1)
    altered_pool = F.normalize(altered.mean(1), dim=-1)
    fake_pool = F.normalize(fake.mean(1), dim=-1)
    real_similarity = (real_pool * altered_pool).sum(-1).mean()
    fake_similarity = (fake_pool @ real_pool.T).amax(dim=-1).mean()
    return F.relu(margin + fake_similarity - real_similarity)


def _limited_shuffle(rows, limit, seed):
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:min(len(selected), limit)]


def tuning_info(output, args, bank):
    path = output / "stage1/dino_lora.json"
    if not path.is_file():
        raise ValueError("請先完成 Stage 1 DINOv3 LoRA 訓練")
    info = read_json(path)
    checkpoint = Path(info["checkpoint"])
    if (info["config"] != args.encoder_tuning or info["plan_sha256"] != sha256(bank / "splits.json")
            or not checkpoint.is_file() or sha256(checkpoint) != info["checkpoint_sha256"]):
        raise ValueError("DINOv3 LoRA 設定、資料切分或權重已改變")
    args.encoder_checkpoint = str(checkpoint)
    return info


def train_lora(plan, args, bank, output):
    metadata = output / "stage1/dino_lora.json"
    if metadata.exists():
        return tuning_info(output, args, bank)

    from transformers import AutoImageProcessor, AutoModel

    config = args.encoder_tuning
    real = image_records(plan["groups"]["bank"], "bank")
    fake_available = image_records(plan["groups"]["train_fake"], "train_fake")
    fake = fake_available if config["fake_weight"] > 0 else []
    train_rows, validation_rows = _split_real(real, config["validation_fraction"], args.seed)
    device = args.devices[0] if args.devices else "cpu"
    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    attach_lora(model, config["rank"], config["alpha"])
    parameters = lora_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    checkpoint = output / "stage1/dino_lora.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("inf"), 0
    fake_cursor = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            selected = _limited_shuffle(train_rows, config["images_per_epoch"], args.seed + epoch)
            fake_selected = _limited_shuffle(fake, len(fake), args.seed + epoch)
            totals = {"loss": 0., "real": 0., "compression": 0., "affinity": 0., "local_anomaly": 0.,
                      "background": 0., "anchor": 0., "fake": 0., "samples": 0, "fake_steps": 0}
            progress = tqdm(range(0, len(selected), config["batch_size"]), desc=f"DINO LoRA {epoch}/{config['epochs']}",
                            unit="batch", dynamic_ncols=True, leave=False)
            for batch_index, start in enumerate(progress):
                batch = selected[start:start + config["batch_size"]]
                originals, mild, severe, pseudo, masks = _batch_images(batch, epoch, batch_index, pool, config)
                optimizer.zero_grad(set_to_none=True)
                real_loss, current, altered, components = _real_loss(
                    model, processor, originals, mild, severe, pseudo, masks, device, config)
                loss = real_loss
                fake_value = current.new_zeros(())
                if config["fake_weight"] > 0 and fake_selected and (batch_index + 1) % config["fake_interval"] == 0:
                    ids = [(fake_cursor + i) % len(fake_selected) for i in range(config["fake_batch_size"])]
                    fake_cursor += config["fake_batch_size"]
                    fake_value = _fake_loss(model, processor, [fake_selected[i] for i in ids], current, altered,
                                            device, config["fake_margin"], pool)
                    loss = loss + config["fake_weight"] * fake_value
                    totals["fake_steps"] += 1
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite DINOv3 LoRA loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
                optimizer.step()
                count = len(batch)
                totals["loss"] += float(loss.detach()) * count
                totals["real"] += float(real_loss.detach()) * count
                totals["compression"] += float(components["consistency"].detach()) * count
                totals["affinity"] += float(components["affinity"].detach()) * count
                totals["local_anomaly"] += float(components["local_anomaly"].detach()) * count
                totals["background"] += float(components["background"].detach()) * count
                totals["anchor"] += float(components["anchor"].detach()) * count
                totals["fake"] += float(fake_value.detach())
                totals["samples"] += count
                progress.set_postfix(loss=f"{totals['loss'] / totals['samples']:.4f}")

            model.eval()
            validation = _limited_shuffle(validation_rows, config["validation_images"], args.seed)
            validation_loss, validation_samples = 0., 0
            with torch.no_grad():
                for batch_index, start in enumerate(range(0, len(validation), config["batch_size"])):
                    originals, mild, severe, pseudo, masks = _batch_images(
                        validation[start:start + config["batch_size"]], 0, batch_index, pool, config)
                    value, _, _, _ = _real_loss(
                        model, processor, originals, mild, severe, pseudo, masks, device, config)
                    count = len(validation[start:start + config["batch_size"]])
                    validation_loss += float(value) * count
                    validation_samples += count
            validation_loss /= max(1, validation_samples)
            row = {"epoch": epoch, "train_total_loss": totals["loss"] / totals["samples"],
                   "real_only_loss": totals["real"] / totals["samples"],
                   "compression_consistency_loss": totals["compression"] / totals["samples"],
                   "relation_loss": totals["affinity"] / totals["samples"],
                   "local_anomaly_loss": totals["local_anomaly"] / totals["samples"],
                   "background_consistency_loss": totals["background"] / totals["samples"],
                   "anchor_loss": totals["anchor"] / totals["samples"],
                   "fake_margin_loss": totals["fake"] / max(1, totals["fake_steps"]),
                   "validation_total_loss": validation_loss}
            history.append(row)
            tqdm.write(
                f"DINO LoRA epoch {epoch}: total={row['train_total_loss']:.5f} "
                f"comp={row['compression_consistency_loss']:.5f} rel={row['relation_loss']:.5f} "
                f"local={row['local_anomaly_loss']:.5f} bg={row['background_consistency_loss']:.5f} "
                f"anchor={row['anchor_loss']:.5f} fake={row['fake_margin_loss']:.5f} "
                f"val={validation_loss:.5f}")
            if validation_loss < best - config["minimum_delta"]:
                best, stale = validation_loss, 0
                save_file(lora_state(model), str(checkpoint))
                best_epoch = epoch
            else:
                stale += 1
            write_json(output / "stage1/dino_lora_history.json", history)
            if stale >= config["patience"]:
                break

    info = {"method": "real-only single-image LoRA: cross-compression patch/affinity consistency; local soft-discrepancy margin; frozen-feature anchor",
            "config": config, "device": device, "trainable_parameters": sum(p.numel() for p in parameters),
            "train_real_images": len(train_rows), "validation_real_images": len(validation_rows),
            "fake_images_available": len(fake_available), "fake_images_used": len(fake),
            "best_epoch": best_epoch, "best_validation_loss": best,
            "epochs_completed": len(history), "stopped_early": len(history) < config["epochs"],
            "train_families": sorted({r["group_id"] for r in train_rows}),
            "validation_families": sorted({r["group_id"] for r in validation_rows}),
            "plan_sha256": sha256(bank / "splits.json"), "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint)}
    write_json(metadata, info)
    args.encoder_checkpoint = str(checkpoint)
    return info
