"""Real-only DINOv3 partial fine-tuning for Stage 1."""

import copy
import math
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F
from tqdm.auto import tqdm

from script.bank_data import image_records, read_json, sha256, write_json
from script.experiment_paths import stage1_output
from Stage1.fsfm import (FeatureDecoder, cache_region_maps, feature_reconstruction_loss,
                         prepare_batch as prepare_fsfm_batch)


def configure_partial_finetuning(model, blocks):
    """Train the final DINOv3 blocks and output norm; keep earlier features fixed."""
    if not hasattr(model, "layer") or not hasattr(model, "norm") or not 1 <= blocks <= len(model.layer):
        raise ValueError("Unsupported DINOv3 model or invalid partial-finetune block count")
    model.requires_grad_(False)
    for layer in model.layer[-blocks:]:
        layer.requires_grad_(True)
    model.norm.requires_grad_(True)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def encoder_parameters(model, config):
    return configure_partial_finetuning(model, config["unfreeze_blocks"])


def encoder_state(model):
    return {name: parameter.detach().cpu().contiguous()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def load_encoder_tuning(model, checkpoint, config, device):
    """Attach the configured tuning structure and load its Stage 1 checkpoint."""
    parameters = encoder_parameters(model, config)
    expected = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    stored = load_file(str(checkpoint), device=str(device))
    if set(stored) != set(expected):
        raise ValueError("Encoder checkpoint parameters do not match the configured partial fine-tuning layers")
    with torch.no_grad():
        for name, parameter in expected.items():
            if stored[name].shape != parameter.shape:
                raise ValueError(f"Encoder checkpoint shape mismatch: {name}")
            parameter.copy_(stored[name])
    if len(parameters) != len(expected):
        raise ValueError("Encoder trainable parameter bookkeeping mismatch")
    return model


def _split_real(records, fraction, seed):
    families = sorted({row["group_id"] for row in records})
    if len(families) < 2:
        raise ValueError("Encoder tuning needs at least two real source families")
    random.Random(seed).shuffle(families)
    count = min(len(families) - 1, max(1, round(len(families) * fraction)))
    validation = set(families[:count])
    return ([row for row in records if row["group_id"] not in validation],
            [row for row in records if row["group_id"] in validation])


def _tokens(model, processor, images, device, mask=None):
    inputs = processor(images=images, return_tensors="pt").to(device)
    if mask is not None:
        mask = mask.flatten(1).to(device=device, dtype=torch.bool)
        expected = (len(images), (inputs["pixel_values"].shape[-1] // model.config.patch_size) ** 2)
        if tuple(mask.shape) != expected:
            raise ValueError(f"DINOv3 token mask shape {tuple(mask.shape)} does not match {expected}")
    hidden = model(**inputs, bool_masked_pos=mask).last_hidden_state
    return hidden[:, 1 + model.config.num_register_tokens:]


@torch.no_grad()
def update_ema(student, teacher, momentum):
    """Update the teacher parameters corresponding to trainable student layers."""
    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    if student_parameters.keys() != teacher_parameters.keys():
        raise ValueError("EMA teacher architecture does not match the student")
    for name, target in teacher_parameters.items():
        source = student_parameters[name]
        if source.requires_grad:
            target.mul_(momentum).add_(source.detach(), alpha=1 - momentum)


def ema_momentum(config, step, total_steps):
    progress = min(1., step / max(1, total_steps - 1))
    start, end = config["ema_start"], config["ema_end"]
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2
def _fsfm_real_loss(model, teacher, decoder, processor, originals, masks, region_masks, device, config):
    with torch.no_grad():
        target = _tokens(teacher, processor, originals, device)
    current = _tokens(model, processor, originals, device)
    masks = masks.to(device)
    region_masks = region_masks.to(device)
    masked = _tokens(model, processor, originals, device, mask=masks)
    reconstruction, components = feature_reconstruction_loss(
        decoder, masked, target, masks, region_masks, config["fsfm"])
    ema_consistency = (1 - F.cosine_similarity(current, target, dim=-1)).mean()
    total = reconstruction + config["ema_weight"] * ema_consistency
    return total, current, masked, dict(components, ema_consistency=ema_consistency)


def _limited_shuffle(rows, limit, seed):
    selected = list(rows)
    random.Random(seed).shuffle(selected)
    return selected[:min(len(selected), limit)]


def exclude_plan_images(plan, image_paths):
    """Remove invalid image pairs from every split while preserving source families."""
    image_paths = set(image_paths)
    removed = 0
    for role, videos in plan["groups"].items():
        kept = []
        for video in videos:
            frames = [frame for frame in video["frames"] if frame["image_path"] not in image_paths]
            removed += len(video["frames"]) - len(frames)
            if frames:
                video["frames"] = frames
                kept.append(video)
        plan["groups"][role] = kept
    if removed:
        plan["audit"]["segface_alignment_excluded"] = (
            plan["audit"].get("segface_alignment_excluded", 0) + removed)
    return removed


def tuning_info(output, args, bank):
    path = stage1_output(output) / "encoder_tuning.json"
    if not path.is_file():
        raise ValueError("請先完成 Stage 1 DINOv3 encoder tuning")
    info = read_json(path)
    checkpoint = Path(info["checkpoint"])
    if (info["config"] != args.encoder_tuning or info["plan_sha256"] != sha256(bank / "splits.json")
            or not checkpoint.is_file() or sha256(checkpoint) != info["checkpoint_sha256"]):
        raise ValueError("DINOv3 encoder tuning 設定、資料切分或權重已改變")
    decoder = info.get("fsfm_decoder")
    if decoder and (not Path(decoder["checkpoint"]).is_file()
                    or sha256(decoder["checkpoint"]) != decoder["checkpoint_sha256"]):
        raise ValueError("FSFM feature decoder checkpoint changed")
    args.encoder_checkpoint = str(checkpoint)
    return info


def train_encoder(plan, args, bank, output):
    stage1 = stage1_output(output)
    metadata = stage1 / "encoder_tuning.json"
    if metadata.exists():
        return tuning_info(output, args, bank)

    from transformers import AutoImageProcessor, AutoModel

    config = args.encoder_tuning
    real = image_records(plan["groups"]["bank"], "bank")
    train_rows, validation_rows = _split_real(real, config["validation_fraction"], args.seed)
    device = args.devices[0] if args.devices else "cpu"
    training_schedule = {epoch: _limited_shuffle(train_rows, config["images_per_epoch"], args.seed + epoch)
                         for epoch in range(1, config["epochs"] + 1)}
    validation = _limited_shuffle(validation_rows, config["validation_images"], args.seed)
    region_cache = bank / "face_regions"
    scheduled = validation + [row for values in training_schedule.values() for row in values]
    region_info = cache_region_maps(scheduled, args, region_cache, device)
    excluded = set(region_info["excluded_images"])
    if excluded:
        removed = exclude_plan_images(plan, excluded)
        if removed != len(excluded):
            raise ValueError("SegFace exclusion report does not match the current data split")
        write_json(bank / "splits.json", plan)
        write_json(stage1 / "splits.json", plan)
        train_rows = [row for row in train_rows if row["image_path"] not in excluded]
        validation_rows = [row for row in validation_rows if row["image_path"] not in excluded]
        training_schedule = {epoch: [row for row in values if row["image_path"] not in excluded]
                             for epoch, values in training_schedule.items()}
        validation = [row for row in validation if row["image_path"] not in excluded]
        if not validation or any(not values for values in training_schedule.values()):
            raise ValueError("Too many invalid SegFace pairs remain for encoder training")
    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    teacher = copy.deepcopy(model).eval().requires_grad_(False)
    tuned_parameters = encoder_parameters(model, config)
    decoder_checkpoint = stage1 / "fsfm_decoder.safetensors"
    decoder = FeatureDecoder(model.config.hidden_size, config["fsfm"]["grid_size"] ** 2,
                             config["fsfm"]).to(device)
    decoder_parameters = list(decoder.parameters())
    parameters = tuned_parameters + decoder_parameters
    groups = [{"params": tuned_parameters, "lr": config["backbone_learning_rate"]},
              {"params": decoder_parameters, "lr": config["decoder_learning_rate"]}]
    optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
    checkpoint = stage1 / "dino_partial.safetensors"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], float("inf"), 0
    total_steps = sum(math.ceil(len(rows) / config["batch_size"]) for rows in training_schedule.values())
    global_step = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            decoder.train()
            selected = training_schedule[epoch]
            totals = {"loss": 0., "masked_reconstruction": 0., "region_reconstruction": 0.,
                      "local_global": 0., "ema_consistency": 0., "ema_momentum": 0., "samples": 0}
            progress = tqdm(range(0, len(selected), config["batch_size"]), desc=f"DINO tune {epoch}/{config['epochs']}",
                            unit="batch", dynamic_ncols=True, leave=False)
            for batch_index, start in enumerate(progress):
                batch = selected[start:start + config["batch_size"]]
                optimizer.zero_grad(set_to_none=True)
                originals, masks, region_masks, _ = prepare_fsfm_batch(
                    batch, epoch, batch_index, pool, region_cache, config["fsfm"])
                loss, _, _, components = _fsfm_real_loss(
                    model, teacher, decoder, processor, originals, masks, region_masks, device, config)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite DINOv3 encoder-tuning loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"])
                optimizer.step()
                momentum = ema_momentum(config, global_step, total_steps)
                update_ema(model, teacher, momentum)
                global_step += 1
                count = len(batch)
                totals["loss"] += float(loss.detach()) * count
                for name in ("masked_reconstruction", "region_reconstruction", "local_global",
                             "ema_consistency"):
                    totals[name] += float(components[name].detach()) * count
                totals["ema_momentum"] += momentum * count
                totals["samples"] += count
                progress.set_postfix(loss=f"{totals['loss'] / totals['samples']:.4f}")

            model.eval()
            decoder.eval()
            validation_loss, validation_samples = 0., 0
            with torch.no_grad():
                for batch_index, start in enumerate(range(0, len(validation), config["batch_size"])):
                    batch = validation[start:start + config["batch_size"]]
                    originals, masks, region_masks, _ = prepare_fsfm_batch(
                        batch, 0, batch_index, pool, region_cache, config["fsfm"])
                    value, _, _, _ = _fsfm_real_loss(
                        model, teacher, decoder, processor, originals, masks, region_masks, device, config)
                    count = len(batch)
                    validation_loss += float(value) * count
                    validation_samples += count
            validation_loss /= max(1, validation_samples)
            if not math.isfinite(validation_loss):
                raise ValueError("Non-finite DINOv3 encoder-tuning validation loss")
            row = {"epoch": epoch, "train_total_loss": totals["loss"] / totals["samples"],
                   **{f"{name}_loss": totals[name] / totals["samples"] for name in
                      ("masked_reconstruction", "region_reconstruction", "local_global", "ema_consistency")},
                   "mean_ema_momentum": totals["ema_momentum"] / totals["samples"],
                   "validation_total_loss": validation_loss}
            history.append(row)
            tqdm.write(f"FSFM partial tune epoch {epoch}: total={row['train_total_loss']:.5f} "
                       f"mask={row['masked_reconstruction_loss']:.5f} "
                       f"region={row['region_reconstruction_loss']:.5f} "
                       f"global={row['local_global_loss']:.5f} ema={row['ema_consistency_loss']:.5f} "
                       f"val={validation_loss:.5f}")
            if validation_loss < best - config["minimum_delta"]:
                best, stale = validation_loss, 0
                save_file(encoder_state(model), str(checkpoint))
                save_file({name: value.detach().cpu().contiguous()
                           for name, value in decoder.state_dict().items()}, str(decoder_checkpoint))
                best_epoch = epoch
            else:
                stale += 1
            write_json(stage1 / "encoder_tuning_history.json", history)
            if stale >= config["patience"]:
                break

    info = {"method": (f"FSFM-inspired real-only partial fine-tuning: final {config['unfreeze_blocks']} "
                       "DINOv3 blocks; native token masking; EMA teacher; facial-region feature reconstruction"),
            "config": config, "device": device, "trainable_parameters": sum(p.numel() for p in parameters),
            "encoder_trainable_parameters": sum(p.numel() for p in tuned_parameters),
            "decoder_trainable_parameters": sum(p.numel() for p in decoder_parameters),
            "train_real_images": len(train_rows), "validation_real_images": len(validation_rows),
            "best_epoch": best_epoch, "best_validation_loss": best,
            "epochs_completed": len(history), "stopped_early": len(history) < config["epochs"],
            "train_families": sorted({r["group_id"] for r in train_rows}),
            "validation_families": sorted({r["group_id"] for r in validation_rows}),
            "plan_sha256": sha256(bank / "splits.json"), "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint)}
    info["face_regions"] = region_info
    info["fsfm_decoder"] = {"checkpoint": str(decoder_checkpoint.resolve()),
                            "checkpoint_sha256": sha256(decoder_checkpoint),
                            "used_for_feature_extraction": False}
    write_json(metadata, info)
    args.encoder_checkpoint = str(checkpoint)
    return info
