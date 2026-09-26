"""FSFM-inspired facial-region masking and normal feature reconstruction."""

import gc
import hashlib
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from script.bank_data import read_json, save_array, sha256, write_json
from script.face_crop import expand_mask


# SegFace CelebAMask-HQ ordering from models/segface/source/segface_celeb.py.
BACKGROUND, SKIN, EYEBROWS, EYES, NOSE, MOUTH, HAIR, FACE_BOUNDARY = range(8)
REGION_NAMES = ("background", "skin", "eyebrows", "eyes", "nose", "mouth", "hair", "face_boundary")
LABEL_GROUPS = {
    SKIN: (2,), EYEBROWS: (6, 7), EYES: (8, 9, 15), NOSE: (10,),
    MOUTH: (11, 12, 13), HAIR: (14,), FACE_BOUNDARY: (4, 5),
}
NON_FACE_LABELS = (0, 1, 3, 14, 16, 17, 18, 255)
MAX_CROP_MATCH_ERROR = 20.
FOREGROUND_THRESHOLDS = (32, 24, 16, 12, 8, 4, 2, 1, 0)


def patchify_labels(labels, grid_size):
    """Reduce a semantic label image to patch-majority class IDs."""
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 2 or labels.shape[0] % grid_size or labels.shape[1] % grid_size:
        raise ValueError(f"Semantic map {labels.shape} cannot be divided into {grid_size}x{grid_size} patches")
    patch_h, patch_w = labels.shape[0] // grid_size, labels.shape[1] // grid_size
    blocks = labels.reshape(grid_size, patch_h, grid_size, patch_w).transpose(0, 2, 1, 3)
    result = np.empty((grid_size, grid_size), dtype=np.uint8)
    for row in range(grid_size):
        for column in range(grid_size):
            result[row, column] = np.bincount(blocks[row, column].reshape(-1), minlength=256).argmax()
    return result


def semantic_regions(labels):
    """Map SegFace classes to the facial regions used by CRFR-P."""
    labels = np.asarray(labels, dtype=np.uint8)
    regions = np.full(labels.shape, BACKGROUND, dtype=np.uint8)
    for region, classes in LABEL_GROUPS.items():
        regions[np.isin(labels, classes)] = region

    skin = labels == 2
    surrounding = np.isin(labels, NON_FACE_LABELS)
    padded = np.pad(surrounding, 1, constant_values=True)
    touches_outside = (padded[:-2, 1:-1] | padded[2:, 1:-1]
                       | padded[1:-1, :-2] | padded[1:-1, 2:])
    regions[skin & touches_outside] = FACE_BOUNDARY
    return regions


def crfrp_mask(labels, foreground, ratio, seed):
    """Cover one informative facial region, then mask other regions proportionally."""
    regions = semantic_regions(labels).reshape(-1)
    foreground = np.asarray(foreground, dtype=bool).reshape(-1)
    valid = np.flatnonzero(foreground)
    if len(valid) < 2:
        raise ValueError("CRFR-P needs at least two foreground patches")
    rng = random.Random(seed)
    groups = {region: np.flatnonzero(foreground & (regions == region))
              for region in range(len(REGION_NAMES))}
    primary_options = [region for region in range(EYEBROWS, len(REGION_NAMES)) if len(groups[region])]
    if not primary_options:
        raise ValueError("SegFace found no informative facial region for CRFR-P")
    primary = rng.choice(primary_options)
    selected_region = groups[primary].tolist()
    target = min(len(valid) - 1, max(1, round(len(valid) * ratio)))
    if len(selected_region) > target:
        selected_region = rng.sample(selected_region, target)

    masked = set(selected_region)
    remaining_target = target - len(masked)
    candidates = []
    if remaining_target:
        other_groups = [values.tolist() for region, values in groups.items()
                        if region != primary and len(values)]
        remaining_count = sum(len(values) for values in other_groups)
        proportional_ratio = remaining_target / remaining_count
        for values in other_groups:
            rng.shuffle(values)
            count = min(len(values), math.floor(len(values) * proportional_ratio))
            masked.update(values[:count])
            candidates.extend(values[count:])
        needed = target - len(masked)
        if needed:
            masked.update(rng.sample(candidates, needed))

    mask = np.zeros(regions.size, dtype=bool)
    region_mask = np.zeros_like(mask)
    mask[list(masked)] = True
    region_mask[selected_region] = True
    return mask.reshape(labels.shape), region_mask.reshape(labels.shape), REGION_NAMES[primary]


def foreground_grid(image, grid_size, minimum=.25):
    pixels = np.asarray(image.convert("RGB"))
    foreground = (pixels.max(axis=-1) > 16).astype(np.uint8) * 255
    occupancy = np.asarray(Image.fromarray(foreground).resize(
        (grid_size, grid_size), Image.Resampling.BOX), dtype=np.float32) / 255
    return occupancy >= minimum


def region_cache_path(directory, row):
    return Path(directory) / Path(row["feature_path"])


def source_face_path(row, faces_dir):
    name = Path(row["feature_path"]).with_suffix(".jpg").name
    path = (Path(faces_dir) / row["video_id"] / name).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RetinaFace crop missing for FSFM region parsing: {path}")
    return path


def _source_signature(rows, faces_dir):
    values = []
    for row in sorted(rows, key=lambda item: (item["video_id"], item["frame_index"])):
        path = source_face_path(row, faces_dir)
        stat = path.stat()
        segmented = Path(row["image_path"])
        segmented_stat = segmented.stat()
        values.append((str(path), stat.st_size, stat.st_mtime_ns,
                       str(segmented), segmented_stat.st_size, segmented_stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def locate_segmented_crop(source_image, segmented_image, proposed):
    """Locate an existing masked SegFace crop within its RetinaFace source image."""
    source = np.asarray(source_image.convert("RGB"), dtype=np.int16)
    segmented = np.asarray(segmented_image.convert("RGB"), dtype=np.int16)
    height, width = segmented.shape[:2]
    if height > source.shape[0] or width > source.shape[1]:
        raise ValueError(f"Saved SegFace crop {width}x{height} exceeds source image dimensions")
    intensity = segmented.max(axis=-1)
    minimum = max(64, math.ceil(intensity.size * .01))
    foreground = None
    for threshold in FOREGROUND_THRESHOLDS:
        candidate = intensity > threshold
        if candidate.sum() >= minimum:
            foreground = candidate
            break
    if foreground is None:
        raise ValueError(
            f"Saved SegFace crop has too little visible foreground for alignment: max={intensity.max()}")

    def match_error(left, top):
        crop = source[top:top + height, left:left + width]
        return float(np.abs(crop - segmented)[foreground].mean())

    left, top, right, bottom = proposed
    if right - left == width and bottom - top == height:
        error = match_error(left, top)
        if error <= MAX_CROP_MATCH_ERROR:
            return proposed

    template_mask = np.repeat(foreground[:, :, None].astype(np.uint8) * 255, 3, axis=2)
    distances = cv2.matchTemplate(source.astype(np.float32), segmented.astype(np.float32),
                                  cv2.TM_SQDIFF, mask=template_mask)
    _, _, (left, top), _ = cv2.minMaxLoc(distances)
    error = match_error(left, top)
    if not math.isfinite(error) or error > MAX_CROP_MATCH_ERROR:
        raise ValueError(f"Saved SegFace crop does not match its RetinaFace source: mean error={error:.2f}")
    return left, top, left + width, top + height


def crop_semantic_map(labels, parser_config, source_image, segmented_image, image_size):
    """Align current semantic labels with the exact crop used by an existing SegFace JPG."""
    mask = np.full(labels.shape, 255, dtype=np.uint8)
    mask[np.isin(labels, parser_config["background_ids"])] = 0
    mask[np.isin(labels, parser_config["face_ids"])] = 1
    expanded = expand_mask(mask, parser_config["dilation_ratio"], parser_config["closing_ratio"],
                           parser_config["min_face_area"])
    if expanded is None:
        raise ValueError("SegFace could not reproduce the existing segmented face crop")
    rows, columns = np.nonzero(expanded)
    left, top = int(columns.min()), int(rows.min())
    right, bottom = int(columns.max()) + 1, int(rows.max()) + 1
    left, top, right, bottom = locate_segmented_crop(
        source_image, segmented_image, (left, top, right, bottom))
    cropped = labels[top:bottom, left:right]
    return np.asarray(Image.fromarray(cropped).resize((image_size, image_size), Image.Resampling.NEAREST))


def cache_region_maps(rows, args, directory, device):
    """Cache compact 14x14 SegFace semantic maps before loading DINOv3."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = args.face_parser
    checkpoint = Path(config["model_dir"]) / config["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError("SegFace weights missing; run: conda run -n pt230 python -m script.setup_segface")
    rows = list({row["image_path"]: row for row in rows}.values())
    exclusion_file = directory / "excluded_images.json"
    exclusions = {item["image_path"]: item for item in read_json(exclusion_file)} if exclusion_file.exists() else {}

    def cache_metadata(items, schema="fsfm-region-cache-v6"):
        return {
            "schema": schema, "image_size": args.encoder_tuning["fsfm"]["image_size"],
            "grid_size": args.encoder_tuning["fsfm"]["grid_size"],
            "source_count": len(items), "source_signature": _source_signature(items, args.faces_dir),
            "parser_checkpoint": str(checkpoint.resolve()), "parser_checkpoint_sha256": sha256(checkpoint),
            "parser_config": config, "region_names": list(REGION_NAMES),
        }

    active = [row for row in rows if row["image_path"] not in exclusions]
    expected = cache_metadata(active)
    metadata = directory / "region_cache.json"
    if metadata.exists():
        current = read_json(metadata)
        stable = ("image_size", "grid_size", "parser_checkpoint", "parser_checkpoint_sha256",
                  "parser_config", "region_names")
        if (current.get("schema") not in {f"fsfm-region-cache-v{version}" for version in range(1, 7)}
                or any(current.get(key) != expected[key] for key in stable)):
            raise ValueError("FSFM face-region cache does not match sources or SegFace settings; use a new experiment name")

    pending = [row for row in active if not region_cache_path(directory, row).is_file()]
    if not pending:
        write_json(exclusion_file, sorted(exclusions.values(), key=lambda item: item["image_path"]))
        write_json(metadata, expected)
        return dict(expected, excluded_images=sorted(set(exclusions) & {row["image_path"] for row in rows}),
                    exclusion_report=str(exclusion_file.resolve()))
    from script.face_parser import SegFaceParser

    parser = SegFaceParser(config, device)
    batch_size = args.encoder_tuning["fsfm"]["parser_batch_size"]
    with tqdm(total=len(pending), desc="Stage 1 FSFM：快取臉部區域", unit="image", dynamic_ncols=True) as progress:
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            images = []
            for row in batch:
                with Image.open(source_face_path(row, args.faces_dir)) as image:
                    images.append(image.convert("RGB"))
            maps = parser.semantic(images)
            for row, source, labels in zip(batch, images, maps):
                path = region_cache_path(directory, row)
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(row["image_path"]) as segmented:
                    try:
                        labels = crop_semantic_map(
                            labels, config, source, segmented.convert("RGB"), expected["image_size"])
                    except ValueError as error:
                        exclusions[row["image_path"]] = {
                            "image_path": row["image_path"],
                            "retinaface_path": str(source_face_path(row, args.faces_dir)),
                            "reason": str(error),
                        }
                        tqdm.write(f"排除錯配 SegFace 圖片：{row['image_path']} ({error})")
                        continue
                save_array(path, patchify_labels(labels, expected["grid_size"]))
            progress.update(len(batch))
    del parser
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    write_json(exclusion_file, sorted(exclusions.values(), key=lambda item: item["image_path"]))
    excluded = sorted(set(exclusions) & {row["image_path"] for row in rows})
    active = [row for row in rows if row["image_path"] not in exclusions]
    missing = [row["image_path"] for row in active if not region_cache_path(directory, row).is_file()]
    if missing:
        raise RuntimeError("FSFM region cache incomplete: " + ", ".join(missing[:5]))
    expected = cache_metadata(active)
    write_json(metadata, expected)
    return dict(expected, excluded_images=excluded, exclusion_report=str(exclusion_file.resolve()))


def prepare_batch(rows, epoch, batch_index, pool, directory, config):
    paths = [row["image_path"] for row in rows]
    images = [image.resize((config["image_size"], config["image_size"]), Image.Resampling.BICUBIC)
              for image in pool.map(_open_rgb, paths)]
    labels = [np.load(region_cache_path(directory, row), allow_pickle=False) for row in rows]
    masks, region_masks, names = [], [], []
    for index, (image, semantic) in enumerate(zip(images, labels)):
        seed = epoch * 1_000_003 + batch_index * 10_007 + index
        foreground = foreground_grid(image, config["grid_size"], config["foreground_minimum"])
        mask, region_mask, name = crfrp_mask(semantic, foreground, config["mask_ratio"], seed)
        masks.append(mask)
        region_masks.append(region_mask)
        names.append(name)
    return (images, torch.from_numpy(np.stack(masks)),
            torch.from_numpy(np.stack(region_masks)), names)


def _open_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


class FeatureDecoder(nn.Module):
    """Small contextual decoder used only for the FSFM-inspired pretext task."""

    def __init__(self, dimensions, patch_count, config):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dimensions))
        self.position = nn.Parameter(torch.zeros(1, patch_count, dimensions))
        layer = nn.TransformerEncoderLayer(
            dimensions, config["decoder_heads"], round(dimensions * config["decoder_mlp_ratio"]),
            dropout=config["decoder_dropout"], activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, config["decoder_layers"], enable_nested_tensor=False)
        self.projection = nn.Linear(dimensions, dimensions)
        nn.init.normal_(self.mask_token, std=.02)
        nn.init.normal_(self.position, std=.02)

    def forward(self, tokens, mask):
        if tokens.shape[1] != self.position.shape[1] or mask.shape != tokens.shape[:2]:
            raise ValueError("FSFM decoder token/mask shape mismatch")
        values = torch.where(mask.unsqueeze(-1), self.mask_token.to(tokens.dtype), tokens)
        return self.projection(self.decoder(values + self.position.to(tokens.dtype)))


def feature_reconstruction_loss(decoder, online, target, mask, region_mask, config):
    mask = mask.flatten(1)
    region_mask = region_mask.flatten(1)
    predicted = decoder(online, mask)
    distance = 1 - F.cosine_similarity(predicted, target.detach(), dim=-1)
    masked = distance[mask].mean()
    region = distance[region_mask].mean()
    local_global = (1 - F.cosine_similarity(predicted.mean(1), target.detach().mean(1), dim=-1)).mean()
    total = masked + config["region_weight"] * region + config["global_weight"] * local_global
    return total, {"masked_reconstruction": masked, "region_reconstruction": region,
                   "local_global": local_global}
