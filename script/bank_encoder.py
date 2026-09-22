"""Frozen DINOv3 encoding for independent face images."""

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm


def load_encoder(args):
    if not getattr(args, "quiet", False):
        tqdm.write(f"[DINOv3] 載入模型至 {args.device}")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(args.device).eval()
    model.requires_grad_(False)
    return model, processor


@torch.inference_mode()
def encode(paths, model, processor, args, image_pool=None):
    cls, patches = [], []
    for start in range(0, len(paths), args.batch_size):
        def read_image(path):
            with Image.open(path) as image:
                return image.convert("RGB")
        batch = paths[start:start + args.batch_size]
        images = list(image_pool.map(read_image, batch)) if image_pool else [read_image(path) for path in batch]
        inputs = processor(images=images, return_tensors="pt").to(args.device)
        hidden = model(**inputs).last_hidden_state
        cls.append(hidden[:, 0].float().cpu().numpy())
        patch_tokens = hidden[:, 1 + model.config.num_register_tokens:]
        height, width = inputs["pixel_values"].shape[-2:]
        grid = (height // model.config.patch_size, width // model.config.patch_size)
        patches.append(patch_tokens.reshape(len(images), *grid, hidden.shape[-1]).float().cpu().numpy())
    cls, patches = np.concatenate(cls), np.concatenate(patches)
    if not np.isfinite(cls).all() or not np.isfinite(patches).all():
        raise ValueError("Encoder returned non-finite features")
    return cls.astype(args.dtype), patches.astype(args.dtype)
