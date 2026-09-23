"""Reusable pretrained semantic face parser; no feature-bank dependencies."""

import numpy as np
import torch
from torch.nn import functional as F


def masks_from_logits(logits, images, config):
    masks = []
    for image, output in zip(images, logits):
        resized = F.interpolate(output[None], size=image.size[::-1], mode="bilinear", align_corners=False)
        confidence, labels = resized.softmax(1).max(1)
        labels = labels[0].cpu().numpy()
        confident = confidence[0].cpu().numpy() >= config["pixel_confidence"]
        mask = np.full(labels.shape, 255, dtype=np.uint8)
        mask[confident & np.isin(labels, config["background_ids"])] = 0
        mask[confident & np.isin(labels, config["face_ids"])] = 1
        masks.append(mask)
    return masks


class SegFaceParser:
    """Official AAAI 2025 Swin-B/CelebAMask-HQ model, fixed RGB preprocessing."""

    def __init__(self, config, device):
        from pathlib import Path
        from torchvision import transforms
        from safetensors.torch import load_file
        from models.segface.source.segface_celeb import SegFaceCeleb

        path = Path(config["model_dir"]) / config["checkpoint"]
        self.model = SegFaceCeleb(512, "swin_base")
        self.model.load_state_dict(load_file(str(path)), strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.config, self.device = config, device

    @torch.inference_mode()
    def __call__(self, images):
        inputs = torch.stack([self.transform(image) for image in images]).to(self.device)
        return masks_from_logits(self.model(inputs, None, None), images, self.config)
