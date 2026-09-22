"""Mask expansion and RGB cropping, independent of model and dataset orchestration."""

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


def expand_mask(mask, dilation_ratio, closing_ratio, min_face_area):
    face = (mask == 1).astype(np.uint8)
    if not face.any() or face.mean() < min_face_area:
        return None
    side = min(face.shape)
    for operation, ratio in ((cv2.MORPH_CLOSE, closing_ratio), (cv2.MORPH_DILATE, dilation_ratio)):
        radius = int(np.ceil(side * ratio))
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            face = cv2.morphologyEx(face, operation, kernel)
    return binary_fill_holes(face).astype(np.uint8)


def crop_masked_face(image, mask, background):
    if image.shape[:2] != mask.shape or image.ndim != 3:
        raise ValueError("RGB image and mask dimensions must match")
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("Cannot crop an empty face mask")
    left, top = int(columns.min()), int(rows.min())
    right, bottom = int(columns.max()) + 1, int(rows.max()) + 1
    pixels = image.copy()
    if background == "black":
        pixels[mask == 0] = 0
    elif background != "keep":
        raise ValueError("background must be black or keep")
    return pixels[top:bottom, left:right], [left, top, right, bottom]
