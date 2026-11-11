"""SEGS debugging overlay for Impact Pack workflows.

Draw bounding boxes, crop regions, SEGS masks and sampled matching pixels
over the original ComfyUI IMAGE. The original SEGS payload passes through.
No frontend JavaScript, image previews, or file writes are required.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .color_math import METHODS, compute_matching_selector

BBOX_COLOR = (255, 191, 0)         # amber RGB
CROP_COLOR = (0, 220, 255)         # cyan RGB
SEGS_COLOR = (64, 224, 128)        # green RGB
MATCH_COLOR = (210, 64, 255)       # magenta RGB
INCLUDE_COLOR = (80, 170, 255)     # blue RGB
EXCLUDE_COLOR = (255, 90, 90)      # red RGB


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _mask_array(value):
    if value is None:
        return None
    arr = _numpy(value).astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _full_image_mask_batches(value, batch, h, w):
    if value is None:
        return None
    arr = _mask_array(value)
    if arr.ndim == 2:
        arr = np.repeat(arr[None, ...], batch, axis=0)
    elif arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = np.repeat(arr, batch, axis=0)
        elif arr.shape[0] != batch:
            raise ValueError(f"Mask batch {arr.shape[0]} does not match image batch {batch}")
    else:
        raise ValueError(f"Unsupported optional mask shape {arr.shape}")

    if arr.shape[1:] != (h, w):
        arr = np.stack([
            cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            for a in arr
        ])
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _parse_segs(segs):
    try:
        (height, width), segments = segs
    except (TypeError, ValueError) as exc:
        raise ValueError("Expected Impact Pack SEGS ((height, width), [SEG, ...])") from exc
    parsed = []
    for seg in segments:
        bbox = tuple(float(x) for x in seg.bbox)
        crop = tuple(int(x) for x in seg.crop_region)
        confidence = getattr(seg, "confidence", None)
        confidence = None if confidence is None else float(_numpy(confidence).reshape(-1)[0])
        parsed.append({
            "bbox": bbox,
            "crop": crop,
            "mask": _mask_array(getattr(seg, "cropped_mask", None)),
            "label": str(getattr(seg, "label", "seg")),
            "confidence": confidence,
        })
    return (int(height), int(width)), parsed


def _rect_scaled(rect, sx, sy):
    x1, y1, x2, y2 = rect
    return tuple(int(round(v)) for v in (x1 * sx, y1 * sy, x2 * sx, y2 * sy))


def _paste_crop(mask, crop, shape):
    h, w = shape
    out = np.zeros((h, w), dtype=np.float32)
    if mask is None:
        return out
    if mask.ndim == 3:
        mask = mask[0]  # single-image detailer; first mask for a batched SEG
    x1, y1, x2, y2 = crop
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return out
    if mask.shape != (y2 - y1, x2 - x1):
        mask = cv2.resize(mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out[y1:y2, x1:x2] = np.clip(mask, 0.0, 1.0)
    return out


def _tint(image, mask, rgb, alpha):
    if alpha <= 0 or mask.max() <= 0:
        return image
    # ComfyUI IMAGE is RGB float [0,1]; colors are conventional RGB uint8.
    color = (np.array(rgb, dtype=np.float32) / 255.0).reshape(1, 1, 3)
    weight = np.clip(mask[..., None], 0.0, 1.0) * alpha
    return np.clip(image * (1 - weight) + color * weight, 0, 1)


def _box(image, rect, rgb, thickness, caption=None):
    x1, y1, x2, y2 = rect
    # OpenCV writes the raw channel tuple; our buffer is RGB, not BGR.
    color = tuple(int(c) for c in rgb)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if caption:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        text_thickness = max(1, thickness - 1)
        (tw, th), base = cv2.getTextSize(caption, font, scale, text_thickness)
        py1 = max(0, y1 - th - base - 6)
        py2 = min(image.shape[0] - 1, py1 + th + base + 6)
        px2 = min(image.shape[1] - 1, x1 + tw + 8)
        cv2.rectangle(image, (x1, py1), (px2, py2), color, -1)
        cv2.putText(image, caption, (x1 + 4, py2 - base - 2), font,
                    scale, (20, 20, 20), text_thickness, cv2.LINE_AA)


class SegsDebugOverlayNode:
    """Visualize Impact Pack BBOX, crop, mask and color-statistics selection."""

    CATEGORY = "ImpactPack/Detailer/Hooks"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "show_bbox": ("BOOLEAN", {"default": True}),
                "show_crop_region": ("BOOLEAN", {"default": True}),
                "show_segs_mask": ("BOOLEAN", {"default": True}),
                "show_matching_region": ("BOOLEAN", {"default": True}),
                "reference_region": (["masked", "whole_crop"],),
                "method": (list(METHODS), {"default": "lab_chroma"}),
                "mask_erosion": ("INT", {"default": 5, "min": 0, "max": 64}),
                "boundary_width": ("INT", {"default": 24, "min": 1, "max": 128}),
                "overlay_alpha": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "matching_alpha": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05}),
                "thickness": ("INT", {"default": 0, "min": 0, "max": 24}),
            },
            "optional": {
                "include_mask_opt": ("MASK",),
                "exclude_mask_opt": ("MASK",),
                "show_include_mask": ("BOOLEAN", {"default": False}),
                "show_exclude_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "SEGS", "MASK", "MASK")
    RETURN_NAMES = ("image", "segs", "combined_segs_mask", "combined_matching_mask")
    FUNCTION = "overlay"

    def overlay(self, image, segs, show_bbox=True, show_crop_region=True,
                show_segs_mask=True, show_matching_region=True,
                reference_region="masked", method="lab_chroma",
                mask_erosion=5, boundary_width=24,
                overlay_alpha=0.35, matching_alpha=0.45, thickness=0,
                include_mask_opt=None, exclude_mask_opt=None,
                show_include_mask=False, show_exclude_mask=False):
        frames = image.detach().cpu().float().numpy()
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected BxHxWx3 IMAGE, got {frames.shape}")
        batch, h, w, _ = frames.shape
        include = _full_image_mask_batches(include_mask_opt, batch, h, w)
        exclude = _full_image_mask_batches(exclude_mask_opt, batch, h, w)
        (seg_h, seg_w), detections = _parse_segs(segs)
        sx = w / seg_w
        sy = h / seg_h
        thickness = thickness if thickness > 0 else max(1, int(round(2 * h / 512)))
        final = []
        combined_segs = np.zeros((batch, h, w), dtype=np.float32)
        combined_match = np.zeros((batch, h, w), dtype=np.float32)

        for b in range(batch):
            frame = frames[b].copy()
            inc = include[b] if include is not None else None
            exc = exclude[b] if exclude is not None else None
            if show_include_mask and inc is not None:
                frame = _tint(frame, inc, INCLUDE_COLOR, overlay_alpha * 0.8)
            if show_exclude_mask and exc is not None:
                frame = _tint(frame, exc, EXCLUDE_COLOR, overlay_alpha * 0.8)

            for det in detections:
                crop = _rect_scaled(det["crop"], sx, sy)
                seg_mask = _paste_crop(det["mask"], crop, (h, w))
                combined_segs[b] = np.maximum(combined_segs[b], seg_mask)
                x1, y1, x2, y2 = crop
                x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
                y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue

                # Calculate against untouched original RGB, before any overlays.
                select = compute_matching_selector(
                    frames[b, y1:y2, x1:x2, :],
                    seg_mask[y1:y2, x1:x2],
                    method=method,
                    reference_region=reference_region,
                    mask_erosion=mask_erosion,
                    boundary_width=boundary_width,
                    include_mask=inc[y1:y2, x1:x2] if inc is not None else None,
                    exclude_mask=exc[y1:y2, x1:x2] if exc is not None else None,
                ).astype(np.float32)
                combined_match[b, y1:y2, x1:x2] = np.maximum(
                    combined_match[b, y1:y2, x1:x2], select
                )
                if show_segs_mask:
                    frame = _tint(frame, seg_mask, SEGS_COLOR, overlay_alpha)
                if show_matching_region:
                    frame = _tint(frame, _paste_crop(select, crop, (h, w)),
                                  MATCH_COLOR, matching_alpha)

            # Render rectangles last so the boundaries remain visible.
            image_u8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            for det in detections:
                crop = _rect_scaled(det["crop"], sx, sy)
                bbox = _rect_scaled(det["bbox"], sx, sy)
                if show_crop_region:
                    _box(image_u8, crop, CROP_COLOR, thickness, "crop")
                if show_bbox:
                    confidence = det["confidence"]
                    caption = det["label"] if confidence is None else (
                        f'{det["label"]} {confidence:.0%}'
                    )
                    _box(image_u8, bbox, BBOX_COLOR, thickness, caption)
            final.append(image_u8.astype(np.float32) / 255.0)

        return (
            torch.from_numpy(np.stack(final)).to(device=image.device, dtype=image.dtype),
            segs,
            torch.from_numpy(combined_segs),
            torch.from_numpy(combined_match),
        )
