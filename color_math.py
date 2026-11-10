"""Colour matching for ComfyUI Impact Pack detailer crops.

All images are HxWx3 float RGB in [0, 1]. Statistics compare spatially
corresponding pixels in the original and generated crop. Only the detailer's
own SEGS mask determines which pixels are ultimately composited back.
"""

import cv2
import numpy as np

METHODS = (
    "lab_chroma",
    "lab_full",
    "chroma_mean_only",
    "boundary_chroma",
    "rgb_mean_std",
    "rgb_histogram",
)


def _rgb(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
    return np.ascontiguousarray(np.clip(image, 0.0, 1.0))


def _erode(binary, radius):
    if radius <= 0:
        return binary
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.erode(binary.astype(np.uint8), k).astype(bool)


def _mask_selector(mask, shape, reference_region, mask_erosion):
    if reference_region == "whole_crop" or mask is None:
        return np.ones(shape, dtype=bool)
    m = np.asarray(mask, dtype=np.float32).squeeze()
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {m.shape}")
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    raw = m >= 0.8  # exclude fractional feathering
    eroded = _erode(raw, int(mask_erosion))
    if np.count_nonzero(eroded) >= 64:
        return eroded
    return raw


def _boundary_selector(original_lab, selected, width):
    """Inner mask boundary, filtered by the original crop's dominant interior colour.

    This is a *heuristic*, not a skin detector. On hair/background-dominated boxes
    it falls back to the masked crop's statistics rather than trusting the ring.
    """
    inner = _erode(selected, max(1, int(width)))
    ring = selected & ~inner
    if np.count_nonzero(ring) < 64 or np.count_nonzero(inner) < 64:
        return selected

    # The central region's LAB median is a robust proxy for the main subject.
    # Eliminate strongly dissimilar material (hair, background or clothing)
    # without using skin-tone thresholds tied to complexion.
    core = _erode(inner, max(1, int(width // 2)))
    if np.count_nonzero(core) < 64:
        core = inner
    median = np.median(original_lab[core], axis=0)
    delta = original_lab - median
    similar = (np.linalg.norm(delta[:, :, 1:3], axis=2) < 20.0) & (np.abs(delta[:, :, 0]) < 35.0)
    filtered = ring & similar
    if np.count_nonzero(filtered) >= 64:
        return filtered
    return selected


def _mean_std_match(reference, generated, selector, channels, scale=True,
                    max_std_scale=1.25):
    out = generated.copy()
    for ch in channels:
        src = reference[:, :, ch][selector]
        dst = generated[:, :, ch][selector]
        ratio = 1.0
        if scale:
            ratio = float(np.clip(float(src.std()) / max(float(dst.std()), 1e-4),
                                  1.0 / max_std_scale, max_std_scale))
        out[:, :, ch] = (generated[:, :, ch] - float(dst.mean())) * ratio + float(src.mean())
    return out


def _hist_match(reference, generated, selector):
    """Quantile-map each RGB channel; intended as an experimental benchmark."""
    result = generated.copy()
    quantiles = np.linspace(0, 1, 257)
    for ch in range(3):
        src = reference[:, :, ch][selector]
        dst = generated[:, :, ch][selector]
        src_q = np.quantile(src, quantiles)
        dst_q = np.quantile(dst, quantiles)
        unique_dst, indices = np.unique(dst_q, return_index=True)
        if len(unique_dst) == 1:
            result[:, :, ch] = generated[:, :, ch] + float(src.mean() - dst.mean())
        else:
            result[:, :, ch] = np.interp(generated[:, :, ch], unique_dst, src_q[indices])
    return result


def match_color(original, generated, mask=None, *, method="lab_chroma", strength=0.5,
                reference_region="masked", match_luminance=False, mask_erosion=5,
                boundary_width=24, max_std_scale=1.25):
    """Return a colour-corrected crop; default retains the original LAB algorithm.

    LAB methods respect `match_luminance` except lab_full, which always
    includes lightness. RGB mean/std and RGB histogram inherently change lightness.
    Boundary mode requires a mask; without one it falls back to masked LAB.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown color matching method: {method}")
    if reference_region not in ("masked", "whole_crop"):
        raise ValueError(f"Unknown reference region: {reference_region}")
    original, generated = _rgb(original), _rgb(generated)
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0:
        return generated.copy()

    h, w = generated.shape[:2]
    if original.shape[:2] != (h, w):
        original = _rgb(cv2.resize(original, (w, h), interpolation=cv2.INTER_LINEAR))

    selector = _mask_selector(mask, (h, w), reference_region, mask_erosion)
    if np.count_nonzero(selector) < 64:
        return generated.copy()

    if method == "rgb_mean_std":
        matched = _mean_std_match(original, generated, selector, (0, 1, 2),
                                  max_std_scale=max_std_scale)
    elif method == "rgb_histogram":
        matched = _hist_match(original, generated, selector)
    else:
        src_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
        dst_lab = cv2.cvtColor(generated, cv2.COLOR_RGB2LAB)
        if method == "boundary_chroma" and mask is not None and reference_region == "masked":
            selector = _boundary_selector(src_lab, selector, max(1, int(boundary_width)))
        channels = (0, 1, 2) if (method == "lab_full" or match_luminance) else (1, 2)
        matched_lab = _mean_std_match(
            src_lab, dst_lab, selector, channels,
            scale=method not in ("chroma_mean_only", "boundary_chroma"),
            max_std_scale=max_std_scale,
        )
        matched_lab[:, :, 0] = np.clip(matched_lab[:, :, 0], 0, 100)
        matched = cv2.cvtColor(np.ascontiguousarray(matched_lab), cv2.COLOR_LAB2RGB)

    return np.clip(generated * (1 - strength) + matched * strength, 0, 1).astype(np.float32)


def match_lab(original, generated, mask=None, strength=0.5, match_luminance=False,
              reference_region="masked", mask_erosion=5, max_std_scale=1.25):
    """Backward-compatible entry point used by the initial node release."""
    return match_color(original, generated, mask, method="lab_chroma", strength=strength,
                       match_luminance=match_luminance, reference_region=reference_region,
                       mask_erosion=mask_erosion, max_std_scale=max_std_scale)
