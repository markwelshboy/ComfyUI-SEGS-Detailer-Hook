"""LAB colour-statistics matching for Impact Pack detailer crops.

Only numpy and OpenCV are required by this module; both are already used by
ComfyUI-Impact-Pack. RGB inputs and outputs are float32 in [0, 1].
"""

import cv2
import numpy as np


def _rgb(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
    return np.ascontiguousarray(np.clip(image, 0.0, 1.0))


def match_lab(
    original,
    generated,
    mask=None,
    strength=0.5,
    match_luminance=False,
    reference_region="masked",
    mask_erosion=5,
    max_std_scale=1.25,
):
    """Match generated RGB crop's LAB stats to the pre-diffusion original crop.

    original / generated: HxWx3 float RGB [0..1]. Can differ in dimensions.
    mask: HxW soft/binary mask in original crop coordinates or final crop coords.
    reference_region: 'masked' or 'whole_crop'. Only controls statistics; the
      Impact detailer itself still controls final compositing.
    match_luminance: False leaves L (brightness/contrast) completely intact.
    strength: interpolation from generated (0) to color-corrected (1).
    """
    original = _rgb(original)
    generated = _rgb(generated)
    height, width = generated.shape[:2]
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0:
        return generated.copy()

    if original.shape[:2] != (height, width):
        original = cv2.resize(original, (width, height), interpolation=cv2.INTER_LINEAR)
        original = _rgb(original)

    selector = np.ones((height, width), dtype=bool)
    if reference_region == "masked" and mask is not None:
        m = np.asarray(mask, dtype=np.float32).squeeze()
        if m.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {m.shape}")
        if m.shape != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
        selector = m >= 0.8  # ignore partially feathered boundary
        if mask_erosion > 0:
            radius = int(mask_erosion)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            eroded = cv2.erode(selector.astype(np.uint8), kernel).astype(bool)
            if np.count_nonzero(eroded) >= 64:
                selector = eroded
        # Small or empty masks: leave image unchanged instead of wildly shifting colors.
        if np.count_nonzero(selector) < 64:
            return generated.copy()

    original_lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    generated_lab = cv2.cvtColor(generated, cv2.COLOR_RGB2LAB)
    corrected_lab = generated_lab.copy()
    channels = (0, 1, 2) if match_luminance else (1, 2)

    # Use the same spatial locations in the pre- and post-diffusion crops.
    # Cap std ratio to prevent low-variance crops becoming over-saturated.
    for channel in channels:
        src = original_lab[:, :, channel][selector]
        dst = generated_lab[:, :, channel][selector]
        src_mean, src_std = float(src.mean()), float(src.std())
        dst_mean, dst_std = float(dst.mean()), float(dst.std())
        ratio = np.clip(src_std / max(dst_std, 1e-4), 1.0 / max_std_scale, max_std_scale)
        corrected_lab[:, :, channel] = ((generated_lab[:, :, channel] - dst_mean) * ratio + src_mean)

    # Float32 OpenCV LAB uses L in [0,100] and a/b around [-128,127].
    corrected_lab[:, :, 0] = np.clip(corrected_lab[:, :, 0], 0.0, 100.0)
    corrected_rgb = cv2.cvtColor(np.ascontiguousarray(corrected_lab), cv2.COLOR_LAB2RGB)
    return np.clip(generated * (1.0 - strength) + corrected_rgb * strength, 0.0, 1.0).astype(np.float32)
