"""Multi-mode Color Match Detailer Hook for ComfyUI-Impact-Pack.

Import the actual Impact hook base lazily in doit(), so custom-node startup
order does not matter. No Impact Pack files need modification.
"""

import logging
import numpy as np
import torch

from .color_math import METHODS, match_color
from .debug_overlay import SegsDebugOverlayNode
from .mask_logic import SegsBBoxMaskNode, MaskExpressionNode, ApplyMaskToSegsNode

log = logging.getLogger(__name__)


class ColorMatchDetailerHookProvider:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enabled": ("BOOLEAN", {"default": True}),
            "strength": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05}),
            "reference_region": (["masked", "whole_crop"],),
            "match_luminance": ("BOOLEAN", {"default": False}),
            "mask_erosion": ("INT", {"default": 5, "min": 0, "max": 64, "step": 1}),
            # Append new widgets so existing workflow widget order is preserved.
            "method": (list(METHODS),),
            "boundary_width": ("INT", {"default": 24, "min": 1, "max": 128, "step": 1}),
        }}

    RETURN_TYPES = ("DETAILER_HOOK",)
    RETURN_NAMES = ("detailer_hook",)
    FUNCTION = "doit"
    CATEGORY = "ImpactPack/Detailer/Hooks"
    DESCRIPTION = (
        "Match each diffused SEGS crop to its original crop with selectable "
        "LAB, boundary-aware, RGB statistical or histogram methods. "
        "Runs after VAE decode and before compositing; supports DetailerHookCombine."
    )

    def doit(self, enabled=True, strength=0.5, reference_region="masked",
             match_luminance=False, mask_erosion=5, method="lab_chroma",
             boundary_width=24):
        # Called only when the workflow runs, after all custom nodes have loaded.
        from impact.hooks import DetailerHook

        class ColorMatchDetailerHook(DetailerHook):
            def __init__(self):
                super().__init__()
                self._original_crop = None
                self._crop_mask = None

            def post_upscale(self, pixels, mask=None):
                # The detailer gives us the *actual resized crop* about to be
                # encoded, not the full image. Save a reference for this SEG.
                self._original_crop = pixels.detach().cpu().float().clone()
                self._crop_mask = (
                    mask.detach().cpu().float().clone() if mask is not None else None
                )
                return pixels

            def post_decode(self, pixels):
                original = self._original_crop
                mask = self._crop_mask
                # Clear after each detailer region; avoids retaining crop memory.
                self._original_crop = None
                self._crop_mask = None
                if not enabled or strength <= 0 or original is None:
                    return pixels

                # Regular ComfyUI IMAGE tensor: B,H,W,C. Some video-compatible
                # decoders return 1,B,H,W,C (Impact Pack squeezes leading 1 later).
                extra_axis = pixels.ndim == 5 and pixels.shape[0] == 1
                decoded = pixels.squeeze(0) if extra_axis else pixels
                if decoded.ndim != 4 or decoded.shape[-1] != 3:
                    log.warning("Color Match Detailer: unsupported decoded image shape: %s", tuple(pixels.shape))
                    return pixels
                if len(original) != len(decoded):
                    log.warning("Color Match Detailer: batch-size mismatch, skipping color match")
                    return pixels

                try:
                    corrected = []
                    for i in range(len(decoded)):
                        src = original[i].numpy()
                        dst = decoded[i].detach().cpu().float().numpy()
                        m = None
                        if mask is not None:
                            # Accept HxW, BxHxW, or BxHxWx1 mask layouts.
                            candidate = mask
                            if candidate.ndim == 4:
                                candidate = candidate[i, ..., 0]
                            elif candidate.ndim == 3:
                                candidate = candidate[i]
                            elif candidate.ndim != 2:
                                raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")
                            m = candidate.numpy()
                        corrected.append(match_color(
                            src, dst, m,
                            strength=strength,
                            match_luminance=match_luminance,
                            reference_region=reference_region,
                            mask_erosion=mask_erosion,
                            method=method,
                            boundary_width=boundary_width,
                        ))
                    result = torch.from_numpy(np.stack(corrected, axis=0)).to(
                        device=decoded.device, dtype=decoded.dtype
                    )
                    return result.unsqueeze(0) if extra_axis else result
                except Exception:
                    log.exception("Color Match Detailer: matching failed; returning decoded crop unchanged")
                    return pixels

        return (ColorMatchDetailerHook(),)


NODE_CLASS_MAPPINGS = {
    "ColorMatchDetailerHookProvider": ColorMatchDetailerHookProvider,
    "SegsDebugOverlayNode": SegsDebugOverlayNode,
    "SegsBBoxMaskNode": SegsBBoxMaskNode,
    "MaskExpressionNode": MaskExpressionNode,
    "ApplyMaskToSegsNode": ApplyMaskToSegsNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorMatchDetailerHookProvider": "Color Match Detailer Hook (Multi-Mode)",
    "SegsDebugOverlayNode": "SEGS Debug Overlay",
    "SegsBBoxMaskNode": "SEGS BBox Mask",
    "MaskExpressionNode": "Mask Logic Expression",
    "ApplyMaskToSegsNode": "Apply Mask To SEGS",
}
