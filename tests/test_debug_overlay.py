"""Smoke tests for the visual diagnostic overlay (no running ComfyUI needed)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
PKG = "segs_detailer_overlay_test"
spec = importlib.util.spec_from_file_location(
    PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)
Overlay = module.SegsDebugOverlayNode


class TestSegsDebugOverlay(unittest.TestCase):
    def setUp(self):
        self.image = torch.ones((1, 128, 128, 3), dtype=torch.float32) * 0.5
        cropped = np.zeros((96, 96), dtype=np.float32)
        cropped[12:84, 12:84] = 1
        seg = SimpleNamespace(
            bbox=(32, 32, 96, 96),
            crop_region=(16, 16, 112, 112),
            cropped_mask=cropped,
            label="face",
            confidence=0.93,
        )
        self.segs = ((128, 128), [seg])

    def test_passthrough_masks_and_overlay(self):
        result, segs, mask, matching = Overlay().overlay(self.image, self.segs)
        self.assertEqual(tuple(result.shape), (1, 128, 128, 3))
        self.assertEqual(tuple(mask.shape), (1, 128, 128))
        self.assertEqual(tuple(matching.shape), (1, 128, 128))
        self.assertIs(segs, self.segs)
        self.assertGreater(float(mask.max()), 0)
        self.assertGreater(float(matching.max()), 0)

    def test_visual_toggles_do_not_change_outputs(self):
        _, _, mask1, match1 = Overlay().overlay(self.image, self.segs)
        _, _, mask2, match2 = Overlay().overlay(
            self.image, self.segs, show_bbox=False, show_crop_region=False,
            show_segs_mask=False, show_matching_region=False
        )
        np.testing.assert_array_equal(mask1.numpy(), mask2.numpy())
        np.testing.assert_array_equal(match1.numpy(), match2.numpy())

    def test_include_exclude_preview_masks(self):
        included = torch.zeros((1, 128, 128), dtype=torch.float32)
        included[:, 20:100, 20:100] = 1
        excluded = torch.zeros((1, 128, 128), dtype=torch.float32)
        excluded[:, 60:120, 60:120] = 1
        _, _, _, match = Overlay().overlay(
            self.image, self.segs, include_mask_opt=included,
            exclude_mask_opt=excluded, show_include_mask=True,
            show_exclude_mask=True
        )
        self.assertGreater(float(match[:, 30:50, 30:50].mean()), 0)
        self.assertEqual(float(match[:, 90:110, 90:110].max()), 0)

    def test_tint_is_transparent_and_preserves_nonmasked_pixels(self):
        result, _, _, _ = Overlay().overlay(
            self.image, self.segs, show_bbox=False, show_crop_region=False,
            show_matching_region=False, overlay_alpha=0.4
        )
        np.testing.assert_allclose(
            result[0, 0, 0].numpy(), [0.5, 0.5, 0.5], atol=1 / 255
        )
        rgb = result[0, 45, 45].numpy()
        self.assertGreater(rgb[1], rgb[0])
        self.assertLess(float(rgb.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
