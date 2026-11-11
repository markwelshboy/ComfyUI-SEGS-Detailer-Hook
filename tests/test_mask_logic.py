"""Tests for boolean mask expressions and applying masks to Impact Pack SEGS."""

import importlib.util
import sys
import unittest
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PKG = "comfy_mask_logic_tests"
spec = importlib.util.spec_from_file_location(
    PKG, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
module = importlib.util.module_from_spec(spec)
sys.modules[PKG] = module
spec.loader.exec_module(module)

SEG = namedtuple("SEG", [
    "cropped_image", "cropped_mask", "confidence",
    "crop_region", "bbox", "label", "control_net_wrapper",
])


class TestMaskLogic(unittest.TestCase):
    def setUp(self):
        self.image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        self.bbox = torch.zeros((1, 64, 64), dtype=torch.float32)
        self.skin = torch.zeros_like(self.bbox)
        self.hair = torch.zeros_like(self.bbox)
        self.bbox[:, 10:50, 10:50] = 1
        self.skin[:, 16:44, 16:44] = 1
        self.hair[:, 30:55, 30:55] = 1
        seg = SEG(
            None, np.ones((40, 40), np.float32), 0.92,
            (10, 10, 50, 50), (15, 15, 45, 45), "face", None
        )
        self.segs = ((64, 64), [seg])

    def test_and_not(self):
        mask, expr = module.MaskExpressionNode().evaluate(
            "(bbox AND skin) AND NOT hair", image_opt=self.image,
            bbox=self.bbox, skin=self.skin, hair=self.hair
        )
        self.assertEqual(mask.shape, (1, 64, 64))
        self.assertTrue(mask[0, 20, 20] > 0)
        self.assertEqual(float(mask[0, 33, 33]), 0.0)
        self.assertIn("&", expr)

    def test_xor_and_or(self):
        node = module.MaskExpressionNode()
        xor, _ = node.evaluate("bbox XOR hair", bbox=self.bbox, hair=self.hair)
        or_mask, _ = node.evaluate("bbox OR hair", bbox=self.bbox, hair=self.hair)
        self.assertEqual(float(xor[0, 35, 35]), 0)
        self.assertEqual(float(or_mask[0, 35, 35]), 1)

    def test_missing_name_is_error(self):
        with self.assertRaisesRegex(ValueError, "not connected"):
            module.MaskExpressionNode().evaluate("bbox AND nose", bbox=self.bbox)

    def test_expression_is_not_python_eval(self):
        with self.assertRaises((ValueError, SyntaxError)):
            module.MaskExpressionNode().evaluate(
                "__import__('os').system('echo hello')", bbox=self.bbox
            )

    def test_bbox_output_and_segs_passthrough(self):
        mask, segs = module.SegsBBoxMaskNode().make_mask(self.image, self.segs)
        self.assertIs(segs, self.segs)
        self.assertEqual(float(mask[0, 20, 20]), 1)
        self.assertEqual(float(mask[0, 2, 2]), 0)

    def test_replace_preserves_seg_metadata(self):
        new_segs, mask = module.ApplyMaskToSegsNode().apply(
            self.image, self.segs, self.skin, mode="replace"
        )
        updated = new_segs[1][0]
        self.assertIsInstance(updated, SEG)
        self.assertEqual(updated.label, "face")
        self.assertEqual(updated.crop_region, (10, 10, 50, 50))
        self.assertEqual(updated.cropped_mask.shape, (40, 40))
        self.assertEqual(float(updated.cropped_mask[0, 0]), 0)
        self.assertEqual(float(updated.cropped_mask[10, 10]), 1)
        self.assertEqual(mask.shape, (1, 64, 64))

    def test_intersection_and_empty_stays_empty_by_default(self):
        blank = torch.zeros_like(self.bbox)
        new_segs, applied = module.ApplyMaskToSegsNode().apply(
            self.image, self.segs, blank, mode="and_existing"
        )
        self.assertEqual(float(new_segs[1][0].cropped_mask.sum()), 0)
        self.assertEqual(float(applied.sum()), 0)

    def test_fallback_can_keep_original(self):
        blank = torch.zeros_like(self.bbox)
        new_segs, _ = module.ApplyMaskToSegsNode().apply(
            self.image, self.segs, blank, fallback_to_original_if_empty=True
        )
        self.assertGreater(float(new_segs[1][0].cropped_mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()
