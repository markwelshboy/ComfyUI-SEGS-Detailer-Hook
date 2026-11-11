import sys
from pathlib import Path
import unittest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from color_math import METHODS, match_color, match_lab


class TestColorMatch(unittest.TestCase):
    def setUp(self):
        self.ref = np.zeros((128, 128, 3), dtype=np.float32)
        self.ref[:] = [0.72, 0.52, 0.43]
        self.gen = np.zeros_like(self.ref)
        self.gen[:] = [0.66, 0.50, 0.51]
        self.mask = np.zeros((128, 128), dtype=np.float32)
        self.mask[16:112, 16:112] = 1.0

    def test_zero_strength(self):
        np.testing.assert_allclose(match_color(self.ref, self.gen, self.mask, strength=0), self.gen)

    def test_masked_chroma_toward_original_without_luminance_change(self):
        out = match_color(self.ref, self.gen, self.mask, strength=1, match_luminance=False)
        src_lab = cv2.cvtColor(self.ref, cv2.COLOR_RGB2LAB)
        dst_lab = cv2.cvtColor(self.gen, cv2.COLOR_RGB2LAB)
        out_lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        self.assertLess(np.linalg.norm(out_lab[60, 60, 1:3] - src_lab[60, 60, 1:3]),
                        np.linalg.norm(dst_lab[60, 60, 1:3] - src_lab[60, 60, 1:3]))
        self.assertLess(abs(float(out_lab[60, 60, 0] - dst_lab[60, 60, 0])), 1)

    def test_empty_mask_noop(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        np.testing.assert_allclose(match_color(self.ref, self.gen, mask), self.gen)

    def test_tiny_mask_noop(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[10:12, 10:12] = 1
        np.testing.assert_allclose(match_color(self.ref, self.gen, mask), self.gen)

    def test_dimension_mismatch(self):
        out = match_color(self.ref[:90], self.gen, self.mask[:90])
        self.assertEqual(out.shape, self.gen.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_whole_crop_without_mask(self):
        out = match_color(self.ref, self.gen, None, reference_region="whole_crop")
        self.assertEqual(out.shape, self.gen.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_all_modes_bounded_and_different(self):
        rng = np.random.default_rng(123)
        ref = np.clip(self.ref + rng.normal(0, 0.04, self.ref.shape), 0, 1).astype(np.float32)
        gen = np.clip(self.gen + rng.normal(0, 0.07, self.gen.shape), 0, 1).astype(np.float32)
        outputs = []
        for method in METHODS:
            with self.subTest(method=method):
                out = match_color(ref, gen, self.mask, method=method, strength=0.65)
                self.assertEqual(out.shape, gen.shape)
                self.assertEqual(out.dtype, np.float32)
                self.assertTrue(np.isfinite(out).all())
                self.assertTrue((out >= 0).all() and (out <= 1).all())
                outputs.append(out)
        self.assertGreater(np.max(np.abs(outputs[0] - outputs[1])), 0.01)
        self.assertGreater(np.max(np.abs(outputs[0] - outputs[2])), 0.001)

    def test_lab_full_changes_luminance(self):
        a = match_color(self.ref, self.gen, self.mask, method="lab_chroma", strength=1)
        b = match_color(self.ref, self.gen, self.mask, method="lab_full", strength=1)
        la = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)[50, 50, 0]
        lb = cv2.cvtColor(b, cv2.COLOR_RGB2LAB)[50, 50, 0]
        src_l = cv2.cvtColor(self.ref, cv2.COLOR_RGB2LAB)[50, 50, 0]
        self.assertLess(abs(float(lb - src_l)), abs(float(la - src_l)))

    def test_boundary_interior_selection(self):
        # Ring is skin-like on left and hair-like on right; should prefer skin.
        original = self.ref.copy()
        original[16:112, 90:112] = [0.08, 0.07, 0.07]
        generated = self.gen.copy()
        out = match_color(original, generated, self.mask, method="boundary_chroma", strength=1)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(np.abs(out - generated).mean(), 0.001)

    def test_boundary_without_mask_fallback(self):
        a = match_color(self.ref, self.gen, None, method="boundary_chroma")
        b = match_color(self.ref, self.gen, None, method="chroma_mean_only")
        np.testing.assert_allclose(a, b)

    def test_legacy_match_lab(self):
        a = match_lab(self.ref, self.gen, self.mask)
        b = match_color(self.ref, self.gen, self.mask, method="lab_chroma")
        np.testing.assert_allclose(a, b)

    def test_unknown_method_rejected(self):
        with self.assertRaises(ValueError):
            match_color(self.ref, self.gen, self.mask, method="bad")

    def test_histogram_constant_input(self):
        out = match_color(self.ref, self.gen, self.mask, method="rgb_histogram")
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
