import sys
from pathlib import Path
import unittest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from color_math import match_lab


class TestColorMatch(unittest.TestCase):
    def setUp(self):
        self.ref = np.zeros((128, 128, 3), dtype=np.float32)
        self.ref[:] = [0.72, 0.52, 0.43]
        self.gen = np.zeros_like(self.ref)
        self.gen[:] = [0.66, 0.50, 0.51]  # generated crop is too pink/cool
        self.mask = np.zeros((128, 128), dtype=np.float32)
        self.mask[16:112, 16:112] = 1.0

    def test_zero_strength(self):
        out = match_lab(self.ref, self.gen, self.mask, strength=0)
        np.testing.assert_allclose(out, self.gen)

    def test_masked_chroma_toward_original_without_luminance_change(self):
        out = match_lab(self.ref, self.gen, self.mask, strength=1, match_luminance=False)
        ref_lab = cv2.cvtColor(self.ref, cv2.COLOR_RGB2LAB)
        gen_lab = cv2.cvtColor(self.gen, cv2.COLOR_RGB2LAB)
        out_lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        self.assertLess(np.linalg.norm(out_lab[60, 60, 1:3] - ref_lab[60, 60, 1:3]),
                        np.linalg.norm(gen_lab[60, 60, 1:3] - ref_lab[60, 60, 1:3]))
        self.assertLess(abs(float(out_lab[60, 60, 0] - gen_lab[60, 60, 0])), 1)

    def test_mask_too_small_noop(self):
        tiny_mask = np.zeros((128, 128), dtype=np.float32)
        tiny_mask[10:12, 10:12] = 1
        out = match_lab(self.ref, self.gen, tiny_mask)
        np.testing.assert_allclose(out, self.gen)

    def test_dimension_mismatch(self):
        out = match_lab(self.ref[:90], self.gen, self.mask[:90])
        self.assertEqual(out.shape, self.gen.shape)
        self.assertTrue(np.isfinite(out).all())

    def test_whole_crop_without_mask(self):
        out = match_lab(self.ref, self.gen, None, reference_region="whole_crop")
        self.assertEqual(out.shape, self.gen.shape)
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
