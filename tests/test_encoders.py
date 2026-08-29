import unittest
import numpy as np

from tabular_to_image import encode_m1, encode_m1_cat, encode_m2, encode_m3


class EncoderSmokeTests(unittest.TestCase):
    def setUp(self):
        self.num = np.array([0.1, 0.8, 0.4], dtype=float)
        self.cat = [np.array([0, 1, 0], dtype=float), np.array([1, 0], dtype=float)]

    def test_shapes(self):
        self.assertEqual(encode_m1(self.num, width=8).shape, (1, 3, 8))
        self.assertEqual(encode_m2(self.num, np.array([0.2, -0.9, 0.1]), grid=4).shape, (1, 4, 4))
        self.assertEqual(encode_m1_cat(self.num, self.cat, width=8, height=8).shape, (1, 8, 8))
        self.assertEqual(encode_m3(self.num, self.cat, width=8, height=8).shape, (3, 8, 8))

    def test_no_g_ablation_zeroes_green_channel(self):
        image = encode_m3(self.num, self.cat, width=8, height=8, use_cat=False)
        self.assertTrue(np.all(image[1] == 0))

    def test_rg_control_zeroes_blue_channel(self):
        image = encode_m3(self.num, self.cat, width=8, height=8, zero_b=True)
        self.assertTrue(np.all(image[2] == 0))


if __name__ == '__main__':
    unittest.main()
