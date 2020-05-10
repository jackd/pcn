import numpy as np
import tensorflow as tf

import pcn.ops.utils as utils_ops


class OpUtilsTest(tf.test.TestCase):
    def test_normalize_rows(self):
        i = tf.constant([0, 0, 1, 1, 1, 2])
        values = tf.range(6)
        out_size = 3
        actual = utils_ops.normalize_rows(i, values, out_size)
        expected = [0, 1, 2 / 9, 3 / 9, 4 / 9, 1]
        np.testing.assert_equal(self.evaluate(actual), expected)

    def test_to_nearest_power(self):
        self.assertEqual(self.evaluate(utils_ops.to_nearest_power(3)), 4)
        self.assertEqual(self.evaluate(utils_ops.to_nearest_power(4)), 4)
        self.assertEqual(self.evaluate(utils_ops.to_nearest_power(5)), 8)

    def test_pad_ragged(self):
        rt = tf.RaggedTensor.from_row_lengths(tf.range(10), [3, 7])
        actual = utils_ops.pad_ragged(rt, 3)

        actual_vals, actual_splits = self.evaluate((actual.values, actual.row_splits))
        np.testing.assert_equal(actual_vals, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0])
        np.testing.assert_equal(actual_splits, [0, 3, 13])

    def test_pad_ragged_to_nearest_power(self):
        rt = tf.RaggedTensor.from_row_lengths(tf.range(10), [3, 7])
        actual = utils_ops.pad_ragged_to_nearest_power(rt)
        actual_vals, actual_splits = self.evaluate((actual.values, actual.row_splits))
        np.testing.assert_equal(
            actual_vals, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0]
        )
        np.testing.assert_equal(actual_splits, [0, 3, 16])


if __name__ == "__main__":
    tf.test.main()
    # OpUtilsTest().test_pad_ragged_to_nearest_power()
