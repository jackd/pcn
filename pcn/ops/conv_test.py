import os
import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from pcn.ops import conv as conv_ops

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def get_conv_kwargs(K=5, F_in=7, F_out=3, N_in=13, N_out=5, T=11):

    num_edges = K * N_out

    while True:
        flat_index = np.random.randint(
            0,
            high=N_in * N_out,
            size=num_edges * 4,  # extras for duplicates
            dtype=np.int64,
        )
        flat_index = np.unique(flat_index)
        if len(flat_index) >= num_edges:
            flat_index = flat_index[:num_edges]
            break
    flat_index = np.sort(flat_index)
    i, j = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        flat_index, (N_out, N_in)
    )

    def random_normal(shape, dtype=tf.float32):
        return tf.constant(np.random.normal(size=shape), dtype=dtype)

    sparse_indices = tf.constant(np.stack((i, j), axis=-1), dtype=tf.int64)
    edge_features = random_normal(shape=(T, num_edges), dtype=tf.float32)
    node_features = random_normal(shape=(N_in, F_in), dtype=tf.float32)
    kernel = random_normal(shape=(T, F_in, F_out))
    dense_shape = [tf.constant(N_out), tf.constant(N_in)]

    return dict(
        features=node_features,
        edge_features=edge_features,
        sparse_indices=sparse_indices,
        kernel=kernel,
        dense_shape=dense_shape,
    )


class SparseOpsTest(tf.test.TestCase, parameterized.TestCase):
    def test_featureless(self):
        K = 5
        F_out = 3
        E = 10

        N_in = 3
        N_out = 5

        i = [0, 0, 1, 1, 1, 2, 3, 3, 3, 4]
        assert len(i) == E
        assert np.max(i) == N_out - 1
        j = np.random.randint(0, N_in, size=E)
        indices = np.stack((i, j), axis=-1)
        kernel = np.random.uniform(size=(K, F_out)).astype(np.float32)
        edge_features = np.random.uniform(size=((K, E))).astype(np.float32)

        expected = np.zeros((N_out, F_out), dtype=np.float32)
        for c, (i, j) in enumerate(indices):
            for k in range(K):
                expected[i] += kernel[k] * edge_features[k, c]

        actual = conv_ops.featureless_conv(
            edge_features, indices, kernel, out_size=N_out
        )
        actual = conv_ops.featureless_conv(
            edge_features, indices, kernel, out_size=N_out
        )
        actual = self.evaluate(actual)
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    # Currently no deterministic support for
    # - sparse_dense_matmul
    # - segment_sum
    # - unsorted_segment_sum
    # https://github.com/NVIDIA/framework-determinism
    @unittest.skip("No deterministic support for any implementations as of 2.4")
    @parameterized.parameters(*conv_ops.SparseImplementation.all())
    def test_deterministic_sparse_impl(self):
        """Ensure the given `SparseImplementation`s gives reproducible results."""
        impl = conv_ops.SparseImplementation.SORTED_GATHER_SUM
        kwargs = get_conv_kwargs()
        grad_args = kwargs["features"], kwargs["kernel"]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(grad_args)
            v0 = conv_ops.sparse_conv(**kwargs, sparse_impl=impl)
            g00, g01 = tape.gradient(v0, grad_args)
            v1 = conv_ops.sparse_conv(**kwargs, sparse_impl=impl)
            g10, g11 = tape.gradient(v0, grad_args)

        v0, g00, g01 = self.evaluate([v0, g00, g01])
        v1, g10, g11 = self.evaluate([v1, g10, g11])
        np.testing.assert_equal(v0, v1)
        np.testing.assert_equal(g00, g10)
        np.testing.assert_equal(g01, g11)

    def test_consistent_sparse_impl(self):
        """Ensure all `SparseImplementation`s give the same results."""
        kwargs = get_conv_kwargs()
        grad_args = kwargs["features"], kwargs["kernel"]
        opts = conv_ops.SparseImplementation.all()
        values = []
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(grad_args)
            for impl in conv_ops.SparseImplementation.all():
                v = conv_ops.sparse_conv(**kwargs, sparse_impl=impl)
                grads = tape.gradient(v, grad_args)
                values.append([v, *grads])

        values = self.evaluate(values)
        v0, g00, g01 = values[0]
        opt0 = opts[0]
        for opt, (v1, g10, g11) in zip(opts[1:], values[1:]):
            try:
                np.testing.assert_allclose(v0, v1, rtol=1e-5)
                np.testing.assert_allclose(g00, g10, rtol=1e-5)
                np.testing.assert_allclose(g01, g11, rtol=1e-5)
            except Exception as e:
                raise ValueError(
                    f"Inconsistent results using implementations {opt0} and {opt}"
                ) from e


if __name__ == "__main__":
    tf.test.main()
