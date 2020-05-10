from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from pcn.ops import conv as conv_ops


class SparseOpsTest(tf.test.TestCase):
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
        actual = self.evaluate(actual)
        np.testing.assert_allclose(actual, expected, atol=1e-3)

    # def test_featured(self):
    #     N_in = 100
    #     N_out = 50
    #     F_in = 2
    #     F_out = 7
    #     T = 5
    #     num_dims = 2
    #     x_in = np.random.normal(size=(N_in, num_dims)).astype(np.float32)
    #     # x_out = np.random.normal(size=(N_out, num_dims)).astype(np.float32)
    #     x_out = x_in[:N_out]
    #     features = np.random.uniform(size=(N_in, F_in)).astype(np.float32)
    #     tree = KDTree(x_in)
    #     ragged_indices = sort_ragged(
    #         tree.query_ball_point(x_out, 0.1, approx_neighbors=64))
    #     i = np.repeat(np.arange(N_out),
    #                   ragged_indices.row_lengths).astype(np.int64)
    #     j = ragged_indices.values
    #     sparse_indices = np.stack((i, j), axis=-1)

    #     E = sparse_indices.shape[0]
    #     assert (E > 0)
    #     edge_features = np.random.uniform(size=(T, E)).astype(np.float32)
    #     kernel = np.random.uniform(size=(T, F_in, F_out)).astype(np.float32)

    #     expected = np.zeros((N_out, F_out))
    #     for t in range(T):
    #         ef = edge_features[t]
    #         k = kernel[t]
    #         for ii, jj, ee in zip(i, j, ef):
    #             expected[ii] += np.matmul(features[jj], k) * ee

    #     dense_shape = N_out, N_in
    #     actual = conv_ops.sparse_conv(features, edge_features, sparse_indices,
    #                                   kernel, dense_shape)
    #     actual = self.evaluate(actual)
    #     np.testing.assert_allclose(actual, expected, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()
