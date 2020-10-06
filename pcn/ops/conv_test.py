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

    def test_csr_sparse_conv(self):
        K = 5
        F_in = 7
        F_out = 3
        N_in = 13
        N_out = 5
        T = 11

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

        kwargs = dict(
            features=node_features,
            edge_features=edge_features,
            sparse_indices=sparse_indices,
            kernel=kernel,
            dense_shape=dense_shape,
        )
        grad_args = node_features, kernel
        with tf.GradientTape() as tape:
            tape.watch(grad_args)
            sparse_impl = conv_ops.sparse_conv(**kwargs)
            sparse_grads = tape.gradient(sparse_impl, grad_args)
        with tf.GradientTape() as tape:
            tape.watch(grad_args)
            csr_impl = conv_ops.sparse_conv(**kwargs, use_csr=True)
            csr_grads = tape.gradient(csr_impl, grad_args)

        sparse_impl, csr_impl, sparse_grads, csr_grads = self.evaluate(
            (sparse_impl, csr_impl, sparse_grads, csr_grads)
        )
        np.testing.assert_allclose(sparse_impl, csr_impl, rtol=1e-5)
        for sparse_grad, csr_grad in zip(sparse_grads, csr_grads):
            np.testing.assert_allclose(sparse_grad, csr_grad, rtol=1e-5)


if __name__ == "__main__":
    tf.test.main()
