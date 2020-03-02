import numpy as np
import tensorflow as tf
from pcn.components import ragged_to_block_sparse
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


def get_ragged_indices(n_in, n_out, num_edges):
    n_in = tf.convert_to_tensor(n_in, tf.int64)
    n_out = tf.convert_to_tensor(n_out, tf.int64)
    ij = tf.random.uniform((num_edges,), 0, n_in * n_out, dtype=tf.int64)
    ij = tf.sort(ij)
    i, j = tf.unravel_index(ij, (n_out, n_in))
    rt = tf.RaggedTensor.from_value_rowids(j, i)
    return rt.values, rt.row_splits, n_in, n_out


def gen():
    yield get_ragged_indices(100, 10, 500)
    yield get_ragged_indices(200, 20, 2000)
    yield get_ragged_indices(300, 50, 10000)


def map_fn(values, row_splits, n_in, n_out):
    return tf.RaggedTensor.from_row_splits(values, row_splits), n_in, n_out


dataset = tf.data.Dataset.from_generator(gen,
                                         (tf.int64,) * 4).map(map_fn).batch(3)

for rt, n_in, n_out in dataset:
    offset = tf.concat([[0], tf.math.cumsum(n_in)], axis=0)
    i, j = ragged_to_block_sparse(rt, offset)

    i = i.numpy()
    j = j.numpy()

    sp = coo_matrix((np.ones(i.shape), (i, j)))
    plt.spy(sp, markersize=0.5)

    plt.show()
