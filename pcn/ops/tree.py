import functools
import tensorflow as tf
from pcn.np_utils import tree as _tree


def ragged_in_place_and_down_sample_query(
        coords: tf.Tensor,
        in_place_radius: float,
        in_place_k: int,
        down_sample_radius: float,
        down_sample_k: int,
        max_sample_size: int = -1,
        leaf_size: int = 16,
):

    in_place_radius = in_place_radius**2
    down_sample_radius = down_sample_radius**2

    fn = functools.partial(_tree.ragged_in_place_and_down_sample_query,
                           in_place_radius=in_place_radius,
                           in_place_k=in_place_k,
                           max_sample_size=max_sample_size,
                           down_sample_radius=down_sample_radius,
                           down_sample_k=down_sample_k,
                           leaf_size=leaf_size)

    out = tf.numpy_function(fn, (coords,), (tf.int64,) * 5)
    for o in out:
        o.set_shape((None,))
    ip_indices, ip_counts, sample_indices, ds_indices, ds_counts = out
    ip_indices = tf.RaggedTensor.from_row_lengths(ip_indices,
                                                  ip_counts,
                                                  validate=False)
    ds_indices = tf.RaggedTensor.from_row_lengths(ds_indices,
                                                  ds_counts,
                                                  validate=False)
    return ip_indices, sample_indices, ds_indices


def ragged_in_place_query(coords: tf.Tensor,
                          radius: float,
                          k: int,
                          leaf_size: int = 16):
    radius = radius**2

    fn = functools.partial(_tree.ragged_in_place_query,
                           radius=radius,
                           k=k,
                           leaf_size=leaf_size)
    indices, lengths = tf.numpy_function(fn, (coords,), (tf.int64,) * 2)
    for o in (indices, lengths):
        o.set_shape((None,))
    indices = tf.RaggedTensor.from_row_lengths(indices, lengths, validate=False)
    return indices
