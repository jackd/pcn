import functools
from typing import Optional

import tensorflow as tf

from pcn.np_utils import tree as _tree


def ragged_in_place_and_down_sample_query(
    coords: tf.Tensor,
    in_place_radius: float,
    down_sample_radius: float,
    rejection_radius: Optional[float] = None,
    in_place_k: Optional[int] = None,
    down_sample_k: Optional[int] = None,
    max_sample_size: Optional[int] = None,
    leaf_size: int = 16,
):
    assert in_place_k is None or in_place_k >= 0
    assert down_sample_k is None or down_sample_k >= 0
    assert max_sample_size is None or max_sample_size >= 0
    in_place_radius = in_place_radius ** 2
    down_sample_radius = down_sample_radius ** 2
    if rejection_radius is not None:
        rejection_radius = rejection_radius ** 2

    fn = functools.partial(
        _tree.ragged_in_place_and_down_sample_query,
        in_place_radius=in_place_radius,
        in_place_k=in_place_k,
        rejection_radius=rejection_radius,
        max_sample_size=max_sample_size,
        down_sample_radius=down_sample_radius,
        down_sample_k=down_sample_k,
        leaf_size=leaf_size,
    )

    out = tf.numpy_function(fn, (coords,), (tf.int64,) * 5)
    for o in out:
        o.set_shape((None,))
    ip_indices, ip_counts, sample_indices, ds_indices, ds_counts = out
    ip_indices = tf.RaggedTensor.from_row_lengths(ip_indices, ip_counts, validate=False)
    ds_indices = tf.RaggedTensor.from_row_lengths(ds_indices, ds_counts, validate=False)
    return ip_indices, sample_indices, ds_indices


def ragged_in_place_query(
    coords: tf.Tensor, radius: float, k: Optional[int] = None, leaf_size: int = 16
):
    assert k is None or k > 0
    radius = radius ** 2

    fn = functools.partial(
        _tree.ragged_in_place_query, radius=radius, k=k, leaf_size=leaf_size
    )
    indices, lengths = tf.numpy_function(fn, (coords,), (tf.int64,) * 2)
    for o in (indices, lengths):
        o.set_shape((None,))
    indices = tf.RaggedTensor.from_row_lengths(indices, lengths, validate=False)
    return indices


def ragged_down_sample_query(
    coords,
    rejection_radius: float,
    query_radius: float,
    k: Optional[int] = None,
    max_sample_size: Optional[int] = None,
    leaf_size: int = 16,
):
    fn = functools.partial(
        _tree.ragged_down_sample_query,
        rejection_radius=rejection_radius,
        query_radius=query_radius,
        k=k,
        max_sample_size=max_sample_size,
        leaf_size=leaf_size,
    )
    sample_indices, query_indices, query_counts = tf.numpy_function(
        fn, (coords,), (tf.int64, tf.int64, tf.int64)
    )
    for t in (sample_indices, query_indices, query_counts):
        t.set_shape((None,))
    query_indices = tf.RaggedTensor.from_row_lengths(
        query_indices, query_counts, validate=False
    )
    return sample_indices, query_indices


def knn_query(in_coords: tf.Tensor, out_coords: tf.Tensor, k: int):
    fn = functools.partial(_tree.knn_query, k=k)
    dists, indices = tf.numpy_function(
        fn, (in_coords, out_coords), (tf.float32, tf.int64)
    )
    indices.set_shape((None, k))
    dists.set_shape((None, k))
    return dists, indices


def ragged_ifp_sample_precomputed(
    dists, indices, row_lengths, sample_size, eps: float = 1e-8
):
    dists.shape.assert_has_rank(1)
    indices.shape.assert_has_rank(1)
    row_lengths.shape.assert_has_rank(1)
    kwargs = dict(eps=eps)
    args = [dists, indices, row_lengths]
    if isinstance(sample_size, int):
        kwargs["sample_size"] = sample_size
    else:
        args.append(sample_size)

    fn = functools.partial(_tree.ifp_sample_precomputed, **kwargs)
    sample_indices = tf.py_function(fn, args, (tf.int64,))
    sample_indices.set_shape((sample_size if isinstance(sample_size, int) else None,))
    return sample_indices


def rect_ifp_sample_precomputed(dists, indices, sample_size, eps: float = 1e-8):
    dists.assert_has_rank(2)
    indices.assert_has_rank(2)
    assert dists.shape[1] == indices.shape[1]
    k = dists.shape[1]

    n = tf.shape(dists)[0]
    dists = tf.reshape(dists, (-1,))
    indices = tf.reshape(indices, (-1,))
    row_lengths = tf.fill((n,), k)
    return ragged_ifp_sample_precomputed(
        dists, indices, row_lengths, sample_size=sample_size, eps=eps
    )
