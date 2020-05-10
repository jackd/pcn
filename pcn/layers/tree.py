"""Lambda wrappers around ops.tree."""
from typing import Optional

import tensorflow as tf

import pcn.ops.tree as _tree_ops

Lambda = tf.keras.layers.Lambda


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
    return Lambda(
        _tree_ops.ragged_in_place_and_down_sample_query,
        arguments=dict(
            in_place_radius=in_place_radius,
            in_place_k=in_place_k,
            rejection_radius=rejection_radius,
            max_sample_size=max_sample_size,
            down_sample_radius=down_sample_radius,
            down_sample_k=down_sample_k,
            leaf_size=leaf_size,
        ),
    )(coords)


def ragged_in_place_query(
    coords: tf.Tensor, radius: float, k: Optional[int] = None, leaf_size: int = 16
):
    return Lambda(
        _tree_ops.ragged_in_place_query,
        arguments=dict(radius=radius, k=k, leaf_size=leaf_size,),
    )(coords)


def ragged_down_sample_query(
    coords,
    rejection_radius: float,
    query_radius: float,
    k: Optional[int] = None,
    max_sample_size: Optional[int] = None,
    leaf_size: int = 16,
):
    kwargs = dict(
        rejection_radius=rejection_radius,
        query_radius=query_radius,
        k=k,
        max_sample_size=max_sample_size,
        leaf_size=leaf_size,
    )
    return Lambda(_tree_ops.ragged_down_sample_query, arguments=kwargs)(coords)


def _knn_query(args, **kwargs):
    return _tree_ops.knn_query(*args, **kwargs)


def knn_query(in_coords: tf.Tensor, out_coords: tf.Tensor, k: int):
    return Lambda(_knn_query, arguments=dict(k=k))([in_coords, out_coords])


def _ragged_ifp_sample_precomputed(args, **kwargs):
    return _tree_ops.ragged_ifp_sample_precomputed(*args, **kwargs)


def ragged_ifp_sample_precomputed(dists, indices, row_lengths, sample_size, eps=1e-8):
    args = [dists, indices]
    kwargs = dict(eps=eps)
    if isinstance(sample_size, int):
        kwargs["sample_size"] = sample_size
    else:
        args.append(sample_size)
    return Lambda(_ragged_ifp_sample_precomputed, arguments=kwargs)(args)


def _rect_ifp_sample_precomputed(args, **kwargs):
    return _tree_ops.rect_ifp_sample_precomputed(*args, **kwargs)


def rect_ifp_sample_precomputed(dists, indices, sample_size, eps=1e-8):
    args = [dists, indices]
    kwargs = dict(eps=eps)
    if isinstance(sample_size, int):
        kwargs["sample_size"] = sample_size
    else:
        args.append(sample_size)
    return Lambda(_rect_ifp_sample_precomputed, arguments=kwargs)(args)
