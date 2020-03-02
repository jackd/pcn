"""Lambda wrappers around ops.tree."""
import tensorflow as tf
import pcn.ops.tree as _tree_ops

Lambda = tf.keras.layers.Lambda


def ragged_in_place_and_down_sample_query(
        coords: tf.Tensor,
        in_place_radius: float,
        in_place_k: int,
        down_sample_radius: float,
        down_sample_k: int,
        max_sample_size: int = -1,
        leaf_size: int = 16,
):
    return Lambda(_tree_ops.ragged_in_place_and_down_sample_query,
                  arguments=dict(
                      in_place_radius=in_place_radius,
                      in_place_k=in_place_k,
                      max_sample_size=max_sample_size,
                      down_sample_radius=down_sample_radius,
                      down_sample_k=down_sample_k,
                      leaf_size=leaf_size,
                  ))(coords)


def ragged_in_place_query(coords: tf.Tensor,
                          radius: float,
                          k: int,
                          leaf_size: int = 16):
    return Lambda(_tree_ops.ragged_in_place_query,
                  arguments=dict(
                      radius=radius,
                      k=k,
                      leaf_size=leaf_size,
                  ))(coords)
