from typing import Optional

import gin
import numpy as np
import tensorflow as tf

import kblocks.multi_graph as mg
import pcn.components as comp
from kblocks.extras.layers import ZeroInit
from kblocks.keras import layers

from .utils import hat_weight, polynomial_edge_features, simple_mlp

IntTensor = tf.Tensor
FloatTensor = tf.Tensor

R0 = 0.1125
SQRT_2 = np.sqrt(2)


def conv_block(
    features: FloatTensor,
    sample_indices: IntTensor,
    filters: int,
    ds_neigh: comp.Neighborhood,
    dropout_rate: Optional[float],
    name: str,
):
    shortcut = tf.gather(features, sample_indices)
    branch = layers.Dense(filters // 4, name=f"{name}-branch-d0")(features)
    branch = tf.keras.layers.Activation("relu")(branch)
    branch = ds_neigh.convolve(branch, filters // 4, activation="relu")
    branch = tf.keras.layers.Activation("relu")(branch)
    branch = layers.Dense(filters, name=f"{name}-branch-d1")(branch)
    if dropout_rate is not None:
        branch = layers.Dropout(dropout_rate)(branch)
    branch = ZeroInit(name=f"{name}-zero")(branch)

    shortcut = layers.Dense(filters, name=f"{name}-short-d0")(shortcut)
    if dropout_rate is not None:
        shortcut = layers.Dropout(dropout_rate)(shortcut)

    features = tf.keras.layers.Add(name=f"{name}-add")([shortcut, branch])
    features = tf.keras.layers.Activation("relu")(features)

    return features


def identity_block(
    features: tf.Tensor,
    ip_neigh: comp.Neighborhood,
    dropout_rate: Optional[float],
    name: str,
):
    shortcut = features
    filters = features.shape[-1]
    assert ip_neigh.in_cloud is ip_neigh.out_cloud

    branch = layers.Dense(filters // 4, name=f"{name}-branch-d0")(features)
    branch = tf.keras.layers.Activation("relu")(branch)
    branch = ip_neigh.convolve(branch, filters // 4, activation="relu")
    branch = tf.keras.layers.Activation("relu")(branch)
    branch = layers.Dense(filters, name=f"{name}-branch-dense")(branch)
    if dropout_rate is not None:
        branch = layers.Dropout(dropout_rate)(branch)
    branch = ZeroInit(name=f"{name}-zero")(branch)

    features = tf.keras.layers.Add(name=f"{name}-add")([shortcut, branch])
    features = tf.keras.layers.Activation("relu")(features)
    return features


@gin.configurable(module="pcn.builders")
def zero_resnet(
    coords,
    labels,
    weights=None,
    num_classes=40,
    bucket_size=False,
    num_levels=3,
    filters0=32,
    edge_features_fn=polynomial_edge_features,
    weight_fn=hat_weight,
    k0=32,
    r0=R0,
    dropout_rate=0.5,
    mlp=simple_mlp,
    ip0=False,
):
    in_cloud = comp.Cloud(coords, bucket_size=bucket_size)
    filters = filters0
    features = None
    radius = r0

    out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
        in_place_radius=radius,
        in_place_k=k0,
        down_sample_radius=radius * SQRT_2,
        down_sample_k=k0 * 2,
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
    )

    # initial in-place conv
    features = ip_neigh.convolve(features, filters=filters)
    features = tf.keras.layers.Activation("relu")(features)

    if ip0:
        features = identity_block(features, ip_neigh, dropout_rate, name="ip0")

    # down sample
    filters *= 2
    features = conv_block(
        features, out_cloud.model_indices, filters, ds_neigh, dropout_rate, name="ds0"
    )

    in_cloud = out_cloud

    for i in range(1, num_levels):
        radius *= 2

        out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
            in_place_radius=radius,
            in_place_k=k0,
            down_sample_radius=radius * SQRT_2,
            down_sample_k=k0 * 2,
            edge_features_fn=edge_features_fn,
            weight_fn=weight_fn,
        )

        # in place conv
        features = identity_block(features, ip_neigh, dropout_rate, name=f"ip{i}")

        # ds conv
        filters *= 2
        features = conv_block(
            features,
            out_cloud.model_indices,
            filters,
            ds_neigh,
            dropout_rate,
            name=f"ds{i}",
        )
        in_cloud = out_cloud

    # final in-place
    ip_neigh = out_cloud.query(radius * 2, k0, edge_features_fn, weight_fn)
    features = identity_block(features, ip_neigh, dropout_rate, name=f"ip{num_levels}")
    features = layers.Dense(filters * 4)(features)
    features = tf.math.unsorted_segment_max(
        features,
        out_cloud.model_structure.value_rowids,
        out_cloud.model_structure.nrows,
    )
    logits = mlp(features, num_classes=num_classes, activate_first=True)

    labels = mg.batch(mg.cache(labels))
    if weights is None:
        return logits, labels
    else:
        return logits, labels, mg.batch(mg.cache(weights))
