from typing import Optional

import gin
import numpy as np
import tensorflow as tf

import kblocks.multi_graph as mg
import pcn.components as comp
from kblocks.keras import layers
from pcn.builders.utils import hat_weight, polynomial_edge_features, simple_mlp

Lambda = tf.keras.layers.Lambda

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
    shortcut = features

    branch = layers.Dense(filters // 4, name=f"{name}-branch-d0")(features)
    branch = tf.keras.layers.Activation("relu")(branch)
    branch = layers.BatchNormalization(name=f"{name}-branch-bn0")(branch)
    branch = ds_neigh.convolve(branch, filters // 4, activation=None)
    branch = tf.keras.layers.Activation("relu")(branch)
    branch = layers.BatchNormalization(name=f"{name}-branch-bn1")(branch)
    branch = layers.Dense(filters, name=f"{name}-branch-d1")(branch)
    # no activation
    branch = layers.BatchNormalization(name=f"{name}-branch-bn2")(branch)

    if dropout_rate is not None and dropout_rate > 0:
        branch = layers.Dropout(dropout_rate)(branch)

    shortcut = tf.gather(shortcut, sample_indices)
    shortcut = layers.Dense(filters, name=f"{name}-short-d0")(shortcut)
    shortcut = layers.BatchNormalization(name=f"{name}-short-bn0")(shortcut)
    if dropout_rate is not None and dropout_rate > 0:
        shortcut = layers.Dropout(dropout_rate)(shortcut)

    features = tf.keras.layers.Add(name=f"{name}-add")([shortcut, branch])
    features = tf.keras.layers.Activation("relu")(features)

    return features


def up_sample_block(
    features: FloatTensor,
    upper_features: FloatTensor,
    us_neigh: comp.Neighborhood,
    dropout_rate: Optional[float],
    name: str,
):
    shortcut = upper_features

    filters = upper_features.shape[-1]
    assert filters is not None
    branch = us_neigh.convolve(features, filters)
    # no activation
    branch = layers.BatchNormalization(name=f"{name}-branch-bn0")(branch)

    if dropout_rate is not None and dropout_rate > 0:
        branch = layers.Dropout(dropout_rate)(branch)

    features = tf.keras.layers.Add(name=f"{name}-add")([shortcut, branch])
    features = tf.keras.layers.Activation("relu")(features)
    return features


def identity_block(
    features: tf.Tensor,
    ip_neigh: comp.Neighborhood,
    dropout_rate: Optional[float],
    bottleneck: bool,
    name: str,
):
    shortcut = features
    filters = features.shape[-1]
    assert ip_neigh.in_cloud is ip_neigh.out_cloud

    if bottleneck:
        branch = layers.Dense(filters // 4, name=f"{name}-branch-d0")(features)
        branch = tf.keras.layers.Activation("relu")(branch)
        branch = layers.BatchNormalization(name=f"{name}-branch-bn0")(branch)
        branch = ip_neigh.convolve(branch, filters // 4, activation=None)
        branch = tf.keras.layers.Activation("relu")(branch)
        branch = layers.BatchNormalization(name=f"{name}-branch-bn1")(branch)
        branch = layers.Dense(filters, name=f"{name}-branch-dense")(branch)
    else:
        branch = ip_neigh.convolve(features, filters, activation=None)
        branch = tf.keras.layers.Activation("relu")(branch)
        branch = layers.BatchNormalization(name=f"{name}-branch-bn1")(branch)
        branch = ip_neigh.convolve(branch, filters, activation=None)

    # no activation
    branch = layers.BatchNormalization(name=f"{name}-branch-bn2")(branch)
    if dropout_rate is not None and dropout_rate > 0:
        branch = layers.Dropout(dropout_rate)(branch)

    features = tf.keras.layers.Add(name=f"{name}-add")([shortcut, branch])
    features = tf.keras.layers.Activation("relu")(features)
    return features


@gin.configurable(module="pcn.builders")
def resnet(
    coords,
    labels,
    weights=None,
    num_classes=40,
    bucket_size=False,
    num_levels=3,
    filters0=32,
    edge_features_fn=polynomial_edge_features,
    weight_fn=hat_weight,
    k0=64,
    r0=R0,
    dropout_rate=0.5,
    mlp=simple_mlp,
    ip0=False,
    bottleneck=True,
    coords_as_inputs=False,
):
    in_cloud = comp.Cloud(coords, bucket_size=bucket_size)
    filters = filters0
    features = in_cloud.model_coords if coords_as_inputs else None
    radius = r0
    k1 = None if k0 is None else 2 * k0  # for down sample

    out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
        in_place_radius=radius,
        in_place_k=k0,
        down_sample_radius=radius * SQRT_2,
        down_sample_k=k1,
    )

    # initial in-place conv
    features = ip_neigh.convolve(features, filters=filters)
    features = layers.BatchNormalization(name="ip0-bn0")(features)
    features = tf.keras.layers.Activation("relu")(features)

    if ip0:
        features = identity_block(
            features, ip_neigh, dropout_rate, bottleneck=bottleneck, name="ip0"
        )

    # down sample
    filters *= 2
    features = conv_block(
        features, out_cloud.model_indices, filters, ds_neigh, dropout_rate, name="ds0"
    )

    in_cloud = out_cloud

    for i in range(1, num_levels):
        radius *= 2

        out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
            edge_features_fn=edge_features_fn,
            weight_fn=weight_fn,
            in_place_radius=radius,
            in_place_k=k0,
            down_sample_radius=radius * SQRT_2,
            down_sample_k=k1,
        )

        # in place conv
        features = identity_block(
            features, ip_neigh, dropout_rate, bottleneck=bottleneck, name=f"ip{i}"
        )

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
    ip_neigh = out_cloud.query(radius * 2, edge_features_fn, weight_fn, k=k0)
    features = identity_block(
        features, ip_neigh, dropout_rate, bottleneck=bottleneck, name=f"ip{num_levels}"
    )
    features = layers.Dense(filters * 4)(features)
    features = out_cloud.mask_invalid(features)
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
