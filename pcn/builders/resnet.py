from typing import Callable, Optional

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

TensorMap = Callable[[tf.Tensor], tf.Tensor]


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


def _resnet_body(
    in_cloud: comp.Cloud,
    features: FloatTensor,
    num_classes: int,
    num_levels: int,
    filters0: int,
    edge_features_fn: Callable,
    weight_fn: Callable,
    k0: int,
    r0: float,
    dropout_rate: float,
    mlp: Callable,
    bottleneck: bool,
    normalize: bool,
):
    radius = r0
    k1 = None if k0 is None else 2 * k0  # for down sample
    filters = filters0
    for i in range(1, num_levels):
        radius *= 2

        out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
            edge_features_fn=edge_features_fn,
            weight_fn=weight_fn,
            in_place_radius=radius,
            in_place_k=k0,
            down_sample_radius=radius * SQRT_2,
            down_sample_k=k1,
            normalize=normalize,
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
    ip_neigh = out_cloud.query(
        radius * 2, edge_features_fn, weight_fn, k=k0, normalize=normalize
    )
    features = identity_block(
        features, ip_neigh, dropout_rate, bottleneck=bottleneck, name=f"ip{num_levels}"
    )
    features = layers.Dense(filters * 4)(features)
    features = out_cloud.mask_invalid(features)
    features = tf.math.segment_max(features, out_cloud.model_structure.value_rowids)
    logits = mlp(features, num_classes=num_classes, activate_first=True)
    return logits


def _finalize(logits: FloatTensor, labels: IntTensor, weights: Optional[FloatTensor]):
    labels = mg.batch(mg.cache(labels))
    if weights is None:
        return logits, labels
    return logits, labels, mg.batch(mg.cache(weights))


@gin.configurable(module="pcn.builders")
def resnet(
    coords: FloatTensor,
    labels: IntTensor,
    weights: Optional[FloatTensor] = None,
    num_classes: int = 40,
    bucket_size: bool = False,
    num_levels: int = 3,
    filters0: int = 32,
    edge_features_fn: TensorMap = polynomial_edge_features,
    weight_fn: TensorMap = hat_weight,
    k0: int = 64,
    r0: float = R0,
    dropout_rate: float = 0.5,
    mlp: Callable = simple_mlp,
    bottleneck: bool = True,
    normalize: bool = True,
):
    in_cloud = comp.Cloud(coords, bucket_size=bucket_size)
    features = None
    k1 = None if k0 is None else 2 * k0  # for down sample

    out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
        in_place_radius=r0,
        in_place_k=k0,
        down_sample_radius=r0 * SQRT_2,
        down_sample_k=k1,
        normalize=normalize,
    )

    # initial in-place conv
    features = ip_neigh.convolve(features, filters=filters0)
    features = layers.BatchNormalization(name="ip0-bn0")(features)
    features = tf.keras.layers.Activation("relu")(features)

    # down sample
    features = conv_block(
        features,
        out_cloud.model_indices,
        filters0 * 2,
        ds_neigh,
        dropout_rate,
        name="ds0",
    )

    in_cloud = out_cloud
    logits = _resnet_body(
        in_cloud=in_cloud,
        features=features,
        num_classes=num_classes,
        num_levels=num_levels,
        filters0=filters0 * 2,
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
        k0=k0,
        r0=r0,
        dropout_rate=dropout_rate,
        mlp=mlp,
        bottleneck=bottleneck,
        normalize=normalize,
    )
    return _finalize(logits, labels, weights)


@gin.configurable(module="pcn.builders")
def resnet_small(
    coords: tf.Tensor,
    labels: tf.Tensor,
    weights: Optional[tf.Tensor] = None,
    num_classes: int = 40,
    bucket_size: bool = False,
    num_levels: int = 3,
    filters0: int = 32,
    edge_features_fn: TensorMap = polynomial_edge_features,
    weight_fn: TensorMap = hat_weight,
    k0: int = 64,
    r0: float = R0,
    dropout_rate: float = 0.5,
    mlp: Callable = simple_mlp,
    bottleneck: bool = True,
    normalize: bool = True,
):
    in_cloud = comp.Cloud(coords, bucket_size=bucket_size)
    features = None
    k1 = None if k0 is None else 2 * k0  # for down sample

    out_cloud, ds_neigh = in_cloud.down_sample_query(
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
        rejection_radius=r0,
        down_sample_radius=r0 * SQRT_2,
        down_sample_k=k1,
        normalize=normalize,
    )

    # initial down-sample conv
    features = ds_neigh.convolve(features, filters0, activation=None)
    features = layers.BatchNormalization(name="c0-bn1")(features)
    if dropout_rate is not None and dropout_rate > 0:
        features = layers.Dropout(dropout_rate)(features)
    features = tf.keras.layers.Activation("relu")(features)

    in_cloud = out_cloud
    logits = _resnet_body(
        in_cloud=in_cloud,
        features=features,
        num_classes=num_classes,
        num_levels=num_levels,
        filters0=filters0,
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
        k0=k0,
        r0=r0,
        dropout_rate=dropout_rate,
        mlp=mlp,
        bottleneck=bottleneck,
        normalize=normalize,
    )
    return _finalize(logits, labels, weights)
