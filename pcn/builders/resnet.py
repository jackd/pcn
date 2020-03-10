from typing import Optional
import numpy as np
import tensorflow as tf
import gin

import kblocks.multi_graph as mg
from kblocks.keras import layers
from kblocks.extras.layers import ragged as ragged_layers
import pcn.components as comp
from .utils import hat_weight, polynomial_edge_features, simple_mlp

Lambda = tf.keras.layers.Lambda

IntTensor = tf.Tensor
FloatTensor = tf.Tensor

R0 = 0.1125
SQRT_2 = np.sqrt(2)


def conv_block(features: FloatTensor, sample_indices: IntTensor, filters: int,
               ds_neigh: comp.Neighborhood, dropout_rate: Optional[float],
               name: str):
    shortcut = features

    branch = layers.Dense(filters // 4, name=f'{name}-branch-d0')(features)
    branch = tf.keras.layers.Activation('relu')(branch)
    branch = layers.BatchNormalization(name=f'{name}-branch-bn0')(branch)
    branch = ds_neigh.convolve(branch, filters // 4, activation=None)
    branch = tf.keras.layers.Activation('relu')(branch)
    branch = layers.BatchNormalization(name=f'{name}-branch-bn1')(branch)
    branch = layers.Dense(filters, name=f'{name}-branch-d1')(branch)
    # no activation
    branch = layers.BatchNormalization(name=f'{name}-branch-bn2')(branch)

    if dropout_rate is not None and dropout_rate > 0:
        branch = layers.Dropout(dropout_rate)(branch)

    shortcut = tf.gather(shortcut, sample_indices)
    shortcut = layers.Dense(filters, name=f'{name}-short-d0')(shortcut)
    shortcut = layers.BatchNormalization(name=f'{name}-short-bn0')(shortcut)
    if dropout_rate is not None and dropout_rate > 0:
        shortcut = layers.Dropout(dropout_rate)(shortcut)

    features = tf.keras.layers.Add(name=f'{name}-add')([shortcut, branch])
    features = tf.keras.layers.Activation('relu')(features)

    return features


def up_sample_block(features: FloatTensor, upper_features: FloatTensor,
                    us_neigh: comp.Neighborhood, dropout_rate: Optional[float],
                    name: str):
    shortcut = upper_features

    filters = upper_features.shape[-1]
    assert (filters is not None)
    branch = us_neigh.convolve(features, filters)
    # no activation
    branch = layers.BatchNormalization(name=f'{name}-branch-bn0')(branch)

    features = tf.keras.layers.Add(name=f'{name}-add')([shortcut, branch])
    features = tf.keras.layers.Activation('relu')(features)
    return features


def identity_block(features: tf.Tensor, ip_neigh: comp.Neighborhood,
                   dropout_rate: Optional[float], bottleneck: bool, name: str):
    shortcut = features
    filters = features.shape[-1]
    assert (ip_neigh.in_cloud is ip_neigh.out_cloud)

    if bottleneck:
        branch = layers.Dense(filters // 4, name=f'{name}-branch-d0')(features)
        branch = tf.keras.layers.Activation('relu')(branch)
        branch = layers.BatchNormalization(name=f'{name}-branch-bn0')(branch)
        branch = ip_neigh.convolve(branch, filters // 4, activation=None)
        branch = tf.keras.layers.Activation('relu')(branch)
        branch = layers.BatchNormalization(name=f'{name}-branch-bn1')(branch)
        branch = layers.Dense(filters, name=f'{name}-branch-dense')(branch)
    else:
        branch = ip_neigh.convolve(features, filters, activation=None)
        branch = tf.keras.layers.Activation('relu')(branch)
        branch = layers.BatchNormalization(name=f'{name}-branch-bn1')(branch)
        branch = ip_neigh.convolve(branch, filters, activation=None)

    # no activation
    branch = layers.BatchNormalization(name=f'{name}-branch-bn2')(branch)
    if dropout_rate is not None and dropout_rate > 0:
        branch = layers.Dropout(dropout_rate)(branch)

    features = tf.keras.layers.Add(name=f'{name}-add')([shortcut, branch])
    features = tf.keras.layers.Activation('relu')(features)
    return features


def _double_search_radius_kwargs(in_place_radius, in_place_k,
                                 down_sample_radius, down_sample_k):
    return dict(in_place_radius=in_place_radius * 2,
                in_place_k=None if in_place_k is None else in_place_k * 4,
                down_sample_radius=down_sample_radius * 2,
                down_sample_k=None if down_sample_k is None else down_sample_k *
                4,
                rejection_radius=in_place_radius)


@gin.configurable(module='pcn.builders')
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
        double_search_radius=False,
        coords_as_inputs=False,
):
    in_cloud = comp.Cloud(coords, bucket_size=bucket_size)
    filters = filters0
    features = in_cloud.model_coords if coords_as_inputs else None
    radius = r0
    k1 = None if k0 is None else 2 * k0  # for down sample

    kwargs = dict(
        in_place_radius=radius,
        in_place_k=k0,
        down_sample_radius=radius * SQRT_2,
        down_sample_k=k1,
    )
    if double_search_radius:
        kwargs = _double_search_radius_kwargs(**kwargs)
    out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
        edge_features_fn=edge_features_fn, weight_fn=weight_fn, **kwargs)

    # initial in-place conv
    features = ip_neigh.convolve(features, filters=filters)
    features = layers.BatchNormalization(name='ip0-bn0')(features)
    features = tf.keras.layers.Activation('relu')(features)

    if ip0:
        features = identity_block(features,
                                  ip_neigh,
                                  dropout_rate,
                                  bottleneck=bottleneck,
                                  name='ip0')

    # down sample
    filters *= 2
    features = conv_block(features,
                          out_cloud.model_indices,
                          filters,
                          ds_neigh,
                          dropout_rate,
                          name='ds0')

    in_cloud = out_cloud

    for i in range(1, num_levels):
        radius *= 2

        kwargs = dict(
            in_place_radius=radius,
            in_place_k=k0,
            down_sample_radius=radius * SQRT_2,
            down_sample_k=k1,
        )
        if double_search_radius:
            kwargs = _double_search_radius_kwargs(**kwargs)

        out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
            **kwargs,
            edge_features_fn=edge_features_fn,
            weight_fn=weight_fn,
        )

        # in place conv
        features = identity_block(features,
                                  ip_neigh,
                                  dropout_rate,
                                  bottleneck=bottleneck,
                                  name=f'ip{i}')

        # ds conv
        filters *= 2
        features = conv_block(features,
                              out_cloud.model_indices,
                              filters,
                              ds_neigh,
                              dropout_rate,
                              name=f'ds{i}')
        in_cloud = out_cloud

    # final in-place
    ip_neigh = out_cloud.query(radius * 2, edge_features_fn, weight_fn, k=k0)
    features = identity_block(features,
                              ip_neigh,
                              dropout_rate,
                              bottleneck=bottleneck,
                              name=f'ip{num_levels}')
    features = layers.Dense(filters * 4)(features)
    features = out_cloud.mask_invalid(features)
    features = tf.math.unsorted_segment_max(
        features, out_cloud.model_structure.value_rowids,
        out_cloud.model_structure.nrows)
    logits = mlp(features, num_classes=num_classes, activate_first=True)

    labels = mg.batch(mg.cache(labels))
    if weights is None:
        return logits, labels
    else:
        return logits, labels, mg.batch(mg.cache(weights))


@gin.configurable(module='pcn.builders')
def resnet_small(
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
        bottleneck=True,
):
    in_cloud = comp.Cloud(coords, bucket_size=bucket_size)
    filters = filters0
    features = None
    radius = r0
    k1 = None if k0 is None else 2 * k0  # for down sample

    out_cloud, ds_neigh = in_cloud.down_sample_query(
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
        rejection_radius=radius,
        down_sample_radius=radius * SQRT_2,
        down_sample_k=k1,
    )

    features = ds_neigh.convolve(features, filters, activation=None)
    features = tf.keras.layers.Activation('relu')(features)
    features = layers.BatchNormalization(name='c0-bn1')(features)
    if dropout_rate is not None and dropout_rate > 0:
        features = layers.Dropout(dropout_rate)(features)

    in_cloud = out_cloud

    for i in range(1, num_levels):
        radius *= 2

        out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
            in_place_radius=radius,
            in_place_k=k0,
            down_sample_radius=radius * SQRT_2,
            down_sample_k=k1,
            edge_features_fn=edge_features_fn,
            weight_fn=weight_fn,
        )

        # in place conv
        features = identity_block(features,
                                  ip_neigh,
                                  dropout_rate,
                                  bottleneck=bottleneck,
                                  name=f'ip{i}')

        # ds conv
        filters *= 2
        features = conv_block(features,
                              out_cloud.model_indices,
                              filters,
                              ds_neigh,
                              dropout_rate,
                              name=f'ds{i}')
        in_cloud = out_cloud

    # final in-place
    ip_neigh = out_cloud.query(radius * 2, edge_features_fn, weight_fn, k=k0)
    features = identity_block(features,
                              ip_neigh,
                              dropout_rate,
                              bottleneck=bottleneck,
                              name=f'ip{num_levels}')
    features = layers.Dense(filters * 4)(features)
    features = out_cloud.mask_invalid(features)
    features = tf.math.unsorted_segment_max(
        features, out_cloud.model_structure.value_rowids,
        out_cloud.model_structure.nrows)
    logits = mlp(features, num_classes=num_classes, activate_first=True)

    labels = mg.batch(mg.cache(labels))
    if weights is None:
        return logits, labels
    else:
        return logits, labels, mg.batch(mg.cache(weights))


def _mask_logits(args):
    logits, mask = args
    return tf.where(mask, logits, -np.inf * tf.ones_like(logits))


def mask_logits(logits, mask):
    return Lambda(_mask_logits)([logits, mask])


@gin.configurable(module='pcn.builders')
def resnet_segmentation(
        inputs,
        labels,
        weights=None,
        logits_mask=None,
        num_classes=None,
        bucket_size=False,
        num_levels=3,
        filters0=32,
        edge_features_fn=polynomial_edge_features,
        weight_fn=hat_weight,
        k0=128,
        r0=R0,
        dropout_rate=0.5,
        mlp=simple_mlp,
        ip0=False,
        bottleneck=True,
        double_search_radius=False,
        coords_as_inputs=False,
):
    coords = inputs['positions']
    if logits_mask is None:
        assert (num_classes is not None)
        num_object_classes = None
    else:
        assert (isinstance(logits_mask, np.ndarray))
        if num_classes is not None:
            assert (logits_mask.shape[1] == num_classes)
        num_object_classes, num_classes = logits_mask.shape
    object_class = inputs.get('object_class')
    if object_class is not None:
        object_class = mg.model_input(mg.batch(mg.cache(
            inputs['object_class'])))

    cloud0 = comp.Cloud(coords, bucket_size=bucket_size)
    in_cloud = cloud0
    filters = filters0
    radius = r0
    k1 = None if k0 is None else k0 * 2

    ip_neighs = []
    ds_neighs = []
    all_features = []

    kwargs = dict(
        in_place_radius=radius,
        in_place_k=k0,
        down_sample_radius=radius * SQRT_2,
        down_sample_k=k1,
    )
    if double_search_radius:
        kwargs = _double_search_radius_kwargs(**kwargs)

    out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
        **kwargs,
        edge_features_fn=edge_features_fn,
        weight_fn=weight_fn,
    )

    features = inputs.get('normals')
    if features is not None:
        features = mg.cache(features)
        with mg.pre_batch_context():
            features = ragged_layers.pre_batch_ragged(features)
        features = mg.batch(features)
        with mg.post_batch_context():
            features = ragged_layers.flat_values(features)
            if bucket_size:
                raise NotImplementedError('TODO: padding')
        features = mg.model_input(features)
    if coords_as_inputs:
        features = tf.concat((features, cloud0.model_coords), axis=-1)

    # initial in-place conv
    features = ip_neigh.convolve(features, filters=filters)
    features = layers.BatchNormalization(name='ip0-bn0')(features)
    features = tf.keras.layers.Activation('relu')(features)

    if ip0:
        features = identity_block(features,
                                  ip_neigh,
                                  dropout_rate,
                                  bottleneck=bottleneck,
                                  name='ip0')

    ip_neighs.append(ip_neigh)
    ds_neighs.append(ds_neigh)
    all_features.append(features)
    # down sample
    filters *= 2
    features = conv_block(features,
                          out_cloud.model_indices,
                          filters,
                          ds_neigh,
                          dropout_rate,
                          name='ds0')

    in_cloud = out_cloud

    for i in range(1, num_levels):
        radius *= 2

        kwargs = dict(in_place_radius=radius,
                      in_place_k=k0,
                      down_sample_radius=radius * SQRT_2,
                      down_sample_k=k1)
        if double_search_radius:
            kwargs = _double_search_radius_kwargs(**kwargs)
        out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
            edge_features_fn=edge_features_fn,
            weight_fn=weight_fn,
            **kwargs,
        )

        # in place conv
        features = identity_block(features,
                                  ip_neigh,
                                  dropout_rate,
                                  bottleneck=bottleneck,
                                  name=f'ip{i}')

        ip_neighs.append(ip_neigh)
        ds_neighs.append(ds_neigh)
        all_features.append(features)
        # down sample conv
        filters *= 2
        features = conv_block(features,
                              out_cloud.model_indices,
                              filters,
                              ds_neigh,
                              dropout_rate,
                              name=f'ds{i}')
        in_cloud = out_cloud

    # add object class embedding if given
    if object_class is not None:
        embedding = tf.keras.layers.Embedding(num_object_classes, filters)(
            tf.gather(object_class, in_cloud.model_structure.value_rowids))

        features = features + embedding

    # final in-place
    ip_neigh = out_cloud.query(radius * 2, edge_features_fn, weight_fn, k=k0)
    features = identity_block(features,
                              ip_neigh,
                              dropout_rate,
                              bottleneck=bottleneck,
                              name=f'ip{num_levels}')

    # bubble back up
    for i in range(num_levels - 1, -1, -1):
        filters //= 2
        # up sample
        features = up_sample_block(features,
                                   all_features.pop(),
                                   ds_neighs.pop().transpose(),
                                   dropout_rate,
                                   name=f'us{i}')
        # in place
        features = identity_block(features,
                                  ip_neighs.pop(),
                                  dropout_rate,
                                  bottleneck,
                                  name=f'us-ip{i}')

    assert (len(ip_neighs) == 0)
    assert (len(ds_neighs) == 0)
    assert (len(all_features) == 0)

    logits = layers.Dense(num_classes, activation=None, name='logits')(features)

    if bucket_size:
        logits = logits[:cloud0.valid_size]

    # mask out invalid classes
    if object_class is not None:
        mask = tf.gather(
            tf.constant(logits_mask, dtype=tf.bool),
            tf.gather(object_class, cloud0.model_structure.value_rowids))

        logits = mask_logits(logits, mask)

    def cache_and_batch(x, name):
        x = mg.cache(x, name=name)
        with mg.pre_batch_context():
            x = ragged_layers.pre_batch_ragged(x)
        x = mg.batch(x, name=name)
        with mg.post_batch_context():
            x = ragged_layers.flat_values(x)
        return x

    labels = cache_and_batch(labels, 'labels')
    if weights is None:
        return logits, labels
    else:
        weights = cache_and_batch(weights, 'weights')
        return logits, labels, weights
