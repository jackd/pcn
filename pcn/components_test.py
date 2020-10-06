import functools

import numpy as np
import tensorflow as tf

import multi_graph as mg
import pcn.components as comp
from pcn.builders.utils import hat_weight, polynomial_edge_features


def ip_build(
    coords,
    labels,
    r0=0.1125,
    k0=32,
    num_classes=10,
    weight_fn=hat_weight,
    normalize=True,
):
    # in-place
    del num_classes
    features = None
    labels = mg.batch(mg.cache(labels))
    cloud = comp.Cloud(coords, bucket_size=True)
    radius = r0
    ip_neigh = cloud.query(
        radius, polynomial_edge_features, weight_fn, k0, normalize=normalize
    )
    features = ip_neigh.convolve(features, filters=3, activation="relu")
    return features, labels


def ip_pool_build(
    coords,
    labels,
    bucket_size=False,
    r0=0.1125,
    k0=32,
    num_classes=10,
    weight_fn=hat_weight,
    normalize=True,
):
    # in-place pooling
    del num_classes
    features = None
    labels = mg.batch(mg.cache(labels))
    cloud = comp.Cloud(coords, bucket_size=bucket_size)
    radius = r0
    ip_neigh = cloud.query(
        radius, polynomial_edge_features, weight_fn, k0, normalize=normalize
    )
    features = ip_neigh.convolve(features, filters=3, activation="relu")
    features = ip_neigh.convolve(features, filters=2, activation="relu")
    valid_size = cloud.valid_size
    if valid_size is not None:
        mask = tf.expand_dims(
            tf.sequence_mask(valid_size, tf.shape(features)[0]), axis=-1
        )
        features = tf.where(mask, features, tf.zeros_like(features))
    features = tf.math.unsorted_segment_max(
        features, cloud.model_structure.value_rowids, cloud.model_structure.nrows
    )
    return features, labels


def ip_ds_pool_build(
    coords,
    labels,
    bucket_size=False,
    r0=0.1125,
    k0=32,
    num_classes=10,
    weight_fn=hat_weight,
    normalize=True,
):
    # in-place down-sample pooling
    in_cloud = comp.Cloud(coords, bucket_size=bucket_size)
    radius = r0

    out_cloud, ip_neigh, ds_neigh = in_cloud.sample_query(
        in_place_radius=radius,
        in_place_k=k0,
        down_sample_radius=radius * np.sqrt(2),
        down_sample_k=k0 * 2,
        edge_features_fn=polynomial_edge_features,
        weight_fn=weight_fn,
        normalize=normalize,
    )

    features = None
    features = ip_neigh.convolve(features, filters=5, activation="relu")
    features = ip_neigh.convolve(features, filters=7, activation="relu")
    features = ds_neigh.convolve(features, filters=11, activation="relu")
    valid_size = out_cloud.valid_size
    if valid_size is not None:
        mask = tf.expand_dims(
            tf.sequence_mask(valid_size, tf.shape(features)[0]), axis=-1
        )
        features = tf.where(mask, features, tf.zeros_like(features))
    features = tf.math.unsorted_segment_max(
        features,
        out_cloud.model_structure.value_rowids,
        out_cloud.model_structure.nrows,
    )
    logits = tf.keras.layers.Dense(num_classes)(features)
    return logits, mg.batch(mg.cache(labels))


class ComponentsTest(tf.test.TestCase):
    def _test_build_fn(self, build_fn):
        batch_size = 4
        num_points = 1024
        num_classes = 2

        coords = tf.random.normal(shape=(batch_size, num_points, 3))
        coords = coords / tf.linalg.norm(coords, axis=-1, keepdims=True)

        dataset = tf.data.Dataset.from_tensor_slices(
            (coords, tf.range(batch_size) % num_classes)
        )

        built = mg.build_multi_graph(build_fn, dataset.element_spec)

        ds = dataset.map(built.pre_cache_map).map(built.pre_batch_map)
        ds1 = ds.batch(1).map(built.post_batch_map)
        val1 = []
        model = built.trained_model
        for features, _ in ds1:
            val = model(features)
            val1.append(self.evaluate(val))
        val1 = np.concatenate(val1, axis=0)

        ds_batched = ds.batch(batch_size).map(built.post_batch_map)
        val_batched = None
        for features, _ in ds_batched:
            assert val_batched is None
            val_batched = self.evaluate(model(features))

        np.testing.assert_allclose(val1, val_batched, atol=1e-4)

    def test_ip(self):
        for normalize in (False, True):
            self._test_build_fn(functools.partial(ip_build, normalize=normalize))

    def test_ip_pool(self):
        for bucket_size in (False, True):
            for normalize in (False, True):
                self._test_build_fn(
                    functools.partial(
                        ip_pool_build, normalize=normalize, bucket_size=bucket_size
                    )
                )

    def test_ip_ds_pool(self):
        for bucket_size in (False, True):
            for normalize in (False, True):
                self._test_build_fn(
                    functools.partial(
                        ip_ds_pool_build, normalize=normalize, bucket_size=bucket_size
                    )
                )


if __name__ == "__main__":
    tf.test.main()
