import functools

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

import meta_model.pipeline as pl
import pcn.components as comp
from meta_model.batchers import RaggedBatcher
from pcn.builders.utils import hat_weight, polynomial_edge_features


def batch(dataset: tf.data.Dataset, batch_size: int):
    return dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size))


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
    labels = pl.batch(pl.cache(labels))
    cloud = comp.Cloud(coords)
    radius = r0
    ip_neigh = cloud.query(
        radius, polynomial_edge_features, weight_fn, k0, normalize=normalize
    )
    features = ip_neigh.convolve(features, filters=3, activation="relu")
    return features, labels


def ip_pool_build(
    coords,
    labels,
    r0=0.1125,
    k0=32,
    num_classes=10,
    weight_fn=hat_weight,
    normalize=True,
):
    # in-place pooling
    del num_classes
    features = None
    labels = pl.batch(pl.cache(labels))
    cloud = comp.Cloud(coords)
    radius = r0
    ip_neigh = cloud.query(
        radius, polynomial_edge_features, weight_fn, k0, normalize=normalize
    )
    features = ip_neigh.convolve(features, filters=3, activation="relu")
    features = ip_neigh.convolve(features, filters=2, activation="relu")
    features = tf.math.unsorted_segment_max(
        features, cloud.model_structure.value_rowids, cloud.model_structure.nrows
    )
    return features, labels


def ip_ds_pool_build(
    coords,
    labels,
    r0=0.1125,
    k0=32,
    num_classes=10,
    weight_fn=hat_weight,
    normalize=True,
):
    # in-place down-sample pooling
    in_cloud = comp.Cloud(coords)
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
    features = tf.math.unsorted_segment_max(
        features,
        out_cloud.model_structure.value_rowids,
        out_cloud.model_structure.nrows,
    )
    logits = tf.keras.layers.Dense(num_classes)(features)
    return logits, pl.batch(pl.cache(labels))


normalize_options = ((False,), (True,))


class ComponentsTest(tf.test.TestCase, parameterized.TestCase):
    def _test_build_fn(self, build_fn):
        batch_size = 3
        num_points = 1024
        num_classes = 2

        coords = tf.random.normal(shape=(batch_size, num_points, 3))
        coords = coords / tf.linalg.norm(coords, axis=-1, keepdims=True)

        dataset = tf.data.Dataset.from_tensor_slices(
            (coords, tf.range(batch_size) % num_classes)
        )

        batcher1 = RaggedBatcher(batch_size=1)
        pipeline1, model = pl.build_pipelined_model(
            build_fn, dataset.element_spec, batcher=batcher1
        )

        ds = dataset.map(pipeline1.pre_cache_map_func()).map(
            pipeline1.pre_batch_map_func()
        )
        ds1 = batcher1(ds).map(pipeline1.post_batch_map_func())
        val1 = []
        for features, _ in ds1.take(batch_size):
            val = model(features)
            val1.append(self.evaluate(val))
        val1 = np.concatenate(val1, axis=0)

        batcher = RaggedBatcher(batch_size=batch_size)
        pipeline, _ = pl.build_pipelined_model(
            build_fn, dataset.element_spec, batcher=batcher
        )

        ds = dataset.map(pipeline.pre_cache_map_func()).map(
            pipeline.pre_batch_map_func()
        )
        ds_batched = batcher(ds).map(pipeline.post_batch_map_func())
        val_batched = None
        for features, _ in ds_batched:
            assert val_batched is None
            val_batched = self.evaluate(model(features))

        np.testing.assert_allclose(val1, val_batched, atol=1e-4)

    @parameterized.parameters(*normalize_options)
    def test_ip(self, normalize):
        self._test_build_fn(functools.partial(ip_build, normalize=normalize))

    @parameterized.parameters(*normalize_options)
    def test_ip_pool(self, normalize):
        self._test_build_fn(functools.partial(ip_pool_build, normalize=normalize))

    @parameterized.parameters(*normalize_options)
    def test_ip_ds_pool(self, normalize):
        self._test_build_fn(functools.partial(ip_ds_pool_build, normalize=normalize))


if __name__ == "__main__":
    tf.test.main()
