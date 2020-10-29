import functools
import os
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf
from absl import logging

from kblocks.extras import cache
from kblocks.framework.batchers import RaggedBatcher
from kblocks.framework.compilers import compile_classification_model
from kblocks.framework.meta_models import meta_model_trainable
from pcn.augment import augment
from pcn.builders.resnet import resnet
from pcn.sources import ModelnetSource


def assert_all_equal(struct0, struct1):
    tf.nest.map_structure(np.testing.assert_equal, struct0, struct1)


def assert_all_close(struct0, struct1, **kwargs):
    tf.nest.map_structure(
        functools.partial(np.testing.assert_allclose, **kwargs), struct0, struct1
    )


class IntegrationTest(tf.test.TestCase):
    def test_all(self):
        # tests call tf.random.set_seed
        self._test_modelnet_base_reproducible()
        self._test_modelnet_resnet_self_consistent()

    def _test_modelnet_base_reproducible(self):
        def get_data(split):
            tf.random.set_seed(0)
            shared_kwargs = dict(up_dim=1, shuffle=True)
            train_map_fn = functools.partial(
                augment,
                jitter_stddev=0.01,
                jitter_clip=0.02,
                angle_stddev=0.06,
                angle_clip=0.18,
                uniform_scale_range=(0.8, 1.25),
                rotate_scheme="none",
                drop_prob_limits=(0, 0.875),
                **shared_kwargs
            )
            val_map_fn = functools.partial(augment, **shared_kwargs)
            base_source = ModelnetSource(train_map_fn, val_map_fn)
            return tuple(
                tuple(x.numpy() for x in tf.nest.flatten(el, expand_composites=True))
                for el in base_source.get_dataset(split)
            )

        train0 = get_data("train")
        train1 = get_data("train")
        assert_all_equal(train0, train1)
        val0 = get_data("validation")
        val1 = get_data("validation")
        assert_all_equal(val0, val1)

    def _test_modelnet_resnet_self_consistent(self):
        num_repeats = 2
        batch_size = 16
        epochs = num_repeats

        def get_weights(cache_impl):
            tf.random.set_seed(0)
            with TemporaryDirectory() as tmp_dir:
                if cache_impl is None:
                    train_impl = None
                else:
                    train_impl = functools.partial(
                        cache.RepeatCacheManager,
                        num_repeats=2,
                        manager_impl=cache_impl,
                        shuffle_datasets=False,
                        take_single=False,
                    )
                cache_managers = cache.cache_managers(
                    os.path.join(tmp_dir, "cache"), train_impl, cache_impl
                )
                del tmp_dir
                build_fn = functools.partial(resnet, filters0=2)
                shared_kwargs = dict(up_dim=1, shuffle=True)
                train_map_fn = functools.partial(
                    augment,
                    jitter_stddev=0.01,
                    jitter_clip=0.02,
                    angle_stddev=0.06,
                    angle_clip=0.18,
                    uniform_scale_range=(0.8, 1.25),
                    rotate_scheme="none",
                    drop_prob_limits=(0, 0.875),
                    **shared_kwargs
                )
                val_map_fn = functools.partial(
                    augment, **shared_kwargs, drop_prob_limits=(0, 0)
                )
                base_source = ModelnetSource(
                    train_map_fn, val_map_fn, shuffle_files=False
                ).take({"train": batch_size * 10, "validation": batch_size * 2})
                batcher = RaggedBatcher(batch_size=batch_size)
                optimizer = tf.keras.optimizers.Adam()
                compiler = functools.partial(
                    compile_classification_model, optimizer=optimizer
                )

                trainable = meta_model_trainable(
                    build_fn,
                    base_source,
                    batcher,
                    compiler,
                    cache_managers=cache_managers,
                    num_parallel_calls=1,
                    shuffle_buffer=32,
                    shuffle_seed=0,
                )
                model = trainable.model
                weights0 = [w.numpy() for w in model.weights]
                trainable.fit(epochs=epochs)
                weights = [w.numpy() for w in model.weights]
                data = tuple(
                    tuple(
                        x.numpy() for x in tf.nest.flatten(el, expand_composites=True)
                    )
                    for el in trainable.source.get_dataset("train")
                )
            return data, weights0, weights

        def test_impl(impl):
            data, w0, w = get_weights(impl)
            data_, w0_, w_ = get_weights(impl)
            assert_all_close(data, data_)
            assert_all_equal(w0, w0_)
            # no idea why the following fails.
            # initial weights are the same, data is the same
            assert_all_close(w, w_)

        test_impl(None)
        test_impl(functools.partial(cache.BaseCacheManager, preprocess=False))
        test_impl(functools.partial(cache.BaseCacheManager, preprocess=True))
        test_impl(cache.TFRecordsCacheManager)
        test_impl(cache.SaveLoadManager)
        test_impl(functools.partial(cache.SnapshotManager, preprocess=False))
        if tf.version.VERSION.startswith("2.4.0-dev"):
            logging.warning(
                "Skipping SnapshotManager(preprocess=True) test - "
                "see https://github.com/tensorflow/tensorflow/issues/44278"
            )
        else:
            test_impl(functools.partial(cache.SnapshotManager, preprocess=True))


if __name__ == "__main__":
    tf.test.main()
