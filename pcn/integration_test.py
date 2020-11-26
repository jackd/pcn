import functools
import os
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf

from kblocks.data import transforms
from kblocks.trainables.meta_models import build_meta_model_trainable
from pcn.builders.resnet import resnet
from pcn.sources import ModelnetSource

os.environ["TF_DETERMINISTIC_OPS"] = "1"


def assert_all_equal(struct0, struct1):
    tf.nest.map_structure(np.testing.assert_equal, struct0, struct1)


def assert_all_close(struct0, struct1, **kwargs):
    tf.nest.map_structure(
        functools.partial(np.testing.assert_allclose, **kwargs), struct0, struct1
    )


class IntegrationTest(tf.test.TestCase):
    def test_modelnet_resnet_self_consistent(self):
        num_repeats = 2
        batch_size = 16

        def get_weights(factory: transforms.CacheFactory):
            # global seed affect initialization
            tf.random.set_seed(0)
            # global generator seed affects split RNG initial state.
            tf.random.set_global_generator(tf.random.Generator.from_seed(0))
            with TemporaryDirectory() as tmp_dir:
                if factory is None:
                    train_cache = None
                else:
                    train_cache = transforms.RandomRepeatedCache(
                        num_repeats=num_repeats,
                        path=os.path.join(tmp_dir, "cache"),
                        cache_factory=factory,
                    )
                del tmp_dir
                build_fn = functools.partial(resnet, filters0=2)

                train_source = ModelnetSource(
                    split="train",
                    shuffle=True,
                    jitter_stddev=0.01,
                    jitter_clip=0.02,
                    angle_stddev=0.06,
                    angle_clip=0.18,
                    uniform_scale_range=(0.8, 1.25),
                    rotate_scheme="none",
                    drop_prob_limits=(0, 0.875),
                    seed=0,
                )

                trainable = build_meta_model_trainable(
                    build_fn,
                    batcher=transforms.DenseToRaggedBatch(batch_size),
                    train_source=train_source,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
                    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
                    optimizer=tf.keras.optimizers.Adam(),
                    shuffle_buffer=1024,
                    pre_emptible=False,
                    train_cache=train_cache,
                )
                model = trainable.model
                weights = [w.numpy() for w in model.weights]
                data = tuple(
                    tuple(
                        x.numpy() for x in tf.nest.flatten(el, expand_composites=True)
                    )
                    for el in trainable.train_source.dataset
                )
                # we make no effort to fit the data, because sparse_dense matmuls are
                # not perfectly deterministic
            return data, weights

        def test_impl(impl):
            data, w = get_weights(impl)
            data_, w_ = get_weights(impl)
            assert_all_equal(data, data_)
            assert_all_equal(w, w_)

        test_impl(None)
        factories = (
            transforms.CacheFactory(),
            transforms.SaveLoadFactory(),
            transforms.SnapshotFactory(),
            transforms.TFRecordsFactory(),
        )
        for factory in factories:
            test_impl(factory)


if __name__ == "__main__":
    tf.test.main()
