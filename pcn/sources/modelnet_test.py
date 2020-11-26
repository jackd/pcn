import numpy as np
import tensorflow as tf
from absl.testing import parameterized

from pcn.sources.modelnet import ModelnetSource


class ModelnetTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters("train[:90%]", "train[90%:]", "train", "test")
    def test_reproducible(self, split):
        def get_data(split):
            kwargs = dict(seed=0)
            if split.startswith("train"):
                kwargs.update(
                    dict(
                        jitter_stddev=0.01,
                        jitter_clip=0.02,
                        angle_stddev=0.06,
                        angle_clip=0.18,
                        uniform_scale_range=(0.8, 1.25),
                        rotate_scheme="none",
                        drop_prob_limits=(0, 0.875),
                    )
                )
            source = ModelnetSource(split, **kwargs)
            return tuple(
                tuple(x.numpy() for x in tf.nest.flatten(el, expand_composites=True))
                for el in source.dataset
            )

        d0 = get_data(split)
        d1 = get_data(split)
        tf.nest.map_structure(np.testing.assert_equal, d0, d1)


if __name__ == "__main__":
    tf.test.main()
