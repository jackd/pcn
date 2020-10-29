import functools
from typing import Optional

import gin
import tensorflow as tf

import wtftf
from kblocks.framework.sources import TfdsSource
from shape_tfds.shape.modelnet.pointnet2 import Pointnet2H5


def _map_fn(
    coords_map_fn,
    inputs,
    labels,
    weights=None,
    rng: Optional[tf.random.Generator] = None,
):
    pos = inputs["positions"]
    if rng is not None:
        with wtftf.random.global_generator_context(rng):
            coords = coords_map_fn(pos)
    else:
        coords = coords_map_fn(pos)
    return tf.keras.utils.pack_x_y_sample_weight(coords, labels, weights)


@gin.configurable(module="pcn.sources")
class ModelnetSource(TfdsSource):
    def __init__(
        self,
        train_map_fn,
        validation_map_fn,
        builder=None,
        split_map=None,
        seed: int = 0,
        num_parallel_calls: int = 1,  # 1 makes things deterministic
        deterministic: bool = True,
        **kwargs
    ):
        if builder is None:
            builder = Pointnet2H5()
            if split_map is None:
                split_map = {"validation": "test"}
        self._train_map_fn = train_map_fn
        self._validation_map_fn = validation_map_fn
        self._train_rng = tf.random.Generator.from_seed(seed)
        self._num_parallel_calls = num_parallel_calls
        self._deterministic = deterministic
        super().__init__(
            builder=builder,
            split_map=split_map,
            modules={"rng": self._train_rng},
            **kwargs,
        )

    def get_dataset(self, split) -> tf.data.Dataset:
        dataset = super().get_dataset(split)
        coords_map_fn = (
            self._train_map_fn if split == "train" else self._validation_map_fn
        )
        map_fn = functools.partial(_map_fn, coords_map_fn, rng=self._train_rng)
        return dataset.map(
            map_fn, self._num_parallel_calls, deterministic=self._deterministic
        )


def vis_source(source: ModelnetSource, split="train"):
    import trimesh  # pylint:disable=import-outside-toplevel

    dataset = source.get_dataset(split)
    for coords, label in dataset:
        coords = coords.numpy()
        pc = trimesh.PointCloud(coords)
        print(label.numpy())
        print(tf.reduce_max(tf.linalg.norm(coords, axis=-1)).numpy())
        pc.show()


if __name__ == "__main__":
    from pcn.augment import augment

    train_map_fn = functools.partial(augment, up_dim=1, rotate_scheme="random")
    source = ModelnetSource(train_map_fn, train_map_fn)
    vis_source(source)
