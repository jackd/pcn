from typing import Optional

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

import shape_tfds.shape.modelnet  # pylint: disable=unused-import
from kblocks.data import transforms
from kblocks.data.sources import DelegatingSource, TfdsSource
from kblocks.utils import delegates
from pcn.augment import augment
from pcn.serialize import register_serializable


@delegates(augment)
def map_coords(cloud, labels, weights=None, **kwargs):
    coords = cloud["positions"]
    coords = augment(coords, **kwargs)
    return tf.keras.utils.pack_x_y_sample_weight(coords, labels, weights)


@gin.register(module="pcn.sources")
@register_serializable
@delegates(map_coords)
class ModelnetSource(DelegatingSource):
    def __init__(
        self,
        split: str,
        name: str = "pointnet2_h5",
        shuffle_files: bool = False,
        seed: Optional[int] = None,
        read_config: Optional[tfds.core.utils.read_config.ReadConfig] = None,
        up_dim: int = 1,
        **kwargs,
    ):
        self._aug_kwargs = kwargs
        self._aug_kwargs["up_dim"] = up_dim
        self._tfds_kwargs = dict(
            split=split, name=name, shuffle_files=shuffle_files, read_config=read_config
        )
        self._seed = seed
        source = TfdsSource(as_supervised=True, **self._tfds_kwargs)
        source = source.apply(
            transforms.map_transform(
                map_coords, arguments=self._aug_kwargs, use_rng=True, seed=seed,
            )
        )
        super().__init__(source=source)

    def get_config(self):
        config = dict(self._aug_kwargs)
        config.update(self._tfds_kwargs)
        config["seed"] = self._seed
        return config


def vis_source(source: ModelnetSource):
    import trimesh  # pylint:disable=import-outside-toplevel

    dataset = source.dataset
    for element in dataset:
        coords, label, _ = tf.keras.utils.unpack_x_y_sample_weight(element)
        coords = coords.numpy()
        pc = trimesh.PointCloud(coords)
        print(f"label = {label.numpy()}")
        print(f"max_norm = {tf.reduce_max(tf.linalg.norm(coords, axis=-1)).numpy()}")
        pc.show()


if __name__ == "__main__":
    source = ModelnetSource("train", up_dim=1, rotate_scheme="random")
    vis_source(source)
