from typing import Callable, Optional

import gin
import tensorflow as tf
import trimesh


@gin.configurable(module="pcn.vis")
def vis_cloud(dataset: tf.data.Dataset, augment_func: Optional[Callable] = None):
    if augment_func is not None:
        dataset = dataset.map(augment_func)
    for coords, label in dataset:
        if isinstance(coords, dict):
            coords = coords["positions"]
        coords = coords.numpy()
        pc = trimesh.PointCloud(coords)
        print(f"label = {label.numpy()}")
        print(f"max_r = {tf.reduce_max(tf.linalg.norm(coords, axis=-1)).numpy()}")
        pc.show()
