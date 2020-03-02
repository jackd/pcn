import functools
import tensorflow as tf
from kblocks.framework.sources import TfdsSource
import gin


def _map_fn(coords_map_fn, inputs, labels, weights=None):
    coords = coords_map_fn(inputs['positions'])
    return (coords, labels) if weights is None else (coords, labels, weights)


@gin.configurable(module='pcn.sources')
class ModelnetSource(TfdsSource):

    def __init__(self,
                 train_map_fn,
                 validation_map_fn,
                 builder=None,
                 num_points=1024,
                 return_normals=False,
                 split_map=None,
                 **kwargs):
        if builder is None:
            from shape_tfds.shape.modelnet.pointnet2 import Pointnet2H5
            builder = Pointnet2H5()
            if split_map is None:
                split_map = {'validation': 'test'}
        self._train_map_fn = train_map_fn
        self._validation_map_fn = validation_map_fn
        super().__init__(builder=builder, split_map=split_map, **kwargs)

    def get_dataset(self, split) -> tf.data.Dataset:
        dataset = super(ModelnetSource, self).get_dataset(split)
        coords_map_fn = (self._train_map_fn
                         if split == 'train' else self._validation_map_fn)
        map_fn = functools.partial(_map_fn, coords_map_fn)
        return dataset.map(map_fn, tf.data.experimental.AUTOTUNE)
