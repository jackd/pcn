from typing import Sequence
import functools
import numpy as np
import tensorflow as tf
import gin
from kblocks.framework.sources import TfdsSource
from kblocks.extras.losses.ciou import ContinuousMeanIouLoss
from kblocks.extras.metrics.prob_mean_iou import ProbMeanIoU
from shape_tfds.shape.shapenet import part


def _get_logits_mask():
    logits_mask = np.zeros((part.NUM_OBJECT_CLASSES, part.NUM_PART_CLASSES),
                           dtype=np.bool)
    for i in range(part.NUM_OBJECT_CLASSES):
        for j in part.part_class_indices(i):
            logits_mask[i, j] = True
    return logits_mask


def _map_fn(points_map_fn,
            inputs,
            object_class,
            weights=None,
            weighted=False,
            return_object_class=True):
    positions = inputs['positions']
    positions = positions / tf.reduce_max(tf.linalg.norm(positions, axis=-1))
    positions, normals, labels = points_map_fn(positions, inputs['normals'],
                                               inputs['labels'])

    features = dict(positions=positions, normals=normals)
    if return_object_class:
        features['object_class'] = object_class

    if weighted:
        class_freq = tf.constant(part.POINT_CLASS_FREQ)
        weights = tf.gather(tf.reduce_mean(class_freq) / class_freq, labels)
        return features, labels, weights
    else:
        return features, labels


class BlockProbMeanIoU(tf.keras.metrics.MeanIoU):

    def __init__(self, start, stop, take_argmax=True, **kwargs):
        self.start = start
        self.stop = stop
        self.take_argmax = True
        super().__init__(num_classes=stop - start, **kwargs)

    def reset_states(self):
        """Resets all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        for v in self.variables:
            tf.keras.backend.set_value(
                v, np.zeros(shape=v.shape, dtype=v.dtype.as_numpy_dtype))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, (-1,))
        mask = tf.logical_and(y_true >= self.start, y_true < self.stop)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        if sample_weight is not None:
            sample_weight = tf.boolean_mask(sample_weight, mask)
        if self.take_argmax:
            y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true - self.start, y_pred - self.start,
                                    sample_weight)


class MetricMean(tf.keras.metrics.Metric):

    def __init__(self, metrics: Sequence[tf.keras.metrics.Metric], **kwargs):
        self.__metrics = tuple(metrics)
        super().__init__(**kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def reset_states(self):
        pass

    def result(self):
        results = [m.result() for m in self.__metrics]
        return tf.reduce_mean(results)


@gin.configurable(module='pcn')
def compile_shapenet_part(model, optimizer, loss=None, metrics=None):
    _, names = part.load_synset_ids()
    if loss is None:
        loss = ContinuousMeanIouLoss(from_logits=True)
    if metrics is None:
        metrics = [
            BlockProbMeanIoU(part.LABEL_SPLITS[i],
                             part.LABEL_SPLITS[i + 1],
                             name=f'{names[sid][0]}-miou')
            for i, sid in enumerate(part.PART_SYNSET_IDS)
        ]
        metrics.append(MetricMean(tuple(metrics), name='_per-class-miou'))
        metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name='acc'))
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


@gin.configurable(module='pcn')
def compile_shapenet_part_single(model, optimizer, loss=None, metrics=None):
    if loss is None:
        # loss = ContinuousMeanIouLoss(from_logits=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if metrics is None:
        metrics = [
            ProbMeanIoU(num_classes=model.outputs[0].shape[-1]),
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
        ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


@gin.configurable(module='pcn.sources')
class ShapenetPartSource(TfdsSource):

    def __init__(self,
                 train_map_fn,
                 validation_map_fn,
                 weighted=False,
                 **kwargs):
        builder = part.ShapenetPart2017()
        self._train_map_fn = train_map_fn
        self._validation_map_fn = validation_map_fn
        self._weighted = weighted
        meta = dict(logits_mask=_get_logits_mask(),)
        super().__init__(builder=builder, meta=meta, **kwargs)

    def get_dataset(self, split) -> tf.data.Dataset:
        dataset = super().get_dataset(split)
        coords_map_fn = (self._train_map_fn
                         if split == 'train' else self._validation_map_fn)
        map_fn = functools.partial(_map_fn,
                                   coords_map_fn,
                                   weighted=self._weighted)
        return dataset.map(map_fn, tf.data.experimental.AUTOTUNE)


@gin.configurable(module='pcn')
def single_problem_id(synset, rotate=True, subvariant=None):
    if rotate:
        out = f'part-single-{synset}'
    else:
        out = f'part-single-no-rot-{synset}'
    if subvariant is not None:
        out = f'{out}-{subvariant}'
    return out


@gin.configurable(module='pcn.sources')
class ShapenetPartSingleSource(TfdsSource):

    def __init__(self,
                 synset,
                 train_map_fn,
                 validation_map_fn,
                 weighted=False,
                 **kwargs):
        builder = part.ShapenetPart2017(config=part.ShapenetPart2017Config(
            synset=synset))
        meta = dict(
            num_classes=builder.info.features['cloud']['labels'].num_classes)
        self._train_map_fn = train_map_fn
        self._validation_map_fn = validation_map_fn
        self._weighted = weighted
        super().__init__(builder=builder, meta=meta, **kwargs)

    def get_dataset(self, split) -> tf.data.Dataset:
        dataset = super().get_dataset(split)
        coords_map_fn = (self._train_map_fn
                         if split == 'train' else self._validation_map_fn)
        map_fn = functools.partial(_map_fn,
                                   coords_map_fn,
                                   weighted=self._weighted,
                                   return_object_class=False)
        return dataset.map(map_fn, tf.data.experimental.AUTOTUNE)


def vis_source(source: TfdsSource, split='train'):
    # import trimesh
    from mayavi import mlab
    dataset = source.get_dataset(split)
    for args in dataset:
        features, labels = args[:2]
        labels = labels.numpy()
        coords = features['positions'].numpy()
        normals = features['normals'].numpy()
        print(np.max(np.linalg.norm(coords, axis=-1)))

        print(features['object_class'].numpy())
        mlab.points3d(*coords.T, labels, scale_mode='none', scale_factor=0.02)
        mlab.quiver3d(*coords.T, *normals.T, scale_factor=0.02)
        mlab.show()

        # COLORS = [
        #     [0, 0, 255],
        #     [0, 255, 0],
        #     [0, 255, 255],
        #     [255, 0, 0],
        #     [255, 0, 255],
        #     [255, 255, 0],
        # ]

        # sorted_labels = sorted(set(labels))
        # c = {s: i for i, s in enumerate(sorted_labels)}
        # colors = np.array([COLORS[c[l]] for l in labels])
        # pc = trimesh.PointCloud(coords, colors=colors)
        # scene = pc.scene()
        # # add normals as black dots
        # nv = coords.shape[0]
        # eps = np.random.uniform(size=coords.shape).astype(np.float32)
        # eps = eps - np.sum(eps * normals, axis=-1, keepdims=True) * eps
        # eps *= 0.01 / np.linalg.norm(eps, axis=-1, keepdims=True)
        # vertices = np.concatenate(
        #     [coords - eps, coords + 0.02 * normals, coords + eps], axis=0)
        # vi = np.arange(nv, dtype=np.int64)
        # faces = np.stack([vi, vi + nv, vi + 2 * nv], axis=-1)
        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        # scene.add_geometry(mesh)

        # # scene.add_geometry(trimesh.PointCloud(coords + normals * 0.02))
        # scene.show()


if __name__ == '__main__':
    import functools
    from pcn.augment import augment_segmentation
    train_map_fn = functools.partial(augment_segmentation,
                                     up_dim=1,
                                     rotate_scheme='pca-xy',
                                     shuffle=True,
                                     drop_prob_limits=(0.5, 0.6),
                                     maybe_reflect_x=True)
    source = ShapenetPartSource(train_map_fn, train_map_fn)
    vis_source(source)
