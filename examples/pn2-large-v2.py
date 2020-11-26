"""Simple non-configurable script demonstrating modelnet40 cls with large model."""
import functools
import os

import gin
import tensorflow as tf

import shape_tfds.shape.modelnet  # pylint: disable=unused-import
from kblocks.data.transforms import (
    Cache,
    DenseToRaggedBatch,
    RepeatedTFRecordsCache,
    TFRecordsFactory,
)
from kblocks.extras import callbacks as cb
from kblocks.extras.callbacks import LearningRateLogger, PrintLogger
from kblocks.trainables.fit import fit
from kblocks.trainables.meta_models import build_meta_model_trainable
from pcn.builders.resnet import resnet
from pcn.sources.modelnet import ModelnetSource

os.environ["TF_DETERMINISTIC_OPS"] = "1"

gin.parse_config(
    """\
import kblocks.keras.layers
import kblocks.keras.regularizers
import pcn.layers.conv

tf.keras.layers.Dense.kernel_regularizer = %l2_reg
pcn.layers.ConvolutionBase.kernel_regularizer = %l2_reg

l2_reg = @tf.keras.regularizers.l2()
tf.keras.regularizers.l2.l2 = 4e-5
"""
)

batch_size = 128
save_dir = "/tmp/pcn/pn2-v2/large"
num_cache_repeats = 32
epochs = 500
reduce_lr_patience = 20
reduce_lr_factor = 0.2
early_stopping_patience = 41
shuffle_buffer = 4096
run = 13
seed = 3
data_seed = 0
validation_freq = 5

tf.random.set_seed(seed)  # used for weight initialization
tf.random.set_global_generator(tf.random.Generator.from_seed(seed))

train_source = ModelnetSource(
    split="train",
    shuffle_files=True,
    jitter_stddev=0.01,
    jitter_clip=0.02,
    angle_stddev=0.06,
    angle_clip=0.18,
    uniform_scale_range=(0.8, 1.25),
    drop_prob_limits=(0, 0.875),
    shuffle=True,
    seed=data_seed,
)
validation_source = ModelnetSource(split="test", drop_prob_limits=(0, 0))

train_cache_dir = os.path.join(save_dir, "cache", "train")
validation_cache_dir = os.path.join(save_dir, "cache", "validation")
monitor_kwargs = dict(monitor="sparse_categorical_accuracy", mode="max")

trainable = build_meta_model_trainable(
    meta_model_fn=functools.partial(resnet, num_classes=40),
    batcher=DenseToRaggedBatch(batch_size),
    train_source=train_source,
    validation_source=validation_source,
    shuffle_buffer=shuffle_buffer,
    train_cache=RepeatedTFRecordsCache(
        path=train_cache_dir, num_repeats=num_cache_repeats, compression="GZIP"
    ),
    validation_cache=Cache(
        cache_dir=validation_cache_dir, factory=TFRecordsFactory(compression="GZIP")
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ],
    optimizer=tf.keras.optimizers.Adam(),
    callbacks=(
        cb.ReduceLROnPlateauModule(
            patience=reduce_lr_patience, factor=reduce_lr_factor, **monitor_kwargs
        ),
        cb.EarlyStoppingModule(patience=early_stopping_patience, **monitor_kwargs),
    ),
)

extra_callbacks = (
    LearningRateLogger(),
    PrintLogger(),
    tf.keras.callbacks.TensorBoard(os.path.join(save_dir, f"run-{run:02d}")),
)

fit(
    trainable, epochs=epochs, callbacks=extra_callbacks, validation_freq=validation_freq
)

# 7: seems to get good results?
# 8: with cb callbacks: still good, reverting
# 9: use set_global_generator rather than get / reset_from_seed: 90.4
# 10: 90.3
# 11: same as 10, 89.7??
# 12: same as 10, deleted cache, 90.7
# 13: same as 10
# 14: same as 13, 91.1?
