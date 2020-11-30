import functools
import os

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

import shape_tfds.shape.modelnet  # pylint: disable=unused-import
from kblocks.data import dense_to_ragged_batch, snapshot
from kblocks.extras.callbacks import (
    EarlyStoppingModule,
    PrintLogger,
    ReduceLROnPlateauModule,
)
from kblocks.models import compiled
from kblocks.trainables import trainable_fit
from kblocks.trainables.meta_models import build_meta_model_trainable
from pcn.augment import augment
from pcn.builders.resnet import resnet

BackupAndRestore = tf.keras.callbacks.experimental.BackupAndRestore
AUTOTUNE = tf.data.experimental.AUTOTUNE
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
run = 0
problem_dir = "/tmp/pcn/pn2/large"
experiment_dir = os.path.join(problem_dir, f"run-{run:04d}")
num_cache_repeats = 32
epochs = 500
reduce_lr_patience = 20
reduce_lr_factor = 0.2
early_stopping_patience = 41
shuffle_buffer = 1024
seed = 0
data_seed = 0
validation_freq = 5

tf.random.set_seed(seed)  # used for weight initialization


def map_func(**aug_kwargs):
    def func(cloud, labels, sample_weight=None):
        coords = augment(cloud["positions"], up_dim=1, **aug_kwargs)
        return tf.keras.utils.pack_x_y_sample_weight(coords, labels, sample_weight)

    return func


train_map_func = map_func(
    jitter_stddev=0.01,
    jitter_clip=0.02,
    angle_stddev=0.06,
    angle_clip=0.18,
    uniform_scale_range=(0.8, 1.25),
    drop_prob_limits=(0, 0.875),
    shuffle=True,
)

validation_map_func = map_func(drop_prob_limits=(0, 0))

train_ds = tfds.load(
    name="pointnet2_h5",
    shuffle_files=True,
    split="train",
    as_supervised=True,
    read_config=tfds.core.utils.read_config.ReadConfig(
        shuffle_seed=0, shuffle_reshuffle_each_iteration=True
    ),
    download_and_prepare_kwargs=dict(
        download_config=tfds.core.download.DownloadConfig(verify_ssl=False)
    ),
)

validation_ds = tfds.load(
    name="pointnet2_h5", shuffle_files=False, split="test", as_supervised=True
)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [
    tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
    tf.keras.metrics.SparseCategoricalAccuracy(),
]
optimizer = tf.keras.optimizers.Adam()


monitor_kwargs = dict(monitor="sparse_categorical_accuracy", mode="max")
model_callbacks = [
    ReduceLROnPlateauModule(
        patience=reduce_lr_patience, factor=reduce_lr_factor, **monitor_kwargs
    ),
    EarlyStoppingModule(patience=early_stopping_patience, **monitor_kwargs),
]

trainable = build_meta_model_trainable(
    meta_model_func=functools.partial(resnet, num_classes=40),
    train_dataset=train_ds,
    validation_dataset=validation_ds,
    batcher=dense_to_ragged_batch(batch_size=batch_size),
    shuffle_buffer=shuffle_buffer,
    compiler=functools.partial(
        compiled, loss=loss, metrics=metrics, optimizer=optimizer
    ),
    cache_factory=functools.partial(snapshot, compression="GZIP"),
    cache_dir=os.path.join(problem_dir, "cache"),
    cache_repeats=num_cache_repeats,
    train_augment_func=train_map_func,
    validation_augment_func=validation_map_func,
    callbacks=model_callbacks,
    seed=data_seed,
)

logging_callbacks = [
    PrintLogger(),
    tf.keras.callbacks.TensorBoard(os.path.join(experiment_dir, "logs")),
    BackupAndRestore(os.path.join(experiment_dir, "backup")),
]

fit = trainable_fit(
    trainable=trainable,
    callbacks=logging_callbacks,
    epochs=epochs,
    validation_freq=validation_freq,
    experiment_dir=experiment_dir,
)

fit.run()
