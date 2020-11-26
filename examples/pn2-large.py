"""Simple non-configurable script demonstrating modelnet40 cls with large model."""
import functools
import os

import gin
import tensorflow as tf
import tensorflow_datasets as tfds

import shape_tfds.shape.modelnet  # pylint: disable=unused-import
import tfrng
from kblocks.data.transforms import Cache, RepeatedTFRecordsCache, TFRecordsFactory
from kblocks.extras.callbacks import LearningRateLogger, PrintLogger
from meta_model import pipeline as pl
from pcn.augment import augment
from pcn.builders.resnet import resnet

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
save_dir = "/tmp/pcn/pn2/large"
num_cache_repeats = 32
epochs = 500
reduce_lr_patience = 20
reduce_lr_factor = 0.2
early_stopping_patience = 41
shuffle_buffer = 4096
run = 12
seed = 3
data_seed = 0
validation_freq = 5

tf.random.set_seed(seed)  # used for weight initialization

# used in data augmentation
tf.random.set_global_generator(tf.random.Generator.from_seed(data_seed))

AUTOTUNE = tf.data.experimental.AUTOTUNE

builder = tfds.builder(name="pointnet2_h5")
builder.download_and_prepare(
    download_config=tfds.core.download.DownloadConfig(verify_ssl=False)
)


def map_func(**aug_kwargs):
    def func(cloud, labels, sample_weight=None):
        coords = augment(cloud["positions"], up_dim=1, **aug_kwargs)
        return tf.keras.utils.pack_x_y_sample_weight(coords, labels, sample_weight)

    return func


train_ds = builder.as_dataset(
    shuffle_files=True,
    split="train",
    as_supervised=True,
    read_config=tfds.core.utils.read_config.ReadConfig(
        shuffle_seed=0, shuffle_reshuffle_each_iteration=True
    ),
).apply(
    tfrng.data.stateless_map(
        map_func(
            jitter_stddev=0.01,
            jitter_clip=0.02,
            angle_stddev=0.06,
            angle_clip=0.18,
            uniform_scale_range=(0.8, 1.25),
            drop_prob_limits=(0, 0.875),
            shuffle=True,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        seed=data_seed,
    )
)
# make validation coords ragged
validation_ds = builder.as_dataset(
    shuffle_files=False, split="test", as_supervised=True
).map(map_func(drop_prob_limits=(0, 0)))

meta_model_fn = functools.partial(resnet, num_classes=40)
batcher = tf.data.experimental.dense_to_ragged_batch(batch_size)

pipeline, model = pl.build_pipelined_model(
    meta_model_fn, train_ds.element_spec, batcher
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ],
    optimizer=tf.keras.optimizers.Adam(),
)

train_examples = len(train_ds)

steps_per_epoch = train_examples // batch_size
if train_examples % batch_size:
    steps_per_epoch += 1

train_cache_dir = os.path.join(save_dir, "cache", "train")
train_ds = (
    train_ds.map(pipeline.pre_cache_map_func(True), num_parallel_calls=1)
    .apply(
        RepeatedTFRecordsCache(path=train_cache_dir, num_repeats=32, compression="GZIP")
    )
    .repeat()
    .shuffle(shuffle_buffer)
    .map(pipeline.pre_batch_map_func(True), AUTOTUNE)
    .apply(batcher)
    .map(pipeline.post_batch_map_func(True))
    .prefetch(AUTOTUNE)
)

validation_cache_dir = os.path.join(save_dir, "cache", "validation")
validation_ds = (
    validation_ds.map(pipeline.pre_cache_map_func(False), AUTOTUNE)
    .map(pipeline.pre_batch_map_func(False), AUTOTUNE)
    .apply(batcher)
    .map(pipeline.post_batch_map_func(False))
    # cache at end
    # .apply(save_load(validation_cache_dir, "GZIP"))
    .apply(
        Cache(
            cache_dir=validation_cache_dir, factory=TFRecordsFactory(compression="GZIP")
        )
    )
    .prefetch(AUTOTUNE)
)


monitor_kwargs = dict(monitor="sparse_categorical_accuracy", mode="max")
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        patience=reduce_lr_patience, factor=reduce_lr_factor, **monitor_kwargs
    ),
    tf.keras.callbacks.EarlyStopping(
        patience=early_stopping_patience, **monitor_kwargs
    ),
    LearningRateLogger(),
    PrintLogger(),
    tf.keras.callbacks.TensorBoard(os.path.join(save_dir, f"run-{run:02d}")),
]

model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_ds,
    validation_freq=validation_freq,
    callbacks=callbacks,
)

# 7: RepeatedTFRecordsCache, 89.9%
# 8: repeated TFRecordsFactory, 90.5%
# 9: RepeatedTFRecordsCache without shuffle, 90.8%
# 10: clear cache
# 12: clear cache, tfrng transform, stateless_map, tfds read_config with shuffle seed
