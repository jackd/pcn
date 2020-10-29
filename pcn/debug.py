import gin
import tensorflow as tf
from tqdm import trange

from kblocks.framework.trainable import Trainable


@gin.configurable(module="pcn.debug")
def run_pipeline(trainable: Trainable):
    source = trainable.source
    train_ds = source.get_dataset("train")
    val_ds = source.get_dataset("validation")
    assert val_ds.cardinality() != tf.data.INFINITE_CARDINALITY
    if train_ds.cardinality() != tf.data.INFINITE_CARDINALITY:
        train_ds = train_ds.repeat()
    train_iter = iter(train_ds)
    train_steps = source.epoch_length("train")
    epoch = 0
    while True:
        epoch += 1
        print(f"Starting epoch {epoch}")
        for _ in trange(train_steps):
            _ = next(train_iter)
        for _ in val_ds:
            pass
