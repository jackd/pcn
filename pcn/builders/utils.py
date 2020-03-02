import gin
import tensorflow as tf

from kblocks.ops import polynomials
from kblocks.keras import layers


@gin.configurable(module='pcn.builders')
def polynomial_edge_features(coords, max_order=2):
    return polynomials.get_nd_polynomials(coords,
                                          max_order=max_order,
                                          unstack_axis=-1,
                                          stack_axis=0)


@gin.configurable(module='pcn.builder')
def hat_weight(dists):
    return tf.nn.relu(1 - dists)


@gin.configurable(module='pcn.builders')
def simple_mlp(features, num_classes, units=(256,), dropout_rate=0.5):
    features = layers.Dropout(dropout_rate)(features)
    for u in units:
        features = layers.Dense(u)(features)
        features = tf.keras.layers.Activation('relu')(features)
        features = layers.BatchNormalization()(features)
        features = layers.Dropout(dropout_rate)(features)
    logits = layers.Dense(num_classes, name='logits')(features)
    return logits
