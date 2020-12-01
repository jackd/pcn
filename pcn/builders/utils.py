import gin
import tensorflow as tf
from kblocks.keras import layers
from kblocks.ops import polynomials


@gin.configurable(module="pcn.builders")
def polynomial_edge_features(coords, max_order=2):
    return polynomials.get_nd_polynomials(
        coords, max_order=max_order, unstack_axis=-1, stack_axis=0
    )


@gin.configurable(module="pcn.builder")
def hat_weight(dists):
    return tf.nn.relu(1 - dists)


@gin.configurable(module="pcn.builders")
def simple_mlp(
    features, num_classes, activate_first=True, units=(256,), dropout_rate=0.5
):
    def activate(x):
        x = tf.keras.layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    if activate_first:
        features = activate(features)
    for u in units:
        features = layers.Dense(u)(features)
        features = activate(features)
    logits = layers.Dense(num_classes, name="logits")(features)
    return logits
