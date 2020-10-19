import abc
from typing import Optional

import gin
import tensorflow as tf

from pcn.ops import conv as conv_ops

IntTensor = tf.Tensor
FloatTensor = tf.Tensor

activations = tf.keras.activations
constraints = tf.keras.constraints
initializers = tf.keras.initializers
layers = tf.keras.layers
regularizers = tf.keras.regularizers
InputSpec = layers.InputSpec


@gin.configurable(module="pcn.layers")
class ConvolutionBase(layers.Layer):
    def __init__(
        self,
        filters: int,
        activation=None,
        use_bias: bool = True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs
    ):
        super(ConvolutionBase, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs
        )
        self.filters = int(filters)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True

    def get_config(self):
        config = {
            "filters": self.filters,
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "activity_regularizer": regularizers.serialize(self.activity_regularizer),
            "kernel_constraint": constraints.serialize(self.kernel_constraint),
            "bias_constraint": constraints.serialize(self.bias_constraint),
        }
        base_config = super(ConvolutionBase, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @abc.abstractmethod
    def call(self, inputs):  # pylint:disable=arguments-differ
        raise NotImplementedError

    def _finalize(self, outputs):
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)  # pylint:disable=no-member
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


@gin.configurable(module="pcn.layers")
class FeaturelessSparseCloudConvolution(ConvolutionBase):
    def __init__(self, *args, **kwargs):
        super(FeaturelessSparseCloudConvolution, self).__init__(*args, **kwargs)
        self.input_spec = [
            InputSpec(ndim=2),  # [S, E] edge features
            InputSpec(shape=(None, 2), dtype=tf.int64),  # [E, 2] sparse indices
            InputSpec(ndim=0),
        ]

    def build(self, input_shape):
        if self.built:
            return

        es = input_shape[0]
        num_edge_features = es[0]
        if num_edge_features is None:
            raise ValueError("Dimension 0 of edge features must be statically known")
        self.kernel = self.add_weight(
            "kernel",
            shape=[num_edge_features, self.filters],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.filters],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.input_spec[0] = InputSpec(shape=(num_edge_features, None))
        super(FeaturelessSparseCloudConvolution, self).build(input_shape)

    def call(self, inputs):
        edge_features, sparse_indices, out_size = (
            tf.convert_to_tensor(i) for i in inputs
        )

        outputs = conv_ops.featureless_conv(
            edge_features, sparse_indices, self.kernel, out_size
        )

        return self._finalize(outputs)


@gin.configurable(module="pcn.layers")
class SparseCloudConvolution(ConvolutionBase):
    def __init__(
        self,
        *args,
        transform_first: Optional[bool] = None,
        combine: str = "unstack",
        use_csr: bool = False,
        **kwargs
    ):
        self._conv_kwargs = dict(
            transform_first=transform_first, combine=combine, use_csr=use_csr
        )
        super(SparseCloudConvolution, self).__init__(*args, **kwargs)
        self.input_spec = [
            InputSpec(ndim=2),  # [N_in, F_in] node features
            InputSpec(ndim=2),  # [S, E] edge features
            InputSpec(shape=(None, 2), dtype=tf.int64),  # [E, 2] sparse indices
            InputSpec(shape=(), dtype=tf.int64),  # [] out_size
        ]

    def get_config(self):
        config = super(SparseCloudConvolution, self).get_config()
        config.update(self._conv_kwargs)
        return config

    def build(self, input_shape):
        if self.built:
            return
        ns, ef = input_shape[:2]
        num_node_features = ns[1]
        num_edge_features = ef[0]
        if num_edge_features is None:
            raise ValueError("Dimension 0 of edge features must be statically known")
        if num_node_features is None:
            raise ValueError("Dimension 1 of node features must be statically known")
        self.kernel = self.add_weight(
            "kernel",
            shape=[num_edge_features, num_node_features, self.filters],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.filters],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        self.input_spec[0] = InputSpec(shape=(None, num_node_features))
        self.input_spec[1] = InputSpec(shape=(num_edge_features, None))
        super().build(input_shape)

    def call(self, inputs):
        node_features, edge_features, indices, out_size = inputs
        dense_shape = (out_size, tf.shape(node_features, out_type=out_size.dtype)[0])
        outputs = conv_ops.sparse_conv(
            node_features,
            edge_features,
            indices,
            self.kernel,
            dense_shape,
            **self._conv_kwargs
        )
        return self._finalize(outputs)


def sparse_cloud_convolution(
    node_features: Optional[FloatTensor],
    edge_features: FloatTensor,
    indices: IntTensor,
    out_size: IntTensor,
    **kwargs
):
    if node_features is None:
        layer = FeaturelessSparseCloudConvolution(**kwargs)
        return layer([edge_features, indices, out_size])
    else:
        layer = SparseCloudConvolution(**kwargs)
        return layer([node_features, edge_features, indices, out_size])


@gin.configurable(module="pcn.layers")
class FlexMaxPool(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(FlexMaxPool, self).__init__(*args, **kwargs)
        self.input_spec = [
            InputSpec(ndim=2, dtype=self.dtype),
            InputSpec(shape=(None, 2), dtype=tf.int64),
        ]

    def call(self, inputs):
        features, sparse_indices = inputs
        return conv_ops.sparse_flex_max_pool(features, sparse_indices)


@gin.configurable(module="pcn.layers")
class FlexUpSample(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(FlexUpSample, self).__init__(*args, **kwargs)
        self.input_spec = [
            InputSpec(ndim=2, dtype=self.dtype),
            InputSpec(shape=(None, 2), dtype=tf.int64),
        ]

    def call(self, inputs):
        features, sparse_indices = inputs
        return conv_ops.sparse_flex_up_sample(features, sparse_indices)
