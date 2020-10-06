import functools
from typing import Any, Optional, Tuple

import tensorflow as tf

# from kblocks.utils import memoized_property
import kblocks.extras.layers.ragged as ragged_layers
import kblocks.extras.layers.shape as shape_layers
import multi_graph as mg
import pcn.ops.utils as utils_ops
from pcn.layers import conv as conv_layers
from pcn.layers import tree as tree_layers

IntTensor = tf.Tensor
FloatTensor = tf.Tensor

Lambda = tf.keras.layers.Lambda


@functools.wraps(tf.stack)
def tf_stack(values, axis=0, name=None):
    # fixed duplicate name in graph issues.
    if name is None:
        name = values[0].graph.unique_name("stack")
    return tf.stack(values, axis=axis, name=name)


class memoized_property(property):  # pylint: disable=invalid-name
    """Descriptor that mimics @property but caches output in member variable."""

    def __get__(self, obj: Any, type: Optional[type] = ...) -> Any:
        # See https://docs.python.org/3/howto/descriptor.html#properties
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError("unreadable attribute")
        attr = "__cached_" + self.fget.__name__
        cached = getattr(obj, attr, None)
        if cached is None:
            # cached = self.fget(obj)
            cached = super(memoized_property, self).__get__(obj, type)
            setattr(obj, attr, cached)
        return cached


class RaggedStructure:
    def __init__(self, row_splits):
        self._row_splits = row_splits

    @property
    def row_splits(self):
        return self._row_splits

    @memoized_property
    def total_size(self):
        return self.row_splits[-1]

    @memoized_property
    def row_starts(self) -> IntTensor:
        return self._row_splits[:-1]

    @memoized_property
    def row_ends(self) -> IntTensor:
        return self._row_splits[1:]

    @memoized_property
    def row_lengths(self) -> IntTensor:
        return self.row_ends - self.row_starts

    @memoized_property
    def value_rowids(self) -> IntTensor:
        # return tf.ragged.row_splits_to_segment_ids(self.row_splits)
        return Lambda(tf.ragged.row_splits_to_segment_ids)(self.row_splits)

    @memoized_property
    def nrows(self) -> IntTensor:
        return shape_layers.dimension(self.row_starts)

    def as_ragged(self, x) -> tf.RaggedTensor:
        return ragged_layers.from_row_splits(x, self.row_splits, validate=False)

    @staticmethod
    def from_ragged(rt: tf.RaggedTensor):
        if rt.ragged_rank != 1:
            raise NotImplementedError("TODO")
        return RaggedStructure(ragged_layers.row_splits(rt))


class Cloud:
    def __init__(self, coords: FloatTensor, bucket_size: bool = False):
        self._bucket_size = bucket_size
        self._coords = coords
        batched_coords = self._batch(coords)
        model_coords = self._model(batched_coords)

        self._batched_coords = ragged_layers.values(batched_coords)
        self._model_coords = ragged_layers.values(model_coords)

    def _batch(self, values):
        mg.assert_is_pre_cache(values)
        values = mg.cache(values)
        with mg.pre_batch_context():
            values = ragged_layers.pre_batch_ragged(values)
        values = mg.batch(values)
        with mg.post_batch_context():
            values = ragged_layers.post_batch_ragged(values, validate=False)
            self._batched_structure = RaggedStructure.from_ragged(values)
            return values

    def _model(self, batched_values: tf.RaggedTensor):
        assert isinstance(batched_values, tf.RaggedTensor)
        mg.assert_is_post_batch(batched_values)
        if self._bucket_size:
            with mg.post_batch_context():
                valid_size = shape_layers.dimension(
                    ragged_layers.values(batched_values), 0
                )
                valid_size = tf.expand_dims(
                    valid_size, axis=0
                )  # so it can be a keras input
                batched_values = Lambda(utils_ops.pad_ragged_to_nearest_power)(
                    batched_values
                )
            valid_size = mg.model_input(valid_size)
            self._valid_size = tf.squeeze(valid_size, axis=0)
        else:
            self._valid_size = None
        model_values = mg.model_input(batched_values)
        self._model_structure = RaggedStructure.from_ragged(model_values)
        return model_values

    @property
    def valid_size(self):
        return self._valid_size

    def mask_invalid(self, features):
        if self.valid_size is None:
            return features
        mg.assert_is_model_tensor(features)
        mask = tf.sequence_mask(self.valid_size, self.model_structure.total_size)
        return tf.where(
            tf.expand_dims(mask, axis=-1), features, tf.zeros_like(features)
        )

    @property
    def coords(self) -> FloatTensor:
        return self._coords

    @property
    def batched_structure(self):
        return self._batched_structure

    @property
    def model_structure(self):
        return self._model_structure

    @property
    def batched_coords(self) -> FloatTensor:
        return self._batched_coords

    @property
    def model_coords(self) -> FloatTensor:
        return self._model_coords

    @property
    def bucket_size(self) -> bool:
        return self._bucket_size

    def down_sample_query(
        self,
        rejection_radius: float,
        down_sample_radius: float,
        edge_features_fn,
        weight_fn,
        down_sample_k=None,
        max_sample_size=None,
        normalize=True,
        leaf_size=16,
    ) -> Tuple["SampledCloud", "Neighborhood"]:
        with mg.pre_cache_context():
            sample_indices, ds_indices = tree_layers.ragged_down_sample_query(
                self.coords,
                rejection_radius,
                down_sample_radius,
                k=down_sample_k,
                max_sample_size=max_sample_size,
                leaf_size=leaf_size,
            )
        out_cloud = SampledCloud(self, sample_indices, bucket_size=self.bucket_size)
        ds_neigh = neighborhood(
            self,
            out_cloud,
            down_sample_radius,
            ds_indices,
            edge_features_fn,
            weight_fn,
            normalize=normalize,
        )
        return out_cloud, ds_neigh

    def sample_query(
        self,
        in_place_radius,
        down_sample_radius,
        edge_features_fn,
        weight_fn,
        rejection_radius=None,
        in_place_k=None,
        down_sample_k=None,
        normalize=True,
        max_sample_size=None,
        leaf_size=16,
    ) -> Tuple["SampledCloud", "Neighborhood", "Neighborhood"]:
        with mg.pre_cache_context():
            (
                ip_indices,
                sample_indices,
                ds_indices,
            ) = tree_layers.ragged_in_place_and_down_sample_query(
                self.coords,
                in_place_radius=in_place_radius,
                in_place_k=in_place_k,
                rejection_radius=rejection_radius,
                down_sample_radius=down_sample_radius,
                down_sample_k=down_sample_k,
                max_sample_size=max_sample_size,
                leaf_size=leaf_size,
            )
        out_cloud = SampledCloud(self, sample_indices, bucket_size=self.bucket_size)
        ip_neigh = neighborhood(
            self,
            self,
            in_place_radius,
            ip_indices,
            edge_features_fn,
            weight_fn,
            normalize=normalize,
        )
        ds_neigh = neighborhood(
            self,
            out_cloud,
            down_sample_radius,
            ds_indices,
            edge_features_fn,
            weight_fn,
            normalize=normalize,
        )
        return out_cloud, ip_neigh, ds_neigh

    def query(
        self, radius, edge_features_fn, weight_fn, k=None, normalize=True, leaf_size=16
    ):
        assert k is None or isinstance(k, int)
        with mg.pre_cache_context():
            indices = tree_layers.ragged_in_place_query(
                self.coords, radius, k, leaf_size=leaf_size
            )
        return neighborhood(
            self,
            self,
            radius,
            indices,
            edge_features_fn,
            weight_fn,
            normalize=normalize,
        )


class SampledCloud(Cloud):
    def __init__(  # pylint:disable=super-init-not-called
        self, in_cloud: Cloud, indices: IntTensor, bucket_size: bool = False
    ):
        mg.assert_is_pre_cache(indices)
        self._in_cloud = in_cloud
        self._indices = indices
        self._bucket_size = bucket_size
        batched_indices = self._batch(indices)
        with mg.post_batch_context():
            self._batched_indices = ragged_layers.values(batched_indices) + tf.gather(
                self.in_cloud.batched_structure.row_starts,
                self.batched_structure.value_rowids,
            )
            batched_indices = self.batched_structure.as_ragged(self._batched_indices)
        model_indices = self._model(batched_indices)
        self._model_indices = ragged_layers.values(model_indices)

    @property
    def in_cloud(self) -> Cloud:
        return self._in_cloud

    @property
    def indices(self) -> IntTensor:
        return self._indices

    @property
    def batched_indices(self) -> IntTensor:
        return self._batched_indices

    @property
    def model_indices(self) -> IntTensor:
        return self._model_indices

    @memoized_property
    def coords(self):
        with mg.pre_cache_context():
            return tf.gather(self.in_cloud.coords, self.indices)

    @memoized_property
    def batched_coords(self):
        with mg.post_batch_context():
            return tf.gather(self.in_cloud.batched_coords, self.batched_indices)

    @memoized_property
    def model_coords(self):
        return tf.gather(self.in_cloud.model_coords, self.model_indices)


def _ragged_to_block_sparse(args):
    ragged_indices, offset = args
    assert ragged_indices.ragged_rank == 2
    b = ragged_indices.value_rowids()
    ragged_indices = ragged_indices.values
    i = ragged_indices.value_rowids()
    b = tf.gather(b, i)
    j = ragged_indices.values
    j = j + tf.gather(offset, b)
    return i, j


def ragged_to_block_sparse(ragged_indices, offset):
    return Lambda(_ragged_to_block_sparse)([ragged_indices, offset])


def neighborhood(
    in_cloud, out_cloud, radius, indices, edge_features_fn, weight_fn, normalize=True
):
    mg.assert_is_pre_cache(indices)
    batched_indices = mg.batch(mg.cache(indices))
    with mg.post_batch_context():
        i, j = ragged_to_block_sparse(
            batched_indices, in_cloud.batched_structure.row_starts
        )
    i = mg.model_input(i)
    j = mg.model_input(j)
    sparse_indices = tf_stack((i, j), axis=-1)
    rel_coords = tf.gather(in_cloud.model_coords / radius, j) - tf.gather(
        out_cloud.model_coords / radius, i
    )
    edge_features = edge_features_fn(rel_coords)
    assert edge_features.shape[0] is not None
    dists = tf.linalg.norm(rel_coords, axis=-1)
    if weight_fn is None:
        weights = None
    else:
        weights = weight_fn(dists)
    return Neighborhood(
        in_cloud, out_cloud, sparse_indices, edge_features, weights, normalize=normalize
    )


class Neighborhood:
    def __init__(
        self,
        in_cloud: Cloud,
        out_cloud: Cloud,
        sparse_indices: IntTensor,
        edge_features: FloatTensor,
        weights: Optional[FloatTensor],
        normalize: bool = True,
    ):
        mg.assert_is_model_tensor(sparse_indices)
        mg.assert_is_model_tensor(edge_features)
        self._in_cloud = in_cloud
        self._out_cloud = out_cloud
        self._edge_features = edge_features
        self._weights = weights
        self._sparse_indices = sparse_indices
        self._normalize = normalize
        if weights is None:
            self._normalized_edge_features = edge_features
        else:
            mg.assert_is_model_tensor(weights)
            if normalize:
                weights = utils_ops.normalize_rows(
                    tf.unstack(sparse_indices, axis=-1)[0],
                    weights,
                    out_cloud.model_structure.total_size,
                )
            self._normalized_edge_features = self._edge_features * tf.expand_dims(
                weights, axis=0
            )
        self._transpose = None

    def transpose(self) -> "Neighborhood":
        if self._transpose is None:
            if self._in_cloud is self._out_cloud:
                self._transpose = self
            else:

                def transpose_params(args):
                    indices, edge_features, weights, out_size, in_size = args
                    values = tf.range(tf.shape(indices)[0])
                    st = tf.SparseTensor(indices, values, (out_size, in_size))
                    st = tf.sparse.reorder(tf.sparse.transpose(st, (1, 0)))
                    values = st.values
                    edge_features = tf.gather(edge_features, values, axis=1)
                    weights = tf.gather(weights, values)
                    return st.indices, edge_features, weights

                sparse_indices, edge_features, weights = Lambda(transpose_params)(
                    [
                        self._sparse_indices,
                        self._edge_features,
                        self._weights,
                        self._out_cloud.model_structure.total_size,
                        self._in_cloud.model_structure.total_size,
                    ]
                )
                self._transpose = Neighborhood(
                    self.out_cloud,
                    self.in_cloud,
                    sparse_indices,
                    edge_features,
                    weights,
                    normalize=self._normalize,
                )
                self._transpose._transpose = self  # pylint:disable=protected-access
        return self._transpose

    @property
    def in_cloud(self) -> Cloud:
        return self._in_cloud

    @property
    def out_cloud(self) -> Cloud:
        return self._out_cloud

    def convolve(
        self,
        node_features: Optional[FloatTensor],
        filters: int,
        activation=None,
        **kwargs
    ):
        if node_features is not None:
            mg.assert_is_model_tensor(node_features)
        return conv_layers.sparse_cloud_convolution(
            node_features,
            self._normalized_edge_features,
            self._sparse_indices,
            self._out_cloud.model_structure.total_size,
            filters=filters,
            activation=activation,
            **kwargs
        )
