import functools
from typing import Any, Optional, Tuple

import tensorflow as tf

import meta_model.pipeline as pl
import pcn.ops.utils as utils_ops
from composite_layers import ragged
from pcn.layers import conv as conv_layers
from pcn.layers import tree as tree_layers

IntTensor = tf.Tensor
FloatTensor = tf.Tensor

Lambda = tf.keras.layers.Lambda


@functools.wraps(tf.stack)
def tf_stack(values, axis=0, name=None):
    # fixed duplicate name in graph issues.
    if isinstance(values, tf.Tensor) and name is None:
        name = values[0].graph.unique_name("stack")
        return tf.stack(values, axis=axis, name=name)
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
            cached = super().__get__(obj, type)
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
        return tf.ragged.row_splits_to_segment_ids(self.row_splits)
        # return Lambda(tf.ragged.row_splits_to_segment_ids)(self.row_splits)

    @memoized_property
    def nrows(self) -> IntTensor:
        return tf.size(self.row_starts)

    def as_ragged(self, x) -> tf.RaggedTensor:
        return ragged.from_row_splits(x, self.row_splits, validate=False)

    @staticmethod
    def from_ragged(rt: tf.RaggedTensor):
        if ragged.ragged_rank(rt) != 1:
            raise NotImplementedError("TODO")
        return RaggedStructure(ragged.row_splits(rt))

    @staticmethod
    def from_dense(dense):
        bs, n = tf.unstack(tf.shape(dense)[:2])
        row_splits = tf.range(0, (bs + 1) * n, n, dtype=tf.int64)
        return RaggedStructure(row_splits)

    @staticmethod
    def from_tensor(x):
        if ragged.is_ragged(x):
            return RaggedStructure.from_ragged(x)
        return RaggedStructure.from_dense(x)


def _as_ragged(x):
    if isinstance(x, tf.Tensor):
        x = tf.RaggedTensor.from_tensor(x)

    assert isinstance(x, tf.RaggedTensor)
    return tf.identity(x)


def as_ragged(x):
    return tf.keras.layers.Lambda(_as_ragged)(x)


class Cloud:
    def __init__(self, coords: FloatTensor):
        self._coords = coords

        batched_coords = as_ragged(pl.batch(pl.cache(coords)))
        self._batched_structure = RaggedStructure.from_ragged(batched_coords)

        model_coords = pl.model_input(batched_coords)
        self._model_structure = RaggedStructure.from_ragged(model_coords)

        self._batched_coords = ragged.values(batched_coords)
        self._model_coords = ragged.values(model_coords)

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
        # pre-cache
        sample_indices, ds_indices = tree_layers.ragged_down_sample_query(
            self.coords,
            rejection_radius,
            down_sample_radius,
            k=down_sample_k,
            max_sample_size=max_sample_size,
            leaf_size=leaf_size,
        )
        out_cloud = SampledCloud(self, sample_indices)
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
        # pre-cache
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

        out_cloud = SampledCloud(self, sample_indices)
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
        # pre-cache
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
        self, in_cloud: Cloud, indices: IntTensor
    ):
        assert indices.shape[0] is None
        self._in_cloud = in_cloud
        self._indices = indices
        batched_indices = pl.batch(pl.cache(indices))
        self._batched_structure = RaggedStructure.from_ragged(batched_indices)
        # post-batch
        flat_batched_indices = ragged.values(batched_indices) + tf.gather(
            self.in_cloud.batched_structure.row_starts,
            self.batched_structure.value_rowids,
        )
        model_indices = pl.model_input(
            self._batched_structure.as_ragged(flat_batched_indices)
        )
        self._batched_indices = flat_batched_indices
        self._model_indices = ragged.values(model_indices)
        self._model_structure = RaggedStructure.from_ragged(model_indices)

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
        # pre-cache
        return tf.gather(self.in_cloud.coords, self.indices)

    @memoized_property
    def batched_coords(self):
        # post-batch
        return tf.gather(self.in_cloud.batched_coords, self.batched_indices)

    @memoized_property
    def model_coords(self):
        return tf.gather(self.in_cloud.model_coords, self.model_indices)


def _ragged_to_block_sparse(args):
    ragged_indices, offset = args
    assert ragged.ragged_rank(ragged_indices) == 2
    b = ragged.value_rowids(ragged_indices)
    ragged_indices = ragged.values(ragged_indices)
    i = ragged.value_rowids(ragged_indices)
    b = tf.gather(b, i)
    j = ragged.values(ragged_indices)
    j = j + tf.gather(offset, b)
    return i, j


def ragged_to_block_sparse(ragged_indices, offset):
    # return Lambda(_ragged_to_block_sparse)([ragged_indices, offset])
    return _ragged_to_block_sparse([ragged_indices, offset])


def neighborhood(
    in_cloud, out_cloud, radius, indices, edge_features_fn, weight_fn, normalize=True
):

    indices = pl.cache(indices)
    batched_indices = pl.batch(indices)
    # post-batch
    i, j = ragged_to_block_sparse(
        batched_indices, in_cloud.batched_structure.row_starts
    )
    i = pl.model_input(i)
    j = pl.model_input(j)
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
        self._in_cloud = in_cloud
        self._out_cloud = out_cloud
        self._edge_features = edge_features
        self._weights = weights
        self._sparse_indices = sparse_indices
        self._normalize = normalize
        if weights is None:
            self._normalized_edge_features = edge_features
        else:
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
        return conv_layers.sparse_cloud_convolution(
            node_features,
            self._normalized_edge_features,
            self._sparse_indices,
            self._out_cloud.model_structure.total_size,
            filters=filters,
            activation=activation,
            **kwargs
        )
