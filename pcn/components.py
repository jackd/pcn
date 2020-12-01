import functools
from typing import Callable, Optional, Tuple

import tensorflow as tf

import meta_model.pipeline as pl
import pcn.ops.utils as utils_ops
from pcn.layers import conv as conv_layers
from pcn.layers import tree as tree_layers
from wtftf.meta import memoized_property
from wtftf.ragged import layers as ragged_layers
from wtftf.ragged.utils import RaggedStructure, ragged_rank

IntTensor = tf.Tensor
FloatTensor = tf.Tensor

Lambda = tf.keras.layers.Lambda
TensorMap = Callable[[tf.Tensor], tf.Tensor]


@functools.wraps(tf.stack)
def tf_stack(values, axis=0, name=None):
    # fixed duplicate name in graph issues.
    if isinstance(values, tf.Tensor) and name is None:
        name = values[0].graph.unique_name("stack")
        return tf.stack(values, axis=axis, name=name)
    return tf.stack(values, axis=axis, name=name)


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

        self._batched_coords = ragged_layers.values(batched_coords)
        self._model_coords = ragged_layers.values(model_coords)

    @property
    def coords(self) -> FloatTensor:
        return self._coords

    @property
    def batched_structure(self) -> RaggedStructure:
        return self._batched_structure

    @property
    def model_structure(self) -> FloatTensor:
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
        flat_batched_indices = ragged_layers.values(batched_indices) + tf.gather(
            self.in_cloud.batched_structure.row_starts,
            self.batched_structure.value_rowids,
        )
        model_indices = pl.model_input(
            self._batched_structure.as_ragged(flat_batched_indices)
        )
        self._batched_indices = flat_batched_indices
        self._model_indices = ragged_layers.values(model_indices)
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
    assert ragged_rank(ragged_indices) == 2
    b = ragged_layers.value_rowids(ragged_indices)
    ragged_indices = ragged_layers.values(ragged_indices)
    i = ragged_layers.value_rowids(ragged_indices)
    b = tf.gather(b, i)
    j = ragged_layers.values(ragged_indices)
    j = j + tf.gather(offset, b)
    return i, j


def ragged_to_block_sparse(ragged_indices, offset):
    # return Lambda(_ragged_to_block_sparse)([ragged_indices, offset])
    return _ragged_to_block_sparse([ragged_indices, offset])


def neighborhood(
    in_cloud: Cloud,
    out_cloud: Cloud,
    radius: float,
    indices: tf.RaggedTensor,
    edge_features_fn: TensorMap,
    weight_fn: Optional[TensorMap] = None,
    normalize: bool = True,
    version: str = "v0",
) -> "Neighborhood":
    """
    Get a `Neighborhood` from relevant clouds and parameters.

    Args:
        in_cloud: input cloud.
        out_cloud: output cloud.
        radius: radius of ball search
        indices: pre-cache indices into `in_cloud` corresponding to neighbors
            in `in_cloud` of `out_cloud`.
        edge_features_fn: function producing (N, E?) edge features.
        weights_fn: weighting function to apply to edge features, function of the
            normalized edge length. If given, edge features are scaled by this value.
        normalize: if True, edge features are divided by the sum over the neighborhood
            of weights given by `weights_fn`.
        version: string indicating version which influences what to cache / when
            to apply various meta-ops. Supported:
            "v0": cache only minimal values and compute relative coords, edge features
                and weights in the model.
            "v1": cache minimal values, compute relative coords post-batch, edge
                features / weights in model.
            "v2": cache minimal values, compute relative coords, edge features and
                weights post-batch.
            "v3": cache relative coordinates, compute edge features / weights in model.
            "v4": cache relative coordinates, edge features and weights.

    Returns:
        `Neighborhood`.
    """

    def get_weights(rel_coords):
        if weight_fn is None:
            return None
        return weight_fn(tf.linalg.norm(rel_coords, axis=-1))

    cached_indices = pl.cache(indices)
    batched_indices = pl.batch(cached_indices)
    # post-batch
    i, j = ragged_to_block_sparse(
        batched_indices, in_cloud.batched_structure.row_starts
    )
    model_i = pl.model_input(i)
    model_j = pl.model_input(j)
    sparse_indices = tf_stack((model_i, model_j), axis=-1)
    if version == "v0":
        # compute rel coords / edges features in model
        rel_coords = tf.gather(in_cloud.model_coords / radius, model_j) - tf.gather(
            out_cloud.model_coords / radius, model_i
        )
        edge_features = edge_features_fn(rel_coords)
        weights = get_weights(rel_coords)
    elif version == "v1":
        # compute rel coords in batch, rest in model
        rel_coords = tf.gather(in_cloud.batched_coords / radius, j) - tf.gather(
            out_cloud.batched_coords / radius, i
        )
        rel_coords = pl.model_input(rel_coords)
        # edge features in model
        edge_features = edge_features_fn(rel_coords)
        weights = get_weights(rel_coords)
    elif version == "v2":
        # compute edge features in batch
        rel_coords = tf.gather(in_cloud.batched_coords / radius, j) - tf.gather(
            out_cloud.batched_coords / radius, i
        )
        edge_features = edge_features_fn(rel_coords)
        weights = get_weights(rel_coords)
        edge_features = pl.model_input(edge_features)
        weights = pl.model_input(weights)
    elif version == "v3":
        # cache relative coords
        i = ragged_layers.value_rowids(indices)
        j = ragged_layers.values(indices)
        rel_coords = tf.gather(in_cloud.coords / radius, j) - tf.gather(
            out_cloud.coords / radius, i
        )
        rel_coords = pl.model_input(pl.batch(pl.cache(rel_coords)))
        rel_coords = ragged_layers.values(rel_coords)
        edge_features = edge_features_fn(rel_coords)
        weights = get_weights(rel_coords)
    elif version == "v4":
        # cache edge features / weights
        i = ragged_layers.value_rowids(indices)
        j = ragged_layers.values(indices)
        rel_coords = tf.gather(in_cloud.coords / radius, j) - tf.gather(
            out_cloud.coords / radius, i
        )
        edge_features = edge_features_fn(rel_coords)  # (N, E?)
        edge_features = tf.transpose(edge_features, (1, 0))  # (E?, N)
        edge_features = pl.batch(pl.cache(edge_features))  # (B, E?, N)
        edge_features = pl.model_input(edge_features)
        edge_features = ragged_layers.values(edge_features)  # (BE, N)
        edge_features = tf.transpose(edge_features, (1, 0))  # (N, BE)

        weights = get_weights(rel_coords)  # (E,)
        weights = pl.model_input(pl.batch(pl.cache(weights)))  # (B, E?)
        weights = ragged_layers.values(weights)  # (BE,)
    else:
        raise ValueError(f"invalid version {version}")

    assert edge_features.shape[0] is not None

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
        **kwargs,
    ):
        return conv_layers.sparse_cloud_convolution(
            node_features,
            self._normalized_edge_features,
            self._sparse_indices,
            self._out_cloud.model_structure.total_size,
            filters=filters,
            activation=activation,
            **kwargs,
        )
