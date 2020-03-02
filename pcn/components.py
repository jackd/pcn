import functools
from typing import Optional, Any, Tuple
import tensorflow as tf
from kblocks import multi_graph as mg
# from kblocks.utils import memoized_property
import kblocks.extras.layers.ragged as ragged_layers
import kblocks.extras.layers.shape as shape_layers
from .layers import conv as conv_layers
from .layers import tree as tree_layers

IntTensor = tf.Tensor
FloatTensor = tf.Tensor


def to_nearest_power(x, base=2):
    x = tf.convert_to_tensor(x, dtype_hint=tf.int64)
    base = tf.convert_to_tensor(base, dtype_hint=x.dtype)
    assert (x.dtype.is_integer)
    return base**tf.cast(
        tf.math.ceil(
            tf.math.log(tf.cast(x, tf.float32)) /
            tf.math.log(tf.cast(base, tf.float32))), x.dtype)


def pad_ragged(rt, padding, values=0):
    flat_values = rt.values

    flat_values = tf.pad(rt.flat_values, [[0, padding]], constant_values=values)
    starts, total = tf.split(rt.row_splits, [1, -1])
    row_splits = tf.concat((starts, total + padding), axis=0)
    return tf.RaggedTensor.from_row_splits(flat_values,
                                           tf.cast(row_splits, tf.int64),
                                           validate=False)


def pad_ragged_to_nearest_power(rt: tf.RaggedTensor, base=2):
    total = rt.row_splits[-1]
    actual = to_nearest_power(total, base=base)
    padding = actual - total
    return pad_ragged(rt, padding)


@functools.wraps(tf.stack)
def tf_stack(values, axis=0, name=None):
    # fixed duplicate name in graph issues.
    if name is None:
        name = values[0].graph.unique_name('stack')
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


class RaggedStructure(object):

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

    @memoized_property
    def nrows(self) -> IntTensor:
        return shape_layers.dimension(self.row_starts)

    def as_ragged(self, x) -> tf.RaggedTensor:
        return ragged_layers.from_row_splits(x, self.row_splits, validate=False)

    @staticmethod
    def from_ragged(rt: tf.RaggedTensor):
        if rt.ragged_rank != 1:
            raise NotImplementedError('TODO')
        return RaggedStructure(ragged_layers.row_splits(rt))


class Cloud(object):

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
        assert (isinstance(batched_values, tf.RaggedTensor))
        mg.assert_is_post_batch(batched_values)
        model_values = mg.model_input(batched_values)
        if self._bucket_size:
            model_values = tf.keras.layers.Lambda(pad_ragged_to_nearest_power)(
                model_values)
        self._model_structure = RaggedStructure.from_ragged(model_values)
        return model_values

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

    def sample_query(
            self,
            in_place_radius,
            in_place_k,
            down_sample_radius,
            down_sample_k,
            edge_features_fn,
            weight_fn,
            max_sample_size=-1,
            leaf_size=16,
    ) -> Tuple['SampledCloud', 'Neighborhood', 'Neighborhood']:
        with mg.pre_cache_context():
            ip_indices, sample_indices, ds_indices = tree_layers.ragged_in_place_and_down_sample_query(
                self.coords,
                in_place_radius=in_place_radius,
                in_place_k=in_place_k,
                down_sample_radius=down_sample_radius,
                down_sample_k=down_sample_k,
                max_sample_size=max_sample_size,
                leaf_size=leaf_size,
            )
        out_cloud = SampledCloud(self,
                                 sample_indices,
                                 bucket_size=self.bucket_size)
        ip_neigh = neighborhood(self, self, in_place_radius, ip_indices,
                                edge_features_fn, weight_fn)
        ds_neigh = neighborhood(self, out_cloud, down_sample_radius, ds_indices,
                                edge_features_fn, weight_fn)
        return out_cloud, ip_neigh, ds_neigh

    def query(self, radius, k, edge_features_fn, weight_fn, leaf_size=16):
        with mg.pre_cache_context():
            indices = tree_layers.ragged_in_place_query(self.coords, radius, k,
                                                        leaf_size)
        return neighborhood(self, self, radius, indices, edge_features_fn,
                            weight_fn)


class SampledCloud(Cloud):

    def __init__(self,
                 in_cloud: Cloud,
                 indices: IntTensor,
                 bucket_size: bool = False):
        mg.assert_is_pre_cache(indices)
        self._in_cloud = in_cloud
        self._indices = indices
        self._bucket_size = bucket_size
        batched_indices = self._batch(indices)
        with mg.post_batch_context():
            self._batched_indices = ragged_layers.values(
                batched_indices) + tf.gather(
                    self.in_cloud.batched_structure.row_starts,
                    self.batched_structure.value_rowids)
            batched_indices = self.batched_structure.as_ragged(
                self._batched_indices)
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
    assert (ragged_indices.ragged_rank == 2)
    b = ragged_indices.value_rowids()
    ragged_indices = ragged_indices.values
    i = ragged_indices.value_rowids()
    b = tf.gather(b, i)
    j = ragged_indices.values
    j = j + tf.gather(offset, b)
    return i, j


def ragged_to_block_sparse(ragged_indices, offset):
    return tf.keras.layers.Lambda(_ragged_to_block_sparse)(
        [ragged_indices, offset])


def neighborhood(in_cloud, out_cloud, radius, indices, edge_features_fn,
                 weight_fn):
    mg.assert_is_pre_cache(indices)
    batched_indices = mg.batch(mg.cache(indices))
    with mg.post_batch_context():
        i, j = ragged_to_block_sparse(batched_indices,
                                      in_cloud.batched_structure.row_starts)
    i = mg.model_input(i)
    j = mg.model_input(j)
    sparse_indices = tf_stack((i, j), axis=-1)
    rel_coords = (tf.gather(in_cloud.model_coords / radius, j) -
                  tf.gather(out_cloud.model_coords / radius, i))
    edge_features = edge_features_fn(rel_coords)
    assert (edge_features.shape[0] is not None)
    dists = tf.linalg.norm(rel_coords, axis=-1)
    weights = weight_fn(dists)
    return Neighborhood(in_cloud, out_cloud, sparse_indices, edge_features,
                        weights)


def _transpose(args):
    indices, edge_features, weights = args
    values = tf.range(tf.shape(indices)[0])
    st = tf.SparseTensor(indices, values, (-1, -1))
    st = tf.sparse.reorder(tf.sparse.transpose(st, (1, 0)))
    values = st.values
    edge_features = tf.gather(edge_features, values, axis=1)
    weights = tf.gather(weights, values)
    return st.indices, edge_features, weights


class Neighborhood(object):

    def __init__(self,
                 in_cloud: Cloud,
                 out_cloud: Cloud,
                 sparse_indices: IntTensor,
                 edge_features: FloatTensor,
                 weights: FloatTensor,
                 eps=1e-4):
        mg.assert_is_model_tensor(sparse_indices)
        mg.assert_is_model_tensor(edge_features)
        mg.assert_is_model_tensor(weights)
        self._in_cloud = in_cloud
        self._out_cloud = out_cloud
        self._edge_features = edge_features
        self._weights = weights

        i, _ = tf.unstack(sparse_indices, axis=-1)
        row_sum = tf.math.unsorted_segment_sum(
            weights, i, out_cloud.model_structure.total_size)
        row_sum = tf.gather(row_sum, i)
        weights = tf.where(row_sum < eps, tf.zeros_like(weights),
                           weights / tf.gather(row_sum, i))
        self._normalized_edge_features = self._edge_features * tf.expand_dims(
            weights, axis=0)
        self._sparse_indices = sparse_indices
        self._transpose = None

    def transpose(self):
        if self._transpose is None:
            if self._in_cloud is self._out_cloud:
                self._transpose = self
            else:
                sparse_indices, edge_features, weights = tf.keras.layers.Lambda(
                    _transpose)([
                        self._sparse_indices, self._edge_features, self._weights
                    ])
                self._transpose = Neighborhood(self.out_cloud, self.in_cloud,
                                               sparse_indices, edge_features,
                                               weights)
                self._transpose._transpose = self
        return self._transpose

    @property
    def in_cloud(self) -> Cloud:
        return self._in_cloud

    @property
    def out_cloud(self) -> Cloud:
        return self._out_cloud

    def convolve(self,
                 node_features: Optional[FloatTensor],
                 filters: int,
                 activation=None,
                 **kwargs):
        if node_features is not None:
            mg.assert_is_model_tensor(node_features)
        return conv_layers.sparse_cloud_convolution(
            node_features,
            self._normalized_edge_features,
            self._sparse_indices,
            self._out_cloud.model_structure.total_size,
            filters=filters,
            activation=activation,
            **kwargs)
