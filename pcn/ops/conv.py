from typing import Callable, Optional, Tuple

import tensorflow as tf

from kblocks.tf_typing import DenseShape, TensorOrVariable

TermImpl = Callable[[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], tf.Tensor]


def map_reduce_sum(
    map_fn, inputs, out_shape, out_type, parallel_iterations=None, method="unstack"
):
    if method == "fold":
        init = tf.zeros(out_shape, dtype=out_type)
        out = tf.foldl(lambda acc, args: acc + map_fn(args), inputs, init)
    elif method == "map":
        out = tf.reduce_sum(
            tf.map_fn(
                map_fn, inputs, parallel_iterations=parallel_iterations, dtype=out_type
            ),
            axis=0,
        )
    elif method == "vmap":
        out = tf.reduce_sum(tf.vectorized_map(map_fn, inputs), axis=0)
    elif method == "unstack":
        inputs = (tf.unstack(i, axis=0) for i in inputs)
        out = tf.add_n([map_fn(args) for args in zip(*inputs)])
    else:
        raise ValueError(
            "Invalid method {}: must be one of {}".format(
                method, ("fold", "map", "vmap", "unstack")
            )
        )
    return out


def featureless_conv(
    edge_features: tf.Tensor,
    sparse_indices: tf.Tensor,
    kernel: TensorOrVariable,
    out_size: tf.Tensor,
):
    """
    Compute a point cloud convolution when there are no input features.

    Equivalent to

    out_{ip} = \\sum_{j, t} n^{(t)}_{ij}^{(t)} \\theta^{(t)}_p.

    Args:
        edge_features: [T, E] float tensor of edge weights.
        sparse_indices: [E, 2] int64 tensor of sparse indices for a sparse
            tensor with dense shape [N_out, N_in], or [E] int64 tensor
            corresponding to the first index (i.e. in [0, N_out)). First column
            (or whole thing if size is [E]) should be in ascending order.
        kernel: [T, F_out] float tensor.

    Returns:
        [N_out, F_out] node features.
    """
    with tf.name_scope("featureless_conv"):
        kernel = tf.convert_to_tensor(kernel, dtype_hint=tf.float32)
        sparse_indices = tf.convert_to_tensor(sparse_indices, dtype_hint=tf.int64)
        edge_features = tf.convert_to_tensor(edge_features, dtype_hint=tf.float32)
        out_size = tf.convert_to_tensor(out_size, dtype_hint=tf.int64)
        if sparse_indices.shape.ndims == 1:
            i = sparse_indices
        else:
            assert sparse_indices.shape.ndims == 2
            assert sparse_indices.shape[1] == 2
            i = sparse_indices[:, 0]
        # edge_features are [K, E] everywhere else in this module
        edge_features = tf.transpose(edge_features, (1, 0))  # [E, K]
        neigh_sum = tf.math.unsorted_segment_sum(
            edge_features, i, num_segments=out_size
        )
        return tf.matmul(neigh_sum, kernel)  # [N, F_out]


def get_sparse_transform(sparse_indices: tf.Tensor, dense_shape: DenseShape):
    if isinstance(sparse_indices, (list, tuple)):
        assert len(sparse_indices) == 2
        sparse_indices = tf.stack(sparse_indices, axis=-1)

    def fn(features, edge_features):
        st = tf.SparseTensor(sparse_indices, edge_features, dense_shape)
        return tf.sparse.sparse_dense_matmul(st, features)

    return fn


def get_csr_transform(sparse_indices: tf.Tensor, dense_shape: DenseShape):
    from tensorflow.python.ops.linalg.sparse import (  # pylint: disable=import-outside-toplevel
        sparse as sparse_lib,
    )

    # sparse_lib requires tf >= 2.3
    if isinstance(sparse_indices, (list, tuple)):
        assert len(sparse_indices) == 2
        sparse_indices = tf.stack(sparse_indices, axis=-1)

    def fn(features, edge_features):
        st = tf.SparseTensor(sparse_indices, edge_features, dense_shape)
        csr = sparse_lib.CSRSparseMatrix(st)
        return sparse_lib.matmul(csr, features)

    return fn


def get_reordered_sparse_transform(sparse_indices: tf.Tensor, dense_shape: DenseShape):
    values = tf.range(tf.shape(sparse_indices, out_type=tf.int64)[0])
    sp = tf.SparseTensor(sparse_indices, values, dense_shape)
    sp = tf.sparse.reorder(sp)
    sparse_indices = sp.indices
    order = sp.values

    def fn(features, edge_features):
        sp = tf.SparseTensor(
            sparse_indices, tf.gather(edge_features, order), dense_shape
        )
        return tf.sparse.sparse_dense_matmul(sp, features)

    return fn


def get_gather_sum_transform(
    sparse_indices: tf.Tensor, dense_shape: Optional[DenseShape] = None
):
    if isinstance(sparse_indices, (list, tuple)):
        i, j = sparse_indices
    else:
        i, j = tf.unstack(sparse_indices, axis=-1)
    out_size = dense_shape[0]

    def fn(features, edge_features):
        features = tf.gather(features, j)
        features = features * tf.expand_dims(edge_features, axis=-1)
        # features = tf.math.segment_sum(features, i)
        # features = segment_sum(features, i, out_size)
        features = tf.math.unsorted_segment_sum(features, i, out_size)
        return features

    return fn


def sparse_conv(
    features: tf.Tensor,
    edge_features: tf.Tensor,
    sparse_indices: tf.Tensor,
    kernel: TensorOrVariable,
    dense_shape: DenseShape,
    transform_first: Optional[bool] = None,
    combine: str = "unstack",
    use_csr: bool = False,
):
    """
    Graph convolution.

    Args:
        features: [N_in, F_in] float tensor of input features.
        edge_features: [T, E] float tensor of sparse neighbors matrix for each
            kernel feature.
        sparse_indices: [E, 2] int tensor of indices of neighbors matrix.
        dense_shape: dense shape of neighbors, value (N_out, N_in).
        kernel: [T, F_in, F_out] float tensor of feature transformations.
        transform_first: bool dictating whether to transform before or after
            sparse multiplication. Defaults to the option with fewer operations
            based on the same number of points.
        combine: one of "fold", "unstack", "map".

    Returns:
        [N_out, F_out] float tensor of features.
    """
    if use_csr:
        # assert combine == "unstack"
        # return csr_sparse_conv(
        #     features=features,
        #     edge_features=edge_features,
        #     sparse_indices=sparse_indices,
        #     kernel=kernel,
        #     dense_shape=dense_shape,
        #     transform_first=transform_first,
        # )
        term_impl = get_csr_transform
    else:
        term_impl = get_sparse_transform
    # term_impl = get_sparse_transform if is_ordered else get_gather_sum_transform
    T, F_in, F_out = kernel.shape
    if isinstance(dense_shape, tf.Tensor):
        dense_shape = tf.unstack(dense_shape, axis=0)
    N_out, N_in = dense_shape
    if transform_first is None:
        transform_first = F_out <= F_in
    get_term_base = term_impl(sparse_indices, dense_shape)

    del N_in

    def get_term(args):
        kernel, edge_features = args
        if transform_first:
            f = tf.matmul(features, kernel)
        else:
            f = features
        f = get_term_base(f, edge_features)
        if not transform_first:
            f = tf.matmul(f, kernel)
        return f

    return map_reduce_sum(
        get_term,
        (kernel, edge_features),
        out_shape=(N_out, F_out),
        out_type=features.dtype,
        parallel_iterations=T,
        method=combine,
    )


def block_conv(
    features: tf.Tensor,
    kernel: TensorOrVariable,
    sparse_indices: tf.Tensor,
    edge_features: tf.Tensor,
    dense_shape: Tuple[tf.Tensor, tf.Tensor],
    transform_first: Optional[bool] = None,
    use_csr: bool = False,
):
    term_impl = get_csr_transform if use_csr else get_sparse_transform
    T, F_in, F_out = kernel.shape
    N_out, N_in = dense_shape
    if transform_first is None:
        transform_first = F_out <= F_in

    if transform_first:
        # concat along axis=1
        # note tf.sparse.concat doesn't seem to support gradients?
        i, j = tf.unstack(sparse_indices, axis=-1)
        i = tf.tile(tf.expand_dims(i, axis=0), (T, 1))
        j = tf.expand_dims(j, axis=0) + tf.expand_dims(
            tf.range(T, dtype=j.dtype) * N_in, axis=1
        )

        i = tf.reshape(i, (-1,))
        j = tf.reshape(j, (-1,))
        edge_features = tf.reshape(edge_features, (-1,))
        sp = tf.sparse.reorder(
            tf.SparseTensor(tf.stack((i, j), axis=-1), edge_features, (N_out, T * N_in))
        )
        features = tf.expand_dims(features, axis=0)
        features = tf.matmul(features, kernel)  # [T, N_in, F_out]
        features = tf.reshape(features, (T * N_in, -1))
        features = term_impl(sp.indices, (N_out, T * N_in))(features, sp.values)
        return features

    # transform last
    # concat along axis=0
    i, j = tf.unstack(sparse_indices, axis=-1)
    i = tf.expand_dims(i, axis=0) + tf.expand_dims(
        tf.range(T, dtype=i.dtype) * N_out, axis=1
    )
    j = tf.tile(j, (T,))

    i = tf.reshape(i, (-1,))
    j = tf.reshape(j, (-1,))
    edge_features = tf.reshape(edge_features, (-1,))
    features = term_impl(tf.stack((i, j), axis=-1), (T * N_out, N_in))(
        features, edge_features
    )
    features = tf.reshape(features, (T, N_out, F_in))
    features = tf.reshape(tf.transpose(features, (1, 0, 2)), (N_out, T * F_in))
    kernel = tf.reshape(kernel, (T * F_in, F_out))
    return tf.matmul(features, kernel)


def sparse_flex_max_pool(features: tf.Tensor, sparse_indices: tf.Tensor):
    """
    Perform flex max pooling as defined in FlexConv paper.

    Args:
        features: [N_in, F] float features.
        sparse_indices: [E, 2] int64 sparse indices corresponding to a sparse
            tensor with dense shape [N_out, N_in] and ordered according to
            `tf.sparse.reorder`.

    Returns:
        [N_out, F] float features.
    """
    i, j = tf.unstack(sparse_indices, axis=-1)
    return tf.math.segment_max(tf.gather(features, j, axis=0), i)


def sparse_flex_up_sample(features: tf.Tensor, sparse_indices: tf.Tensor):
    """
    Perform flex up sampling as defined in FlexConv paper.

    Args:
        features: [N_out, F] float features.
        sparse_indices: [E, 2] int64 sparse indices corresponding to sparse
            tensor with dense shape [N_out, N_in] and ordered according to
            `tf.sparse.reorder`.

    Returns:
        [N_in, F] float features.
    """
    i, j = tf.unstack(sparse_indices, axis=-1)
    N_in = tf.reduce_max(j) + 1
    return tf.math.unsorted_segment_max(
        tf.gather(features, i, axis=0), j, num_segments=N_in
    )
