import tensorflow as tf

IntTensor = tf.Tensor
FloatTensor = tf.Tensor

# def normalize_rows_sparse(sparse_indices, values, out_size, in_size):
#     sp = tf.SparseTensor(sparse_indices, values, (out_size, in_size))
#     sp = sp / tf.sparse.reduce_sum(sp, axis=1, keepdims=True)
#     return sp.values


def normalize_rows(row_indices: IntTensor, values: FloatTensor,
                   out_size: IntTensor):
    out = tf.math.unsorted_segment_sum(values, row_indices, out_size)
    return values / tf.gather(out, row_indices)


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
    full_padding = [[0, 0] for _ in range(flat_values.shape.ndims)]
    full_padding[0][1] = padding
    flat_values = tf.pad(flat_values, full_padding, constant_values=values)
    starts, total = tf.split(rt.row_splits, [-1, 1])
    row_splits = tf.concat((starts, total + padding), axis=0)
    return tf.RaggedTensor.from_row_splits(flat_values,
                                           tf.cast(row_splits, tf.int64),
                                           validate=False)


def pad_ragged_to_nearest_power(rt: tf.RaggedTensor, base=2):
    total = rt.row_splits[-1]
    actual = to_nearest_power(total, base=base)
    padding = actual - total
    return pad_ragged(rt, padding)
