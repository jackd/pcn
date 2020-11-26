"""
Script for benchmarking performance of matmuls with `SparseTensor` vs `CSRSparseMatrix`.

Requires tensorflow 2.3 and absl-py

```bash
pip install tensorflow==2.3
pip install absl-py
```
"""
from typing import Callable, Dict

import numpy as np
import tensorflow as tf
from absl import app, flags

from pcn.ops.conv import (
    ReductionImplementation,
    SparseImplementation,
    featureless_conv,
    sparse_conv,
)

flags.DEFINE_integer("ni", default=4096, help="number of points per cloud in")
flags.DEFINE_integer("no", default=4096, help="number of points per cloud out")
flags.DEFINE_integer("fi", default=64, help="number of filters in")
flags.DEFINE_integer("fo", default=64, help="number of filters out")
flags.DEFINE_integer("k", default=9, help="mean number of edges")
flags.DEFINE_integer("b", default=8, help="batch_size")
flags.DEFINE_integer("t", default=4, help="number of edge features")
flags.DEFINE_bool("transform_first", default=False, help="dense matmul first")
flags.DEFINE_string(
    "combine",
    default=ReductionImplementation.UNSTACK,
    help=f"one of {ReductionImplementation.all()}",
)
flags.DEFINE_integer("burn", default=10, help="number of burn iterations")
flags.DEFINE_integer("iters", default=100, help="number of iterations to average over")
flags.DEFINE_boolean("jit", default=False, help="XLA jit compilation")
flags.DEFINE_string(
    "sparse_impl",
    default=SparseImplementation.COO,
    help=f"Use sparse implementation to use, one of {SparseImplementation.all()}",
)
flags.DEFINE_integer("seed", default=0, help="seed for random number generation")


def summarize(result, print_fn=print):
    """
    Args:
        result: output of a tf.test.Benchmark.run_op_benchmark call.
        print_fn: print-like function.
    """
    print_fn("Wall time (ms): {}".format(result["wall_time"] * 1000))
    gpu_mem = result["extras"].get("allocator_maximum_num_bytes_GPU_0_bfc", 0)
    print_fn("Memory (Mb):    {}".format(gpu_mem / 1024 ** 2))


def summarize_all(*args, print_fn=print):
    """
    Applies `summarize` to (name, result) pairs.

    Args:
        *args: (name, result) pairs
        print_fn: print-like function.
    """
    for name, result in args:
        print_fn(name)
        summarize(result, print_fn)


def get_conv_args(
    N_in=4096, N_out=4096, F_in=64, F_out=64, K=9, B=8, T=4, seed=0
) -> Dict:
    print("{:10}: {}".format("N_in", N_in))
    print("{:10}: {}".format("N_out", N_out))
    print("{:10}: {}".format("F_in", F_in))
    print("{:10}: {}".format("F_out", F_out))
    print("{:10}: {}".format("K", K))
    print("{:10}: {}".format("B", B))
    print("{:10}: {}".format("T", T))
    print("{:10}: {}".format("seed", seed))

    np.random.seed(seed)
    N_in = N_in * B
    N_out = N_out * B
    num_edges = K * N_out

    features = tf.constant(np.random.normal(size=(N_in, F_in)), dtype=tf.float32)
    edge_features = tf.constant(np.random.normal(size=(T, num_edges)), dtype=tf.float32)
    while True:
        flat_index = np.random.randint(
            0,
            high=N_in * N_out,
            size=num_edges * 2,  # extras for duplicates
            dtype=np.int64,
        )
        flat_index = np.unique(flat_index)
        if len(flat_index) >= num_edges:
            flat_index = flat_index[:num_edges]
            break
    flat_index = np.sort(flat_index)
    i, j = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
        flat_index, (N_out, N_in)
    )
    sparse_indices = tf.constant(np.stack((i, j), axis=-1), dtype=tf.int64)

    if F_in > 0:
        kernel = tf.Variable(
            np.random.uniform(size=(T, F_in, F_out)).astype(np.float32)
        )
    else:
        kernel = tf.Variable(np.random.uniform(size=(T, F_out)).astype(np.float32))

    dense_shape = [tf.constant(N_out), tf.constant(N_in)]
    return sparse_indices, edge_features, dense_shape, features, kernel


def conv_example(
    sparse_impl: str, combine: str, transform_first: bool, F_in: int, **kwargs
):
    sparse_indices, edge_features, dense_shape, features, kernel = get_conv_args(
        F_in=F_in, **kwargs
    )
    with tf.GradientTape() as tape:
        if F_in == 0:
            params = kernel
            tape.watch(params)
            x = featureless_conv(edge_features, sparse_indices, kernel, dense_shape[0])
        else:
            params = kernel, features
            tape.watch(params)
            x = sparse_conv(
                features,
                edge_features,
                sparse_indices,
                kernel,
                dense_shape,
                sparse_impl=sparse_impl,
                combine=combine,
                transform_first=transform_first,
            )
        grad = tape.gradient(x, params)

    return ("forward", x), ("backward", grad)


def benchmark_fn(fn: Callable, burn_iters: int, min_iters: int, **fn_kwargs):
    with tf.Graph().as_default() as graph:
        names_and_outputs = fn(**fn_kwargs)
        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            bm = tf.test.Benchmark()
            results = []
            for name, output in names_and_outputs:
                print(f"Starting benchmark {name}...")
                result = bm.run_op_benchmark(
                    sess, output, burn_iters=burn_iters, min_iters=min_iters
                )
                summarize(result)
                results.append((name, result))
    print("-----")
    summarize_all(*results)
    return results


def main(_):
    FLAGS = flags.FLAGS
    tf.config.optimizer.set_jit(FLAGS.jit)
    call_kwargs = dict(
        N_in=FLAGS.ni,
        N_out=FLAGS.no,
        F_in=FLAGS.fi,
        F_out=FLAGS.fo,
        B=FLAGS.b,
        T=FLAGS.t,
        K=FLAGS.k,
        sparse_impl=FLAGS.sparse_impl,
        combine=FLAGS.combine,
        transform_first=FLAGS.transform_first,
        seed=FLAGS.seed,
    )
    benchmark_fn(
        conv_example, burn_iters=FLAGS.burn, min_iters=FLAGS.iters, **call_kwargs
    )


if __name__ == "__main__":
    app.run(main)
