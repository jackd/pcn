from absl import flags
from absl import app

from typing import Dict
import numpy as np
import tensorflow as tf
from kblocks.benchmark_utils import summarize_all
from pcn.ops import conv as conv_ops

flags.DEFINE_integer('ni', default=4096, help='number of points per cloud in')
flags.DEFINE_integer('no', default=4096, help='number of points per cloud out')
flags.DEFINE_integer('fi', default=64, help='number of filters in')
flags.DEFINE_integer('fo', default=64, help='number of filters out')
flags.DEFINE_integer('k', default=9, help='mean number of edges')
flags.DEFINE_integer('b', default=8, help='batch_size')
flags.DEFINE_integer('t', default=4, help='number of edge features')
flags.DEFINE_bool('transform_first', default=False, help='dense matmul first')
flags.DEFINE_string('combine',
                    default='fold',
                    help='one of "fold", "map", "unstack"')
flags.DEFINE_integer('burn', default=10, help='number of burn iterations')
flags.DEFINE_integer('iters',
                     default=100,
                     help='number of iterations to average over')
flags.DEFINE_boolean('jit', default=False, help='XLA jit compilation')


def get_conv_args(N_in=4096, N_out=4096, F_in=64, F_out=64, K=9, B=8,
                  T=4) -> Dict:
    print('{:10}: {}'.format('N_in', N_in))
    print('{:10}: {}'.format('N_out', N_out))
    print('{:10}: {}'.format('F_in', F_in))
    print('{:10}: {}'.format('F_out', F_out))
    print('{:10}: {}'.format('K', K))
    print('{:10}: {}'.format('B', B))
    print('{:10}: {}'.format('T', T))

    N_in = N_in * B
    N_out = N_out * B
    num_edges = K * N_out

    features = tf.random.normal((N_in, F_in), dtype=tf.float32)
    edge_features = tf.random.normal((T, num_edges), dtype=tf.float32)
    while True:
        flat_index = np.random.randint(
            0,
            high=N_in * N_out,
            size=num_edges * 2,  # extras for duplicates
            dtype=np.int64)
        flat_index = np.unique(flat_index)
        if len(flat_index) >= num_edges:
            flat_index = flat_index[:num_edges]
            break
    flat_index = np.sort(flat_index)
    i, j = np.unravel_index(flat_index, (N_out, N_in))  # pylint: disable=unbalanced-tuple-unpacking
    sparse_indices = tf.constant(np.stack((i, j), axis=-1), dtype=tf.int64)

    if F_in > 0:
        kernel = tf.Variable(
            np.random.uniform(size=(T, F_in, F_out)).astype(np.float32))
    else:
        kernel = tf.Variable(
            np.random.uniform(size=(T, F_out)).astype(np.float32))

    dense_shape = [tf.constant(N_out), tf.constant(N_in)]
    return dict(features=features,
                edge_features=edge_features,
                sparse_indices=sparse_indices,
                dense_shape=dense_shape,
                kernel=kernel)


class SparseOpsBenchmark(tf.test.Benchmark):

    def benchmark_base(self,
                       burn_iters,
                       min_iters,
                       F_in,
                       call_kwargs={},
                       **arg_kwargs):
        graph = tf.Graph()
        arg_kwargs['F_in'] = F_in
        with graph.as_default():
            kwargs = get_conv_args(**arg_kwargs)
            if F_in == 0:
                forward = conv_ops.featureless_conv(
                    edge_features=kwargs['edge_features'],
                    kernel=kwargs['kernel'],
                    sparse_indices=kwargs['sparse_indices'],
                    out_size=kwargs['dense_shape'][0])
                train_params = kwargs['kernel'],
                params = tuple(kwargs[k] for k in ('edge_features', 'kernel'))
            else:
                forward = conv_ops.sparse_conv(**kwargs, **call_kwargs)
                train_params = kwargs['kernel'], kwargs['features']
                params = tuple(
                    kwargs[k] for k in ('features', 'edge_features', 'kernel'))
            backward = tf.gradients(forward, params)
            backward_kernel = tf.gradients(forward, train_params)

        run_kwargs = dict(burn_iters=burn_iters, min_iters=min_iters)

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(kwargs['kernel'].initializer)
            print('forward')
            f = self.run_op_benchmark(sess, forward, **run_kwargs)
            print('backward (kernel)')
            bk = self.run_op_benchmark(sess, (forward, backward_kernel),
                                       **run_kwargs)
            print('backward')
            b = self.run_op_benchmark(sess, (forward, backward), **run_kwargs)
            summarize_all(
                ('forward', f),
                ('backward (kernel)', bk),
                ('backward', b),
            )


def main(_):
    # # tf.test.main()
    # def printb(x):
    #     print('-------------------')
    #     print(x)
    #     print('-------------------')

    # printb('base')
    FLAGS = flags.FLAGS
    tf.config.optimizer.set_jit(FLAGS.jit)
    bm = SparseOpsBenchmark()
    bm.benchmark_base(N_in=FLAGS.ni,
                      N_out=FLAGS.no,
                      F_in=FLAGS.fi,
                      F_out=FLAGS.fo,
                      B=FLAGS.b,
                      T=FLAGS.t,
                      K=FLAGS.k,
                      burn_iters=FLAGS.burn,
                      min_iters=FLAGS.iters,
                      call_kwargs=dict(transform_first=FLAGS.transform_first,
                                       combine=FLAGS.combine))

    # printb('fold, first')
    # bm.benchmark_base(call_kwargs=dict(transform_first=True, impl='fold'))
    # printb('fold, last')
    # bm.benchmark_base(call_kwargs=dict(transform_first=False, impl='fold'))

    # printb('unstack, first')
    # bm.benchmark_base(call_kwargs=dict(transform_first=True, impl='unstack'))
    # printb('unstack, last')
    # bm.benchmark_base(call_kwargs=dict(transform_first=False, impl='unstack'))


if __name__ == '__main__':
    app.run(main)
