# Point Cloud Convolutions

Point-cloud implementations of [Sparse Convolutions on Continuous Domains for
Point Cloud and Event Stream Networks ACCV2020](TODO).

## Quick Start

Installation:

```bash
# pip dependencies
# not included in requirements.txt because tf-nightly might be installed
pip install tensorflow==2.3  # 2.2 also works except for CSR

# pip packages from git
git clone https://github.com/jackd/pcn.git
pip install -e pcn
```

Train the large resnet model from the paper:

```bash
python -m pcn '$KB_CONFIG/trainables/fit.gin' '$PCN_CONFIG/trainables/large.gin'
```

See also [examples/pn2-large.py](examples/pn2-large.py) for a complete training example.

Note the first time this is run 32 epochs of augmented data are cached. This may take some time.

## Project Structure

This project depends on multiple custom python packages. These are:

- [meta-model](https://github.com/jackd/meta-model.git) for simultaneously building and connecting the multiple models associated with data pipelining and model training.
- [kblocks](https://github.com/jackd/kblocks.git) for experiment management and configuration via [gin-config](https://github.com/google/gin-config.git).
- [shape-tfds](https://github.com/jackd/shape-tfds.git) for[tensorflow-datasets](https://github.com/tensorflow/datasets.git) implementations that manage dataset downloading and model-independent preprocessing for 3D shape-based datasets.
- [numba-neighbors](https://github.com/jackd/numba-neighbors.git) for [numba](https://github.com/numba/numba.git) implementations of KDTrees and subsampling.

## `python -m pcn`

`python -m pcn` is a light wrapper around `python -m kblocks` which exposes `$PCN_CONFIG` for command line argument just like `$KB_CONFIG` is exposed in `kblocks`. In particular, note that `$PCN_CONFIG` is set inside the python script, so must be passed as a string, e.g. `python -m pcn '$PCN_CONFIG/foo'` rather than `python -m pcn $PCN_CONFIG/foo`.

### Example configurations

Fit a large model with online data preprocessing, with run id 1.

```bash
python -m pcn '$KB_CONFIG/trainables/fit.gin' \
    '$PCN_CONFIG/trainables/large.gin' \
    '$PCN_CONFIG/data/aug/online.gin' \
    --bindings='run=1'
```

Visualize data augmentation (requires trimesh: `pip install trimesh`)

```bash
python -m pcn '$PCN_CONFIG/vis.gin' '$PCN_CONFIG/data/pn2.gin'
```

## Operation benchmarks

Our operation benchmarks (Table 2) are generated using tensorflow's CSR matmul implementation (requires tensorflow 2.3), though the standard `tf.sparse.sparse_dense_matmul` implementation is almost as fast. The following generates the `N(F \Theta)-JIT` row from the paper.

```bash
python pcn/ops/conv_benchmark.py --sparse_impl=csr --transform_first --jit
```

Results on a GTX-1080Ti:

```txt
-----
forward
Wall time (ms): 2.6831626892089844
Memory (Mb):    40.0
backward
Wall time (ms): 4.087090492248535
Memory (Mb):    49.00001525878906
```

## Network benchmarks

Values in Table 3 corresponds to `batch_time * examples_per_epoch / batch_size`.

## FAQ

Q: Caching takes a while. Is there any way to make it faster?

A: There are many different caching options available in [kblocks/data/cache](https://github.com/jackd/kblocks/blob/master/kblocks/data/cache.py) Default configuration is defined in [data/aug/offline.gin](pcn/configs/data/aug/offline.gin). Currently we use `tfrecords_cache` which is a custom implementation. Options include:

- `kblocks.data.tfrecords_cache`: a custom implementation
  - supports compression
  - eager
  - possibly ~10% slower than other options?
- `kblocks.data.cache`: wraps `tf.data.Dataset.cache`
  - does not support compression
  - lazy
- `kblocks.data.snapshot`: wraps `tf.data.experimental.snapshot`
  - supports compression
  - lazy
  - memory leak in tf-nightly (2.5) ?
- `kblocks.data.save_load_cache`: combines `tf.data.experimental.[save,load]`
  - supports compression
  - eager
  - memory leak in tf-nightly (2.5) ?

Other options to reduce initial caching time include:

- reducing the number of `cache_repeats` (e.g. `cache_repeats = 16`)
- using online augmentation (`include "$PCN_CONFIG/data/aug/online.gin"`)
