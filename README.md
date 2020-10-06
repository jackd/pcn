# Point Cloud Convolutions

Point-cloud implementations of [Sparse Convolutions on Continuous Domains for
Point Cloud and Event Stream Networks ACCV2020](TODO).

## Quick Start

Installation:

```bash
# pip dependencies
pip install tensorflow==2.3  # 2.2 also works except for CSR
pip install tensorflow-datasets==2.0.0 Pillow numba trimesh

# pip packages from git
git clone https://github.com/jackd/shape-tfds.git
git clone https://github.com/jackd/kblocks.git
git clone https://github.com/jackd/numba-neighbors.git
git clone https://github.com/jackd/pcn.git
pip install -e shape-tfds
pip install -e kblocks
pip install -e numba-neighbors
pip install -e pcn

cd pcn
```

Train the large resnet model from the paper:

```bash
python -m kblocks '$KB_CONFIG/fit' pcn/configs/pn2-resnet/large.gin
```

## Benchmark Results

To generate results from the paper

### Operation benchmarks

Our operation benchmarks (Table 2) are generated using tensorflow's CSR matmul implementation (requires tensorflow 2.3), though the standard `tf.sparse.sparse_dense_matmul` implementation is almost as fast. The following generates the `N(F \Theta)-JIT` row from the paper.

```bash
python pcn/ops/conv_benchmark.py --csr --transform_first --jit
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

### Network benchmarks

Values in Table 3 corresponds to `wall_time * examples_per_epoch / batch_size`.

Results below are on an 8-core machine with GTX-1080Ti.

#### Online preprocessing

```bash
python -m kblocks '$KB_CONFIG/benchmark' \
    pcn/configs/pn2-resnet/large.gin \
    --bindings='
        shuffle_buffer = 1  # ensure preprocess time is included
        PipelinedSource.cache_managers = None  # preprocesses online
        batch_size=1024  # default is 128
        burn_iters=5  # default is 5
        min_iters=20  # default is 100
    '
```

Results:

```txt
Wall time (ms): 1259.9842548370361
Memory (Mb):    1852.4022178649902
```

#### Offline preprocessing

This will take a while - mostly because it will generate cached files for 32 epochs.

```bash
python -m kblocks '$KB_CONFIG/benchmark' \
    pcn/configs/pn2-resnet/large.gin \
    --bindings='
        batch_size=1024
    '
```

Results

```txt
Wall time (ms): 1259.9842548370361
Memory (Mb):    1852.4022178649902
```

## FAQ

Q: `~/pcn` is massive! Is there any way to make things smaller?

A: Most of this is preprocessed cache files. Compression can be turned on to reduce this at the cost of a slight training slow-down by using `TFRecordsCacheManager.compression = 'GZIP'`.

Q: Why is caching so complicated? Why not just use `tf.data.Dataset.cache` or `tf.data.experimental.snapshot`?

A: These other options are supported, though all except the default exhibit slow but monotonic memory increases, hence the default `TFRecordsCacheManager`.

```bash
# using tf.data.Dataset.cache
cache_managers.train_impl = @BaseCacheManager
cache_managers.validation_impl = @BaseCacheManager
```

```bash
# using tf.data.experimental.snapshot
cache_managers.train_impl = @SnapshotManager
cache_managers.validation_impl = @SnapshotManager
```

```bash
# using tf.data.experimental.[save|load]
cache_managers.train_impl = @SaveLoadManager
cache_managers.validation_impl = @SaveLoadManager
```
