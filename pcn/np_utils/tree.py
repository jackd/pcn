import numpy as np
from numba import njit
from numba import prange
from numba_neighbors import binary_tree as bt
from numba_neighbors.kd_tree import KDTree3


@njit()
def flat_ragged_values(values, lengths):
    total = np.sum(lengths)
    out = np.empty((total, *values.shape[2:]), dtype=values.dtype)
    row_start = 0
    n = lengths.size
    for i in range(n):
        rl = lengths[i]
        row_end = row_start + rl
        out[row_start:row_end] = values[i, :rl]
        row_start = row_end
    return out


@njit(parallel=bt.PARALLEL, fastmath=True)
def ragged_in_place_and_down_sample_query(coords: np.ndarray,
                                          in_place_radius: float,
                                          in_place_k: int,
                                          down_sample_radius: float,
                                          down_sample_k: int,
                                          max_sample_size: int = -1,
                                          leaf_size: int = 16):
    if max_sample_size == -1:
        max_sample_size = coords.shape[0]
    in_tree = KDTree3(coords, leaf_size)
    start_nodes = in_tree.get_node_indices()
    _, ip_indices, ip_counts = in_tree.query_radius_bottom_up(
        coords, in_place_radius, start_nodes, in_place_k)

    sample_indices, sample_size = bt.rejection_sample_precomputed(
        ip_indices, ip_counts, max_sample_size)

    sample_indices = sample_indices[:sample_size]

    out_coords = coords[sample_indices]
    _, ds_indices, ds_counts = in_tree.query_radius_bottom_up(
        out_coords, down_sample_radius, start_nodes[sample_indices],
        down_sample_k)

    ip_indices = flat_ragged_values(ip_indices, ip_counts)
    ds_indices = flat_ragged_values(ds_indices, ds_counts)

    return (
        ip_indices,
        ip_counts,
        sample_indices,
        ds_indices,
        ds_counts,
    )


@njit(fastmath=True)
def ragged_in_place_query(coords: np.ndarray,
                          radius: float,
                          k: int,
                          leaf_size: int = 16):
    tree = KDTree3(coords, leaf_size=leaf_size)
    r = tree.query_radius_bottom_up(coords, radius, tree.get_node_indices(), k)
    indices = r.indices
    counts = r.counts
    indices = flat_ragged_values(indices, counts)
    return (indices, counts)
