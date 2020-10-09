from typing import Optional

import numpy as np
from numba import njit

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
def ragged_in_place_and_down_sample_query(
    coords: np.ndarray,
    in_place_radius: float,
    down_sample_radius: float,
    rejection_radius: Optional[float] = None,
    in_place_k: Optional[int] = None,
    down_sample_k: Optional[int] = None,
    max_sample_size: Optional[int] = None,
    leaf_size: int = 16,
):
    in_tree = KDTree3(coords, leaf_size)
    start_nodes = in_tree.get_node_indices()
    dists, ip_indices, ip_counts = in_tree.query_radius_bottom_up(
        coords,
        in_place_radius,
        start_nodes,
        coords.shape[0] if in_place_k is None else in_place_k,
    )

    sample_indices, sample_size = bt.rejection_sample_precomputed(
        ip_indices,
        ip_counts,
        coords.shape[0] if max_sample_size is None else max_sample_size,
        valid=None if rejection_radius is None else dists < rejection_radius,
    )

    sample_indices = sample_indices[:sample_size]

    out_coords = coords[sample_indices]
    _, ds_indices, ds_counts = in_tree.query_radius_bottom_up(
        out_coords,
        down_sample_radius,
        start_nodes[sample_indices],
        out_coords.shape[0] if down_sample_k is None else down_sample_k,
    )

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
def ragged_in_place_query(
    coords: np.ndarray, radius: float, k: Optional[int] = None, leaf_size: int = 16
):
    tree = KDTree3(coords, leaf_size=leaf_size)
    r = tree.query_radius_bottom_up(
        coords, radius, tree.get_node_indices(), coords.shape[0] if k is None else k
    )
    indices = r.indices
    counts = r.counts
    indices = flat_ragged_values(indices, counts)
    return (indices, counts)


@njit(fastmath=True)
def ragged_down_sample_query(
    coords,
    rejection_radius: float,
    query_radius: float,
    k: Optional[int] = None,
    max_sample_size: Optional[int] = None,
    leaf_size: int = 16,
):
    (
        _,
        _,
        sample_indices,
        ds_indices,
        ds_counts,
    ) = ragged_in_place_and_down_sample_query(
        coords,
        in_place_radius=rejection_radius,
        down_sample_radius=query_radius,
        rejection_radius=None,
        in_place_k=k,
        down_sample_k=k,
        max_sample_size=max_sample_size,
        leaf_size=leaf_size,
    )
    return sample_indices, ds_indices, ds_counts
    # TODO: work out why the below isn't faster. Is it just the parallel search?
    # tree = KDTree3(coords, leaf_size=leaf_size)
    # sample_result, query_result = tree.rejection_sample_query(
    #     rejection_radius ** 2,
    #     query_radius ** 2,
    #     tree.get_node_indices(),
    #     max_counts=coords.shape[0] if k is None else k,
    #     max_samples=coords.shape[0] if max_sample_size is None else max_sample_size,
    # )

    # sample_count = sample_result.count
    # sample_indices = sample_result.indices[:sample_count]
    # query_counts = query_result.counts[:sample_count]
    # query_indices = flat_ragged_values(
    #     query_result.indices[:sample_count], query_counts
    # )
    # return sample_indices, query_indices, query_counts


def knn_query(in_coords, out_coords, k: int):
    from pykdtree.kdtree import (  # pylint: disable=no-name-in-module,import-outside-toplevel
        KDTree as pyKDTree,
    )

    tree = pyKDTree(in_coords)
    dists, indices = tree.query(out_coords, k)
    return dists, indices
    # return np.reshape(indices, (-1)), np.full((out_coords.shape[0],), k)


@njit()
def ifp_sample_precomputed(dists, query_indices, counts, sample_size: int, eps=1e-8):
    result = bt.ifp_sample_precomputed(
        dists=dists,
        query_indices=query_indices,
        counts=counts,
        sample_size=sample_size,
        eps=eps,
    )
    return result.indices
