import pickle
from typing import List

import gudhi as gd
import numpy as np
import torch_geometric.data
from gudhi.wasserstein.barycenter import lagrangian_barycenter
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from curvature.compute_curvature import compute_curvature
from ph import compute_ph
from utils import load_data


def get_subsample(adj: np.ndarray, n: int, size: int) -> List[np.ndarray]:
    """
    Sample multiple sub-adjacency matrices from the original adjacency matrix.
    :param adj: adjacency matrix.
    :param n: number of sub-adjacency matrices.
    :param size: row/column size of the sub-adjacency matrices.
    :return: list of sub-adjacency matrices.
    """
    row_total = adj.shape[0]  # this is also the column total due to the adjacency matrix being a square matrix
    sub_adj = []
    for i in range(n):
        row_sample = np.random.choice(row_total, size, replace=False, p=None)  # also a column sample
        sub_adj.append(adj[row_sample, :][:, row_sample])
    return sub_adj


def estimate_ph(data: torch_geometric.data.data.Data, size: int = 1000, stochastic: bool = False) -> np.ndarray:
    """
    Efficiently estimate the persistence homology for the input data.
    :param data: input graph data.
    :param size: row/column size of the adjacency sub-matrices.
    :param stochastic: if True, the sub-adjacency matrices are computed using the get_subsample function;
                       if False, the sub-adjacency matrices are created sequentially.
    :return: numpy array of mean PH points.
    """
    G = to_networkx(data, node_attrs=['x']).to_undirected()
    G, C = compute_curvature(G, 'formanCurvature')
    N = len(G.edges)

    if stochastic:
        subs = get_subsample(C, N // size + 1, size)
    else:
        subs = [C[size * i:size * (i + 1), size * i:size * (i + 1)] for i in range(N // size + 1)]

    diags = []

    for A in tqdm(subs):
        points = compute_ph.compute_ph(A)
        if len(points) > 1:
            diags.append(points)

    mean, _ = lagrangian_barycenter(diags, verbose=True)
    return mean


if __name__ == '__main__':
    data1 = load_data.load_data('../data', 'Cora')
    with open('../jctops/rewired', 'rb') as f:
        data2 = pickle.load(f)
    ph1 = estimate_ph(data1)

    with open('ph_before', 'wb') as f:
        pickle.dump(ph1, f)

    ph2 = estimate_ph(data2)

    with open('ph_after', 'wb') as f:
        pickle.dump(ph2, f)

    print(gd.bottleneck_distance(ph1, ph2))

    # 23.333333
