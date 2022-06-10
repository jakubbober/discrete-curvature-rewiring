from copy import deepcopy
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch_geometric.data
from torch_geometric.utils import to_networkx

from experiment.data_loader import DataLoader

matplotlib.use('WebAgg')


def compute_adj_powers(data: torch_geometric.data.Data, r: int) -> List[float]:
    """
    Compute the normalised augmented adjacency matrix and smallest non-zero values of its powers.
    :param data: graph data of interest.
    :param r: number of powers to compute.
    :return: list of the smallest non-zero values for each power.
    """
    G = to_networkx(data)

    adj = nx.adjacency_matrix(G)
    A = adj + np.eye(adj.shape[0])

    D_inv_sq = np.sqrt(np.diag([1 / (1 + val) for (node, val) in G.degree()]))
    
    A_norm_aug = D_inv_sq @ A @ D_inv_sq

    A_pow = deepcopy(A_norm_aug)

    result = []

    edges = []

    for _ in range(r):
        i, j = np.where(A_pow == np.min(A_pow[np.nonzero(A_pow)]))
        # print(np.where(A_pow == np.min(A_pow[np.nonzero(A_pow)])))

        edges.append((i, j))
        result.append(np.min(A_pow[np.nonzero(A_pow)]))
        A_pow = A_pow @ A_norm_aug
    return result


if __name__ == '__main__':
    dnames = ['Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel']
    # dnames = ['Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor']
    i = 50
    for d in dnames:
        # print(d)
        dt = DataLoader(d, undirected=True, data_dir='dt')
        res = compute_adj_powers(dt.data, i)
        plt.plot(range(1, len(res) + 1), res, label=d)
    plt.yscale('log')
    plt.xscale('log')
    plt.title('Log-log plot of minimum non-zero values of powers of normalised augmented adjacency matrix')
    plt.xlabel('Power')
    plt.ylabel('Minimum non-zero value')
    # plt.xticks(range(1, i + 1, 5))
    plt.legend()
    plt.show()
