import pickle
from copy import deepcopy
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch_geometric.data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

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

    for _ in tqdm(range(r)):
        result.append(np.min(A_pow[np.nonzero(A_pow)]))
        A_pow = A_pow @ A_norm_aug
    # result = np.min(A_pow[np.nonzero(A_pow)])

    # for i in range(max(r) - 1):
    #     A_pow = A_pow @ A_norm_aug
    #     if i + 2 in r:
    #         cell = np.min(A_pow[np.nonzero(A_pow)])
    #         result.append(f'{cell: .2e}')
    return result


if __name__ == '__main__':
    plt.rc('font', size=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    dnames = ['Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor']
    i = 50
    for d in tqdm(dnames):
        # print(d)
        dt = DataLoader(d, undirected=True, data_dir='dt')
        # if d not in ('Chameleon', 'Squirrel', 'Actor'):
        #     with open(f'edge_indices/{d}/edge_index_1d_best.pk', 'rb') as f:
        #         edge_index = pickle.load(f)
        #         dt.data.edge_index = edge_index
        # else:
        #     with open(f'edge_indices/{d}/edge_index_1d.pk', 'rb') as f:
        #         edge_index = pickle.load(f)
        #         dt.data.edge_index = edge_index
        res = compute_adj_powers(dt.data, i)
        plt.plot(range(1, len(res) + 1), res, label=d)
    plt.yscale('log')
    plt.xscale('log')
    # plt.title('Log-log plot of minimum non-zero values of powers of normalised augmented adjacency matrix')
    plt.xlabel('Power')
    plt.ylabel('Minimum non-zero value')
    # plt.xticks(range(1, i + 1, 5))
    plt.legend()
    plt.show()


    # i = 5
    
    

    # resultt = {}
    # for dname in tqdm(dnames):
    #     for curvature in [None, '1d', 'augmented', 'haantjes', 'bfc']:
    #         dt = DataLoader(dname, undirected=True, data_dir='dt')
    #         if curvature is not None:
    #             if dname not in ('Chameleon', 'Squirrel', 'Actor'):
    #                 with open(f'edge_indices/{dname}/edge_index_{str(curvature)}_best.pk', 'rb') as f:
    #                     edge_index = pickle.load(f)
    #                     dt.data.edge_index = edge_index
    #             else:
    #                 with open(f'edge_indices/{dname}/edge_index_{str(curvature)}.pk', 'rb') as f:
    #                     edge_index = pickle.load(f)
    #                     dt.data.edge_index = edge_index
    #         res = compute_adj_powers(dt.data, [1, 5, 10, 20, 40])
    #         if dname not in resultt:
    #             resultt[dname] = {}
    #         if str(curvature) not in resultt[dname]:
    #             resultt[dname][str(curvature)] = {}
    #         resultt[dname][str(curvature)] = res
    # pd.DataFrame(resultt, index=[str(c) for c in [None, '1d', 'augmented', 'haantjes', 'bfc']]).to_excel('adj.xlsx')
