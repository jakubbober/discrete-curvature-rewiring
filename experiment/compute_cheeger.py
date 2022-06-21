import os
import pickle
from copy import deepcopy
from random import randint

import matplotlib
import networkx as nx
import pandas as pd
import torch_geometric.data
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from experiment.data_loader import DataLoader

matplotlib.use('WebAgg')


def random_subset(nodes):
    subset = set()
    for node in nodes:
        if randint(0, 1) == 0:
            subset.add(node)
    return subset


def vol(S):
    # print(S.degree())
    return sum([val for (node, val) in S.degree()])


def boundary_size(G, S):
    result = 0
    for i, j in G.edges:
        if i in S.nodes and j not in S.nodes:
            result += 1
    return result


def cheeger_S(G, S):
    b = boundary_size(G, S)
    diff = deepcopy(G)
    diff.remove_nodes_from(S.nodes)
    m = min(vol(S), vol(diff))
    return float('inf') if m == 0 else b / m


def estimate_cheeger(data: torch_geometric.data.Data, iterations: int) -> float:
    """
    MC estimator of the Cheeger constant
    :param data: graph data for which to estimate the Cheeger constant.
    :param iterations: number of iterations.
    :return: Cheeger constant estimation.
    """
    G = to_networkx(data, to_undirected=True)
    result = float('inf')
    all_results = []
    for _ in tqdm(range(iterations)):
        S = random_subset(set(i for i in range(len(G.nodes))))
        cheeger = cheeger_S(G, G.subgraph(S))
        all_results.append(cheeger)
        if result > cheeger:
            result = cheeger
    return result, all_results


if __name__ == '__main__':
    plt.rc('font', size=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    result = {}
    # dnames = ['Cornell', 'Texas', 'Wisconsin']
    dnames = ['Cora', 'Citeseer', 'Pubmed', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor',
              'Computers', 'Photo', 'CoauthorCS']
    for dname in dnames:
        # result[dname] = {}
        # for i in [30, 100, 300, 1000, 3000]:
        dt = DataLoader(dname, undirected=True, data_dir='dt')
        # if dname not in ('Chameleon', 'Squirrel', 'Actor', 'CoauthorCS', 'Computers', 'Photo', 'Pubmed'):
        #     with open(f'edge_indices/{dname}/edge_index_1d_best.pk', 'rb') as f:
        #         edge_index = pickle.load(f)
        #         dt.data.edge_index = edge_index
        # else:
        #     with open(f'edge_indices/{dname}/edge_index_1d.pk', 'rb') as f:
        #         edge_index = pickle.load(f)
        #         dt.data.edge_index = edge_index
        # _, r = estimate_cheeger(dt.data, 1000000)
        # result[d][i] = round(r, 4)
        # with open(f'mc_mil/{dname}.pk', 'wb') as f:
        #     pickle.dump(r, f)
        # pd.DataFrame(result).to_excel('mc.xlsx')

        with open(f'past/mc/{dname}.pk', 'rb') as f:
            cheegers = pickle.load(f)
        # with open(f'mc/{dname}.pk', 'rb') as f:
        #     cheegers3k = pickle.load(f)
        # print(1)
        result = []
        result3k = []
        m = float('inf')
        for c in cheegers:
            if c < m:
                m = c
            result.append(m)
        plt.plot(range(1, len(result) + 1), result, label=dname)

    plt.title('Cheeger Constant Monte Carlo Estimation')
    plt.xlabel('Number of iterations')
    plt.ylabel('Estimate')
    plt.legend(loc='lower left')
    plt.show()

