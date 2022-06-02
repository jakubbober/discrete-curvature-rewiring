from copy import deepcopy
from random import randint

import networkx as nx
import torch_geometric.data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from experiment.data_loader import DataLoader


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
    return b / min(vol(S), vol(diff))


def estimate_cheeger(data: torch_geometric.data.Data, iterations: int) -> float:
    """
    MC estimator of the Cheeger constant
    :param data: graph data for which to estimate the Cheeger constant.
    :param iterations: number of iterations.
    :return: Cheeger constant estimation.
    """
    G = to_networkx(data, to_undirected=True)
    result = float('inf')
    for _ in tqdm(range(iterations)):
        S = random_subset(set(i for i in range(len(G.nodes))))
        cheeger = cheeger_S(G, G.subgraph(S))
        if result > cheeger:
            result = cheeger
    return result


if __name__ == '__main__':
    dnames = ['Cornell', 'Texas', 'Wisconsin'] #  ['Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel']
    for d in dnames:
        dt = DataLoader(d, undirected=True, data_dir='dt')
        # for million: 0.05952, 0.07857
        # for 1000: 0.15152, 0.16379, 0.23370
        res = estimate_cheeger(dt.data, 1000)
        print(res)
