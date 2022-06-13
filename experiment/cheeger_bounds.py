import pickle

import networkx as nx
import numpy as np
import scipy
from torch_geometric.utils import to_networkx

from experiment.data_loader import DataLoader


def cheeger_bounds(data):
    G = to_networkx(data, to_undirected=True)
    N = nx.normalized_laplacian_matrix(G).toarray()

    lambdas = scipy.linalg.eigh(N, eigvals_only=True)
    lambda1 = lambdas[np.where(lambdas > 0)[0]][0]

    left = lambda1 / 2
    right = np.sqrt(2 * lambda1)

    return f'{left: .2e}', f'{right: .2e}'


if __name__ == '__main__':
    # dnames = ['Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor', 'Computers', 'Photo']
    dnames = ['Pubmed', 'CoauthorCS']
    for d in dnames:
        dt = DataLoader(d, undirected=True, data_dir='dt')
        if d in ('Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin'):
            with open(f'edge_indices/{d}/edge_index_bfc_best.pk', 'rb') as f:
                edge_index = pickle.load(f)
                dt.data.edge_index = edge_index
        else:
            with open(f'edge_indices/{d}/edge_index_bfc.pk', 'rb') as f:
                edge_index = pickle.load(f)
                dt.data.edge_index = edge_index

        res = cheeger_bounds(dt.data)
        print(res)
