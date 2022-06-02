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

    left =  lambda1 / 2
    right = np.sqrt(2 * lambda1)

    return left, right


if __name__ == '__main__':
    dnames = ['Cora', 'Citeseer', 'Cornell', 'Texas', 'Wisconsin', 'Chameleon', 'Squirrel']
    for d in dnames:
        dt = DataLoader(d, undirected=True, data_dir='dt')
        # for million: 0.05952, 0.07857
        # for 1000: 0.15152, 0.16379, 0.23370
        res = cheeger_bounds(dt.data)
        print(res)
