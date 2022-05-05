import pickle

import gudhi
import matplotlib
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx

from compute_curvature import compute_curvature
from jctops.sdrf import sdrf_w_cuda
from ph import compute_ph

matplotlib.use('WebAgg')


if __name__ == '__main__':
    # adj = random_graph_adj(10, p=0.1)
    # nx.draw(fast_gnp_random_graph(100, p=0.1, directed=False))
    # plt.show()
    bnds = []
    for i in range(10):
        # G1 = nx.fast_gnp_random_graph(200, 0.3)
        # nx.set_node_attributes(G1, 0, 'x')
        # print(len(G1.edges))
        # adj1 = nx.adjacency_matrix(G1)
        adj1 = compute_ph.random_graph_adj(100, p=0.1)
        G1 = nx.from_scipy_sparse_matrix(adj1)
        G1, C = compute_curvature(G1, 'formanCurvature')
        # print(len(.edges))
        ph1 = compute_ph.compute_ph_giotto(C)[0, :, [0,1]].T

        nx.set_node_attributes(G1, 0, 'x')
        data1 = from_networkx(G1)
        data2 = sdrf_w_cuda(data1, loops=100)
        G2 = to_networkx(data2)
        adj2 = nx.adjacency_matrix(G2)
        ph2 = compute_ph.compute_ph_giotto(adj2)[0, :, [0,1]].T
        bnds.append(gudhi.bottleneck_distance(ph1, ph2))

        with open('bnds', 'wb') as f:
            pickle.dump(bnds, f)
