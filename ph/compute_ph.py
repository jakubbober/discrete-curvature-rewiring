import pickle
import time

import matplotlib
import networkx as nx
import numpy as np
# import ripserplusplus as rpp
import torch_geometric.data
from gtda.plotting import plot_diagram
from matplotlib import pyplot as plt
from networkx import fast_gnp_random_graph, adjacency_matrix
# from ripser import ripser
from persim import plot_diagrams
from gtda.homology import VietorisRipsPersistence
from torch_geometric.utils import to_networkx
import gudhi

from compute_curvature import compute_curvature
from ph import ApproxPH
from utils.load_data import load_data

matplotlib.use('WebAgg')
# fig, (ax1, ax2) = plt.subplots(1, 2)


def random_graph_adj(nodes, ax=None, p=0.3, directed=False, w_low=1, w_high=20):
    G = fast_gnp_random_graph(int(nodes), p, directed=directed)

    # nx.draw(G, ax=ax1)
    # plt.show()
    # adj2 = G.adjacency()
    adj = adjacency_matrix(G)
    # print(adj.toarray())
    # print(adj.toarray())
    a = np.random.randint(w_low, w_high, adj.shape)
    # with open('a', 'rb') as f:
    #     a = pickle.load(f)
    # print(a)
    a = (a + a.T) // 2
    # print(a)
    b = adj.multiply(a)
    # print(b.toarray())
    # print('stop')
    return b


# def compute_ph(data: torch_geometric.data.Data, curv_type: str, filename: str=None):
#     G = to_networkx(data, node_attrs=['x'])
#     G, C = compute_curvature(G, curv_type)
#     # print(C)
#     C[C != 0] = C[C != 0] + C.min() + 1
# 
#     pairs = ripscomplex(C, hdim=3)
#     plot_diagrams([np.array((pairs[0].tolist())).reshape((len(pairs[0]), 2)),
#                    np.array((pairs[1].tolist())).reshape((len(pairs[1]), 2)),
#                    np.array((pairs[2].tolist())).reshape((len(pairs[2]), 2))]
#                   , show=False, ax=ax2)
# 
#     with open(filename, 'wb') as f:
#         pickle.dump(pairs, f)
# 
#     return pairs


def compute_ph_giotto(C, i=None, filename: str=None):
    # print(len(data.edge_index))
    # G = to_networkx(data, node_attrs=['x'])
    # G, C = compute_curvature(G, curv_type)

    VR = VietorisRipsPersistence(metric='precomputed', reduced_homology=True, collapse_edges=True, n_jobs=-1, homology_dimensions=tuple([1]), max_edge_length=2)

    # C = random_graph_adj(30, p=0.9, ax=None).toarray() * 1.0
    # C[C != 0] += 1 - C.min()
    start = time.time()
    if i:
        points = VR.fit_transform([C[:i, :i]])
    else:
        points = VR.fit_transform([C])
    # print(time.time() - start)

    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(points, f)

    return points


def ripscomplex(X, hdim=1):
    pairs = rpp.run("--format distance --dim " + str(hdim), X)
    return pairs


def compute_ph_subsample(data: torch_geometric.data.Data, curv_type: str, i=None, filename: str=None):
    G = to_networkx(data, node_attrs=['x'])
    G, C = compute_curvature(G, curv_type)
    ApproxPH.get_subsample()


if __name__ == '__main__':
    # adj = random_graph_adj(30, p=0.9, ax=None).toarray() * 1.0
    # VR = VietorisRipsPersistence(metric='precomputed', reduced_homology=False)
    # diagrams = VR.fit_transform_plot([adj])
    # print(diagrams)
    # VR.plot(diagrams, sample=0)
    # plt.show()
    # print(adj.toarray())
    # pairs = ripscomplex(adj, hdim=3)
    # # print(pairs)
    # plot_diagrams([np.array((pairs[0].tolist())).reshape((len(pairs[0]), 2)),
    #                np.array((pairs[1].tolist())).reshape((len(pairs[1]), 2)),
    #                np.array((pairs[2].tolist())).reshape((len(pairs[2]), 2))]
    #               , show=False, ax=ax2)
    # plt.show()


    data = load_data('../data', 'Cora')

    # a = 2000
    G = to_networkx(data, node_attrs=['x']).to_undirected()
    G, C = compute_curvature(G, 'formanCurvature')
    print(len(G.edges))
    results = compute_ph_giotto(C)

    print(results)

    # a *= 2

    with open('forman_Cora_ph_long', 'rb') as f:
        print(pickle.dump(results, f))

    # with open('forman_Cora_ph', 'rb') as f:
    #     ph = pickle.load(f)
    # print(ph[0])
    # fig = plot_diagram(ph[0])
    # fig.show()
