from typing import Union

import networkx as nx
import numpy as np
import scipy.sparse.csr
from gtda.homology import VietorisRipsPersistence


def random_graph_adj(nodes: int, p: float = 0.3, w_low: int = 1, w_high: int = 100) -> np.ndarray:
    """
    Create an adjacency matrix of a random graph.
    :param nodes: number of nodes for the random graph.
    :param p: probability of creating an edge between each two nodes.
    :param w_low: minimum possible edge weight (inclusive).
    :param w_high: maximum possible edge weight (exclusive).
    :return: weighted adjacency matrix of the random graph.
    """
    G = nx.fast_gnp_random_graph(int(nodes), p, directed=False)
    adj = nx.adjacency_matrix(G)

    a = np.random.randint(w_low, w_high, adj.shape)
    a = (a + a.T) // 2

    b = adj.multiply(a)
    return b


def compute_ph(C: Union[np.ndarray, scipy.sparse.csr.csr_matrix]) -> np.ndarray:
    """
    Compute persistent homology points for homology dimension of 1.
    :param C: weighted adjacency matrix.
    :return: PH birth-death points (without the PH dimension, as it is always 1).
    """
    VR = VietorisRipsPersistence(metric='precomputed', reduced_homology=True, collapse_edges=True, n_jobs=-1,
                                 homology_dimensions=tuple([1]), max_edge_length=2)
    points = VR.fit_transform([C])
    return points[0, :, [0, 1]].T  # parse the giotto_tda format to gudhi format
