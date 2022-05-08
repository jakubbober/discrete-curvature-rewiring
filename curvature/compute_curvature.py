from typing import Tuple

import networkx as nx
import numpy as np
import torch
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci

from curvature.bfc import bfc


def compute_curvature(G: nx.Graph, curv_type, dataset=None) -> Tuple[nx.Graph, np.ndarray]:
    """
    Compute a specified type of discrete curvature for a graph.
    :param G: (undirected) graph under consideration.
    :param curv_type: type of discrete curvature to compute.
    :param dataset: name of dataset to save the file with the discrete curvatures with a relevant name;
                    if set to None, the file is not saved.
    :return: [1] the graph with assigned curvature for each edge.
             [2] the adjacency matrix weighted by the curvature.
    """
    N = len(G.edges)
    C = torch.zeros(N, N)
    if curv_type == 'ricciCurvature':
        orc = OllivierRicci(G, alpha=0.5)
        orc.compute_ricci_curvature()
        G = orc.G
        for v1, v2 in G.edges:
            C[v1, v2] = G[v1][v2]['ricciCurvature']
            C[v2, v1] = C[v1, v2]
    elif curv_type == 'formanCurvature':
        frc = FormanRicci(G, method="1d")
        frc.compute_ricci_curvature()
        G = frc.G
        for v1, v2 in G.edges:
            C[v1, v2] = G[v1][v2]['formanCurvature']
            C[v2, v1] = C[v1, v2]
    elif curv_type == 'bfc':
        G = bfc(G)
    else:
        assert True, f'curv_type of {curv_type} not available. Supported curvature types:' \
                     f' {"ricciCurvature", "formanCurvature", "bfc"}'

    if dataset:
        out = ""
        for v1, v2 in G.edges:
            out += f"{v1} {v2} {G[v1][v2][curv_type]}\n"
        with open(f"graph_{dataset}_{curv_type}.edge_list", "w") as f:
            f.write(out)

    return G, C.cpu().numpy()
