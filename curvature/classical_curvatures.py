from typing import Tuple

import networkx as nx


def compute_curvature_edge(G: nx.Graph, e: Tuple[int, int], curv_type: str) -> int:
    """
    Compute a specified type of discrete curvature for a graph.
    :param G: (undirected) graph under consideration.
    :param e: edge under consideration.
    :param curv_type: type of discrete curvature to compute.
    :return: curvature of the edge.
    """
    v1, v2 = e
    if curv_type == '1d':
        return 4 - G.degree[v1] - G.degree[v2]
    elif curv_type == 'augmented':
        v1_nbr = set(G.neighbors(v1))
        v2_nbr = set(G.neighbors(v2))
        triangles = v1_nbr.intersection(v2_nbr)
        return 4 - G.degree[v1] - G.degree[v2] + 3 * len(triangles)
    elif curv_type == 'haantjes':
        v1_nbr = set(G.neighbors(v1))
        v2_nbr = set(G.neighbors(v2))
        triangles = v1_nbr.intersection(v2_nbr)
        return len(triangles)
    else:
        raise Exception(f'Method {curv_type} not available.')


def compute_curvature_graph(G: nx.Graph, curv_type: str) -> dict:
    """
    Compute discrete curvature for all the edges of a given graph.
    :param G: (undirected) graph under consideration.
    :param curv_type: type of discrete curvature to compute.
    :return: dictionary containing the values of curvature for all graph edges.
    """
    if curv_type == "1d" or curv_type == "augmented" or curv_type == 'haantjes':
        curv_dict = {}
        for (v1, v2) in G.edges():
            if v1 not in curv_dict:
                curv_dict[v1] = {}
            curv_dict[v1][v2] = compute_curvature_edge(G, (v1, v2), curv_type)
        return curv_dict
    else:
        assert True, 'Method %s not available.' % curv_type
