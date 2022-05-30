"""
Unused implementation of Balanced Forman Curvature with naive implementation to help understand the definition of BFC.
"""
import networkx as nx


def bfc_edge(G: nx.Graph, v1: int, v2: int) -> float:
    """
    Balanced Forman curvature computation for a given edge in a graph.
    :param G: (undirected) graph under consideration.
    :param v1: first endpoint of the edge under consideration.
    :param v2: second endpoint of the edge under consideration.
    :return: Balanced Forman curvature for the edge under consideration.
    """
    deg1 = G.degree[v1]
    deg2 = G.degree[v2]
    deg_min = min(deg1, deg2)
    if deg_min == 1:
        return 0
    deg_max = max(deg1, deg2)

    S1_1 = set(G[v1])
    S1_2 = set(G[v2])

    triangles = S1_1.intersection(S1_2)
    squares_1 = set(
        k for k in S1_1.difference(S1_2) if k != v2 and set(G[k]).intersection(S1_2).difference(S1_1.union({v1})))
    squares_2 = set(
        k for k in S1_2.difference(S1_1) if k != v1 and set(G[k]).intersection(S1_1).difference(S1_2.union({v2})))
    if len(squares_1) == 0 or len(squares_2) == 0:
        return 2 / G.degree[v1] + 2 / G.degree[v2] - 2 + 2 * len(triangles) / deg_max + len(
            triangles) / deg_min

    A = nx.adj_matrix(G)

    gamma = max(max([(A[k] @ (A[v2] - A[v1].multiply(A[v2])).T)[0, 0] - 1 for k in squares_1]),
                max([(A[k] @ (A[v1] - A[v2].multiply(A[v1])).T)[0, 0] - 1 for k in squares_2]))

    return 2 / G.degree[v1] + 2 / G.degree[v2] - 2 + 2 * len(triangles) / deg_max + len(
        triangles) / deg_min + 1 / gamma / deg_max * (len(squares_1) + len(squares_2))


def bfc(G: nx.Graph) -> nx.Graph:
    """
    Compute Balanced Forman curvature for the entire graph.
    :param G: (undirected) graph under consideration.
    :return: input graph with Balanced Forman curvature assigned for each edge.
    """
    for v1, v2 in G.edges:
        G[v1][v2]['bfc'] = bfc_edge(G, v1, v2)

    return G
