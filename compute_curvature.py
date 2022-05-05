import argparse
from time import time

import torch
# from forman_curvature import FormanRicci
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from torch_geometric.utils import to_networkx
import networkx as nx

from utils import load_data


def bfc_edge(G: nx.Graph, v1: int, v2: int) -> float:
    deg1 = G.degree[v1]
    deg2 = G.degree[v2]
    deg_min = min(deg1, deg2)
    if deg_min == 1:
        return 0
    deg_max = max(deg1, deg2)

    S1_1 = set(G[v1])
    S1_2 = set(G[v2])

    triangles = S1_1.intersection(S1_2)
    squares_1 = set(k for k in S1_1.difference(S1_2) if k != v2 and set(G[k]).intersection(S1_2).difference(S1_1.union({v1})))
    squares_2 = set(k for k in S1_2.difference(S1_1) if k != v1 and set(G[k]).intersection(S1_1).difference(S1_2.union({v2})))
    # print(squares_1, squares_2)
    if len(squares_1) == 0 or len(squares_2) == 0:
        return 2 / G.degree[v1] + 2 / G.degree[v2] - 2 + 2 * len(triangles) / deg_max + len(
        triangles) / deg_min

    A = nx.adj_matrix(G)

    gamma = max(max([(A[k] @ (A[v2] - A[v1].multiply(A[v2])).T)[0, 0] - 1 for k in squares_1]),
                max([(A[k] @ (A[v1] - A[v2].multiply(A[v1])).T)[0, 0] - 1 for k in squares_2]))
    if gamma == 0:
        print('WRONG')
    return 2 / G.degree[v1] + 2 / G.degree[v2] - 2 + 2 * len(triangles) / deg_max + len(
        triangles) / deg_min + 1 / gamma / deg_max * (len(squares_1) + len(squares_2))

def bfc(G: nx.Graph) -> nx.Graph:
    for v1, v2 in G.edges:
        G[v1][v2]['bfc'] = bfc_edge(G, v1, v2)

    return G


def compute_curvature(G, curv_type, dataset=None):
    # save file if dataset name given
    # start = time()
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
        # print(FormanRicci(G, method="1d").compute_ricci_curvature())
        frc = FormanRicci(G, method="1d")
        frc.compute_ricci_curvature()
        G = frc.G
        for v1, v2 in G.edges:
            # if G[v1][v2]['formanCurvature'] != G[v2][v1]['formanCurvature']:
            #     print((v1, v2))
            # else:
                # print(1)
            C[v1, v2] = G[v1][v2]['formanCurvature']
            C[v2, v1] = C[v1, v2]
    else:
        G = bfc(G)
    # print("Curvature computation time:", time() - start)

    if dataset:
        out = ""
        for v1, v2 in G.edges:
            out += f"{v1} {v2} {G[v1][v2][curv_type]}\n"
        with open(f"graph_{dataset}_{curv_type}.edge_list", "w") as f:
            f.write(out)
    return G, C.cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CGNN')
    parser.add_argument('--data_path', type=str, help="Path to the data files.", default='./data')
    parser.add_argument('--dataset', type=str, help="Name of the dataset", required=True)
    parser.add_argument('--curv_type', type=str, help="Curvature type", required=True)
    parser.add_argument('--save_file', action='store_true', help="Whether to save the .edge_list file", default=False)
    args = parser.parse_args()

    dt = load_data(args.data_path, args.dataset)

    G = to_networkx(dt).to_undirected()

    compute_curvature(G, args.curv_type, args.save_file)
