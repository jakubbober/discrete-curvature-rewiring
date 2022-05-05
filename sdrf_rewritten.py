import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.utils import from_networkx, to_networkx
from tqdm import tqdm
from forman_curvature import FormanRicci

from compute_curvature import compute_curvature, bfc_edge
from copy import deepcopy


def sdrf(data: torch_geometric.data.Data, curv_type: str, tau: float = 1, max_iter: int = 100, C: float = None) -> nx.Graph:
    def softmax(a, tau: float=1.):
        exp_a = np.exp(a * tau)
        return exp_a / exp_a.sum()

    G = to_networkx(data)
    for n in tqdm(range(max_iter)):
        # print(n)
        change = False
        G, _ = compute_curvature(G, curv_type)
        v1, v2 = min(list(G.edges), key=lambda e: G[e[0]][e[1]][curv_type])
        neighbours1 = list(G.neighbors(v1)) + [v1]
        neighbours2 = list(G.neighbors(v2)) + [v2]
        # print(neighbours1, neighbours2)
        candidates = []
        for i in neighbours1:
            if (i != v2) and not G.has_edge(i, v2):
                candidates.append((i, v2))
        for j in neighbours2:
            if (j != v1) and (not G.has_edge(j, v1)):
                candidates.append((j, v1))
        x = []
        for i, j in candidates:
            G_new = deepcopy(G)
            G_new.add_edge(i, j)
            # x.append(bfc_edge(G_new, v1, v2) - min_curv)
            if curv_type == 'formanRicci':
                frc = FormanRicci(G_new, method='1d')
                frc.compute_ricci_curvature_edge(v1, v2)
                G_new = frc.G
            elif curv_type == 'bfc':
                G_new[v1][v2][curv_type] = bfc_edge(G_new, v1, v2)
            x.append(G_new[v1][v2][curv_type])
        if candidates:
            # k, l = candidates[np.random.choice(range(len(candidates)), p=softmax(np.array(x), tau=tau))]
            am = np.argmax(x)
            k, l = candidates[am]
            # print((k, l) in G.edges)
            G.add_edge(k, l)
            G[k][l][curv_type] = x[am]
            print(f'Adding edge ({k, l}) to improve curvature of ({v1, v2}) from {G[v1][v2][curv_type]} to {x[am]}')
            # print((k, l) in G.edges)
            # print(k, l, G[k][l][curv_type])
            change = True
        # for e in G.edges:
            # try:
            #     G[e[0]][e[1]][curv_type]
            # except:
            #     print(e)
        v1, v2 = max(list(G.edges), key=lambda e: G[e[0]][e[1]][curv_type])
        max_curv = G[v1][v2][curv_type]

        if C and max_curv > C:
            G.remove_edge(v1, v2)
            change = True
            print(f'Deleting edge ({v1, v2}) with highest curvature {max_curv}')

        if not change:
            print('break')
            break

    return from_networkx(G)
