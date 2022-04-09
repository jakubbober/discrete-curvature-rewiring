import networkx as nx
import numpy as np
import torch_geometric
from torch_geometric.utils import from_networkx

from compute_curvature import compute_curvature, bfc_edge
from copy import deepcopy


def sdrf(G: torch_geometric.data.Data, curv_type: str, tau: float = 1, max_iter: int = 100, C: float = None) -> nx.Graph:
    def softmax(a, tau: float=1.):
        exp_a = np.exp(a * tau)
        return exp_a / exp_a.sum()

    for n in range(max_iter):
        print(n)
        change = False
        G = compute_curvature("./data", "Cora", "bfc")
        v1, v2 = min(list(G.edges), key=lambda x: G[x[0]][x[1]][curv_type])
        min_curv = G[v1][v2][curv_type]
        neighbors1 = list(G.neighbors(v1)) + [v1]
        neighbors2 = list(G.neighbors(v2)) + [v2]
        candidates = []
        for i in neighbors1:
            for j in neighbors2:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))
        x = []
        for i, j in candidates:
            G_new = deepcopy(G)
            G_new.add_edge(i, j)
            x.append(bfc_edge(G_new, v1, v2) - min_curv)

        if candidates:
            k, l = candidates[np.random.choice(range(len(candidates)), p=softmax(np.array(x), tau=tau))]
            G.add_edge(k, l)
        else:
            change = True

        v1, v2 = max(G.edges, key=lambda x: G[x[0]][x[1]][curv_type])
        max_curv = G[v1][v2][curv_type]

        if C and max_curv > C:
            G.remove_edge(v1, v2)
            change = True

        if not change:
            break

    return from_networkx(G)
