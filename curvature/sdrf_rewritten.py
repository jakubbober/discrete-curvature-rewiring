import networkx as nx
import numpy as np
import torch_geometric

import utils.load_data
from curvature.frc_rewritten import FormanRicci
from torch_geometric.utils import from_networkx, to_networkx
from tqdm import tqdm
from curvature.compute_curvature import compute_curvature
from curvature.bfc import bfc_edge
from copy import deepcopy


def sdrf(data: torch_geometric.data.Data, curv_type: str, max_iter: int = 100, C: float = None) -> nx.Graph:
    G = to_networkx(data)
    for n in tqdm(range(max_iter)):
        change = False
        # G, _ = compute_curvature(G, curv_type)
        frc = FormanRicci(G, method='augmented')
        frc.compute_ricci_curvature()
        G = frc.G
        v1, v2 = min(list(G.edges), key=lambda e: G[e[0]][e[1]][curv_type])
        neighbours1 = list(G.neighbors(v1)) + [v1]
        neighbours2 = list(G.neighbors(v2)) + [v2]
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
            if curv_type == 'formanRicci':
                frc = FormanRicci(G_new, method='augmented')
                frc.compute_ricci_curvature_edge(v1, v2)
                G_new = frc.G
            elif curv_type == 'bfc':
                G_new[v1][v2][curv_type] = bfc_edge(G_new, v1, v2)
            x.append(G_new[v1][v2][curv_type])
        if candidates:
            print(x)
            am = np.argmax(x)
            k, l = candidates[am]
            G.add_edge(k, l)
            print(f'Adding edge {k, l} to improve curvature of {v1, v2} from {G[v1][v2][curv_type]} to {x[am]}')
            G[k][l][curv_type] = x[am]
            change = True
        v1, v2 = max(list(G.edges), key=lambda e: G[e[0]][e[1]][curv_type])
        max_curv = G[v1][v2][curv_type]

        if not C or max_curv > C:
            G.remove_edge(v1, v2)
            # change = True
            print(f'Deleting edge ({v1, v2}) with highest curvature {max_curv}')

        if not change:
            print('break')
            break

    return from_networkx(G)


if __name__ == '__main__':
    data = utils.load_data.load_data('../data', 'Cora')
    sdrf(data, 'formanCurvature')
