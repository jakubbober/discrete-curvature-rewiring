import logging
import pickle

import numpy as np
# import pandas as pd
from copy import deepcopy
import torch
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj, remove_self_loops, to_undirected

from balanced_forman import ricci, ricci_post_delta

_logger = logging.getLogger(__name__)


def softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()


def sdrf_w_cuda(
        data,
        loops=10,
        remove_edges=True,
        tau=1,
        is_undirected=False,
):
    edge_index = data.edge_index
    if is_undirected:
        edge_index = to_undirected(edge_index)
    A = to_dense_adj(remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    G = to_networkx(data, node_attrs=['x'])
    if is_undirected:
        G = G.to_undirected()
    A = A.cuda()

    for x in range(loops):
        can_add = True
        _logger.warn(f'\n#######\nLoop {x}!\n#######\n')
        # C = ricci(A)  # TODO change to other curvatures
        frc = FormanRicci(G, method='1d')
        frc.compute_ricci_curvature()
        # orc = OllivierRicci(G)
        # orc.compute_ricci_curvature()
        C = torch.zeros(N, N).cuda()
        for v1, v2 in frc.G.edges:
            C[v1, v2] = frc.G[v1][v2]['formanCurvature']
        # for v1, v2 in orc.G.edges:
        #     C[v1, v2] = orc.G[v1][v2]['ricciCurvature']
        if x == loops - 1:
            out_balanced = ""
            for v1, v2 in G.edges:
                out_balanced += f"{v1} {v2} {C[v1, v2]}\n"
            with open('data/BalancedForman/balanced_forman.edge_list', 'w') as file:
                file.write(out_balanced)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N
        _logger.warn(f'Min curvature {C[x, y]} at {x}->{y}')

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
            # x_neighbors = list(G.neighbors(x))
            # y_neighbors = list(G.neighbors(y))
        else:
            raise Exception('not undirected')
            # x_neighbors = list(G.successors(x)) + [x]
            # y_neighbors = list(G.predecessors(y)) + [y]
        _logger.warn(f'x has successors {x_neighbors}')
        _logger.warn(f'y has successors {y_neighbors}')
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        # for i in x_neighbors:
        #     if (i != y) and (not G.has_edge(i, y)):
        #         candidates.append((i, y))
        # for i in y_neighbors:
        #     if (i != x) and (not G.has_edge(i, x)):
        #         candidates.append((x, i))
        if len(candidates):
            D = ricci_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append((D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item())

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)),
                    p=softmax(np.array(improvements), tau=tau)
                )
            ]
            _logger.warn(f'New edge chosen is {k, l}')
            G.add_edge(k, l)

            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                _logger.warn(f'Nothing changed this round - breaking')
                break

        if remove_edges:
            ix_max = C.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > 0.2:
                _logger.warn(f'Max curvature {C[x, y]} at {x}->{y} - removing edge')
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                _logger.warn('Max curvature is <= 1.2 - leaving in place')
                if can_add is False:
                    _logger.warn(f'Nothing changed this round - breaking')
                    break

    return from_networkx(G)
