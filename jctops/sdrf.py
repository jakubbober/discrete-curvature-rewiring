import copy
import logging
import pickle

import numpy as np
import torch
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj, remove_self_loops, to_undirected
from tqdm import tqdm

from jctops.curvature import ricci, ricci_post_delta
from utils.load_data import load_data

_logger = logging.getLogger(__name__)


def softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()


def sdrf_w_cuda(data, loops=10, remove_edges=True, tau=1, is_undirected=True, remove_threshold=0.5):
    edge_index = data.edge_index
    if is_undirected:
        edge_index = to_undirected(edge_index)
    A = to_dense_adj(remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    G = to_networkx(data, node_attrs=['x'])
    if is_undirected:
        G = G.to_undirected()
    A = A.cuda()
    C = torch.zeros(N, N).cuda()

    for x in tqdm(range(loops)):
        can_add = True
        _logger.warning(f'\n#######\nLoop {x}!\n#######\n')
        ricci(A, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N
        _logger.warning(f'Min curvature {C[x, y]} at {x}->{y}')

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [x]
        _logger.warning(f'x has successors {x_neighbors}')
        _logger.warning(f'y has successors {y_neighbors}')
        candidates = []

        for i in x_neighbors:
            if (i != y) and (not G.has_edge(i, y)):
                candidates.append((i, y))
        for j in y_neighbors:
            if (j != x) and (not G.has_edge(j, x)):
                candidates.append((x, j))
        if len(candidates):
            D = ricci_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append((D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item())

            k, l = candidates[np.random.choice(range(len(candidates)), p=softmax(np.array(improvements), tau=tau))]
            _logger.warning(f'New edge chosen is {k, l}')
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                _logger.warning(f'Nothing changed this round - breaking')
                break

        if remove_edges:
            # Set curvature of nonexistent edges to -inf for deleting the highest curved edges
            # to avoid deleting nonexistent edges if the current max curvature is negative.
            C_remove = copy.deepcopy(C)
            C_remove[C_remove == 0] = float('-inf')
            ix_max = C_remove.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C_remove[x, y] > remove_threshold:
                _logger.warning(f'Max curvature {C_remove[x, y]} at {x}->{y} - removing edge')
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                _logger.warning(f'Max curvature is <= {remove_threshold} - leaving in place')
                if can_add is False:
                    _logger.warning(f'Nothing changed this round - breaking')
                    break

    return from_networkx(G)
