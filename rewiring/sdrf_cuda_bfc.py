"""
Adpoted from https://github.com/jctops/understanding-oversquashing/blob/main/gdl/src/gdl/curvature/cuda.py.
"""
import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import from_networkx, to_undirected, to_dense_adj, to_networkx, remove_self_loops
from tqdm import tqdm

from curvature.bfc_cuda import balanced_forman_curvature, balanced_forman_post_delta
from utils.softmax import softmax


def sdrf_cuda_bfc(data: torch_geometric.data.Data, loops: int, remove_edges: bool,
                  removal_bound: float, tau: int, is_undirected: bool) -> torch_geometric.data.Data:
    """
    Perform SDRF graph rewiring using the CUDA version for BFC.
    :param data: data to be rewired.
    :param loops: number of edge addition/deletion iterations.
    :param remove_edges: whether to delete highly curved edges each iteration to compensate for the addition.
    :param removal_bound: curvature lower bound of deleting edges (delete edges only with higher curvature).
    :param tau: parameter specifying the randomness of choosing candidate edge to add; if infinite, max value is chosen.
    :param is_undirected: flag specifying whether the data is undirected.
    :return: rewired data.
    """
    edge_index = data.edge_index
    if is_undirected:
        edge_index = to_undirected(edge_index)
    A = to_dense_adj(remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    G = to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    A = A.cuda()
    C = torch.zeros(N, N).cuda()

    for _ in tqdm(range(loops)):
        can_add = True
        balanced_forman_curvature(A, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        if len(candidates):
            D = balanced_forman_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break

    return from_networkx(G)
