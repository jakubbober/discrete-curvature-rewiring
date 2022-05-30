import numpy as np
from torch_geometric.utils import from_networkx, to_networkx
from tqdm import tqdm

from utils.softmax import softmax
from curvature.classical_curvatures import compute_curvature_graph, compute_curvature_edge


def sdrf_no_cuda(data, curv_type, loops, remove_edges, removal_bound, tau):
    """
    Perform SDRF graph rewiring using the given classical discrete curvature type.
    :param data: data to be rewired (undirected by default in this work).
    :param curv_type: type of discrete curvature used for the rewiring.
    :param loops: number of edge addition/deletion iterations.
    :param remove_edges: whether to delete highly curved edges each iteration to compensate for the addition.
    :param removal_bound: curvature lower bound of deleting edges (delete edges only with higher curvature).
    :param tau: parameter specifying the randomness of choosing candidate edge to add; if infinite, max value is chosen.
    :return: rewired data.
    """
    G = to_networkx(data, node_attrs=['x'], to_undirected=True)

    for _ in tqdm(range(loops)):
        can_add = True
        curv_dict = compute_curvature_graph(G, curv_type)

        # Choose the edge with the lowest curvature.
        x, y = min(G.edges, key=lambda e: curv_dict[e[0]][e[1]])

        x_neighbors = list(G.neighbors(x)) + [x]
        y_neighbors = list(G.neighbors(y)) + [y]

        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    # Sort so that the lower index is first to remove ambiguity.
                    candidates.append(sorted((i, j)))

        if len(candidates):
            improvements = []
            for (i, j) in candidates:
                before = compute_curvature_edge(G, (x, y), curv_type)
                G.add_edge(i, j)
                after = compute_curvature_edge(G, (x, y), curv_type)
                improvements.append(after - before)
                G.remove_edge(i, j)
            # Choose the supporting edge to add with softmax probability.
            # Sort so that the lower index is first to remove ambiguity.
            k, l = sorted(
                candidates[np.random.choice(range(len(candidates)), p=softmax(np.array(improvements), tau=tau))])
            G.add_edge(k, l)
        else:
            can_add = False
            if not remove_edges:
                break
        if remove_edges:
            # Choose the edge with the highest curvature and potentially delete it.
            x, y = max([edge for edge in G.edges if edge != (k, l)], key=lambda e: curv_dict[e[0]][e[1]])
            if curv_dict[x][y] > removal_bound:
                G.remove_edge(x, y)
            else:
                if can_add is False:
                    break

    return from_networkx(G)
