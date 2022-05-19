import numpy as np
from torch_geometric.utils import from_networkx, to_networkx
from tqdm import tqdm

from bronstein_paper.utils import softmax
from curvature.frc_rewritten import compute_ricci_curvature, compute_ricci_curvature_edge


def sdrf_no_cuda(data, curv_type, loops, remove_edges=True, removal_bound=0.5, tau=1, is_undirected=True):
    G = to_networkx(data, node_attrs=['x'], to_undirected=is_undirected)

    for x in tqdm(range(loops)):
        can_add = True
        curv_dict = compute_ricci_curvature(G, curv_type)
        x, y = min(G.edges, key=lambda e: curv_dict[e[0]][e[1]])

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]  # changed from x to y
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]  # changed from x to y
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append(sorted((i, j)))

        if len(candidates):
            improvements = []
            for (i, j) in candidates:
                before = compute_ricci_curvature_edge(G, x, y, curv_type)
                G.add_edge(i, j)
                after = compute_ricci_curvature_edge(G, x, y, curv_type)
                improvements.append(after - before)
                G.remove_edge(i, j)
            k, l = sorted(candidates[np.random.choice(range(len(candidates)), p=softmax(np.array(improvements), tau=tau))])
            G.add_edge(k, l)
        else:
            can_add = False
            if not remove_edges:
                break
        if remove_edges:
            x, y = max([edge for edge in G.edges if edge != (k, l)], key=lambda e: curv_dict[e[0]][e[1]])
            if curv_dict[x][y] > removal_bound:
                G.remove_edge(x, y)
            else:
                if can_add is False:
                    break

    return from_networkx(G)
