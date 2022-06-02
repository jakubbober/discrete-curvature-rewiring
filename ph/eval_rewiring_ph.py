from typing import List

import gudhi
import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx

# from curvature.compute_curvature import compute_curvature
# from jctops.sdrf import sdrf_w_cuda
from ph import compute_ph
from rewiring.sdrf_cuda_bfc import sdrf_cuda_bfc


def eval_rewiring_ph(nodes: int, p: float, n: int = 20, rewiring_loops=100) -> List[float]:
    """
    Evaluate the robustness of the SDRF rewiring algorithm in terms of the differences
    in persistence homology bottleneck distances of before and after rewiring for multiple random graphs.
    :param nodes: number of nodes for the random graphs.
    :param p: probability of creating an edge between each two nodes.
    :param n: number of random graphs for which to compute the bottleneck distance.
    :param rewiring_loops: number of loops for which to run the SDRF algorithm.
    :return: list of bottleneck distances for each random graph between before and after SDRF.
    """
    bnds = []  # bottleneck distances list
    for i in range(n):
        adj1 = compute_ph.random_graph_adj(nodes, p)
        G1 = nx.from_scipy_sparse_matrix(adj1)
        # G1, C1 = compute_curvature(G1, 'formanCurvature')
        # print(adj1.max())
        ph1 = compute_ph.compute_ph(adj1)
        print(ph1)
        nx.set_node_attributes(G1, 0, 'x')
        data1 = from_networkx(G1)

        data2 = sdrf_cuda_bfc(data1, loops=rewiring_loops, remove_edges=True, removal_bound=0, tau=100, is_undirected=True)
        G2 = to_networkx(data2)

        adj2 = compute_ph.random_graph_adj(nodes, p)
        # G2, C2 = compute_curvature(G2, 'formanCurvature')
        # print(adj2.max())
        ph2 = compute_ph.compute_ph(adj2)
        print(ph2)
        bnds.append(gudhi.bottleneck_distance(ph1, ph2))

    return bnds


if __name__ == '__main__':
    distances = eval_rewiring_ph(100, 0.1)
    print(distances)
    # [7.103827476501465, 6.23202109336853, 5.862726211547852, 8.007595539093018, 6.445271968841553, 6.0, 8.687578201293945, 6.889535188674927, 6.536870956420898, 5.5, 5.0, 6.75018310546875, 8.394240379333496, 7.472712993621826, 5.201995372772217, 6.515475511550903, 6.820135116577148, 5.968339920043945, 6.847666263580322, 7.434313774108887]
