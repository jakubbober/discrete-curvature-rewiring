import time

from rewiring.sdrf_cuda_bfc import sdrf_cuda_bfc
from rewiring.sdrf_no_cuda import sdrf_no_cuda


def rewire(dt, curv_type, max_iterations, removal_bound, tau):
    if curv_type == 'bfc':
        dt = sdrf_cuda_bfc(dt, loops=max_iterations, remove_edges=True,
                           removal_bound=removal_bound, tau=tau, is_undirected=True)
    elif curv_type is not None:
        dt = sdrf_no_cuda(dt, curv_type, loops=max_iterations, remove_edges=True,
                          removal_bound=removal_bound, tau=tau)
    return dt.edge_index
