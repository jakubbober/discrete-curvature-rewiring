import pickle
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
from copy import deepcopy
import torch_geometric.transforms as T

import numpy as np

transform = T.Compose([
    T.RandomNodeSplit(num_val=499, num_test=500),
    T.TargetIndegree(),
])
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=transform)
data = dataset[0]


def compute_edge_fc(edge_attr, start_attr, end_attr):
    sum_start = sum([1 / np.sqrt(edge_attr * attr) for attr in start_attr[:, 0]])
    sum_end = sum([1 / np.sqrt(edge_attr * attr) for attr in end_attr[:, 0]])
    return edge_attr * (1 / edge_attr + 1 / edge_attr - sum_start - sum_end)


if __name__ == '__main__':
    data_fc = deepcopy(data)
    for i, (start, end) in tqdm(enumerate(data_fc.edge_index.T)):
        start = int(start)
        end = int(end)

        _, start_edges, _, start_edges_mask = k_hop_subgraph(start, 1, data.edge_index)
        start_attr = data.edge_attr[start_edges_mask]
        _, end_edges, _, end_edges_mask = k_hop_subgraph(end, 1, data.edge_index)
        end_attr = data.edge_attr[end_edges_mask]

        mask_start = [start in start_edges[:, j] for j in range(start_edges.shape[1])]
        start_edges = start_edges[:, mask_start]
        start_attr = start_attr[mask_start, :]

        mask_start2 = [not (start in col and end in col) for col in start_edges.T]
        start_edges = start_edges[:, mask_start2]
        start_attr = start_attr[mask_start2, :]

        mask_end = [end in end_edges[:, j] for j in range(end_edges.shape[1])]
        end_edges = end_edges[:, mask_end]
        end_attr = end_attr[mask_end, :]

        mask_end2 = [not (start in col and end in col) for col in end_edges.T]
        end_edges = end_edges[:, mask_end2]
        end_attr = end_attr[mask_end2, :]

        data_fc.edge_attr[i, 0] = compute_edge_fc(data.edge_attr[i, 0], start_attr, end_attr)

        with open('data_fc.pkl', 'wb') as out:
            pickle.dump(data_fc, out, pickle.HIGHEST_PROTOCOL)
