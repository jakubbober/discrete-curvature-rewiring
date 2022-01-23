import pickle

from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph
from copy import deepcopy
import torch_geometric.transforms as T
from tqdm import tqdm


in_features = 1433  # dataset.num_node_features
out_features = 7  # dataset.num_classes


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(in_features, 16)
        self.conv2 = GCNConv(16, out_features)

    def forward(self, dt):
        x, edge_index = dt.x, dt.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train_data(data_path):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    model = GCN().to(device)
    with open(data_path, 'rb') as data:
        data_fc = pickle.load(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in tqdm(range(200)):
        optimizer.zero_grad()
        out = model(data_fc)
        loss = F.nll_loss(out[data_fc.train_mask], data_fc.y[data_fc.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data_fc).argmax(dim=1)
    correct = (pred[data_fc.test_mask] == data_fc.y[data_fc.test_mask]).sum()
    acc = int(correct) / int(data_fc.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')


if __name__ == '__main__':
    train_data('data_fc.pkl')
    train_data('data_original.pkl')
    with open('data_fc.pkl', 'rb') as dt:
        data_fc = pickle.load(dt)
    with open('data_original.pkl', 'rb') as dt:
        data = pickle.load(dt)
    print()
