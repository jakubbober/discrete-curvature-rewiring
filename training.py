import pickle

import numpy as np
import torch
import argparse
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv

# from curvature.adding import sdrf
from torch_geometric.utils import to_networkx

from bronstein_paper.cuda import sdrf
from jctops.sdrf import sdrf_w_cuda
from utils.load_data import load_data


# GCN model
# class GCN(torch.nn.Module):
#     def __init__(self, d_input, d_output, d_hidden=16):
#         super(GCN, self).__init__()
#
#         # Initialize the layers
#         self.conv1 = GCNConv(d_input, d_hidden)
#         self.conv2 = GCNConv(d_hidden, d_output)
#         self.out = Linear(d_hidden, d_output)
#
#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         # First Message Passing Layer (Transformation)
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = F.dropout(x, training=self.training)
#
#         # Second Message Passing Layer
#         x = self.conv2(x, edge_index)
#
#         return x


from typing import List

import torch
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset


class GCN(torch.nn.Module):
    def __init__(
        self, data: InMemoryDataset, num_classes, hidden: List[int] = [64], dropout: float = 0.5
    ):
        super(GCN, self).__init__()

        num_features = [data.x.shape[1]] + hidden + [num_classes]
        # print(1)
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(GCNConv(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_weight=edge_attr)

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)
            x = self.dropout(x)

        return torch.nn.functional.log_softmax(x, dim=1)


def eval_performance(data, iterations: int):
    results = []
    for i in range(iterations):
        # Initialize model
        print(len(set(int(y) for y in data.y)))
        model = GCN(data, len(set(int(y) for y in data.y)))

        # Use CPU
        device = torch.device("cuda")
        model = model.to(device)
        data = data.to(device)

        # Initialize Optimizer
        learning_rate = args.lr
        decay = args.decay
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=decay)
        # Define loss function (CrossEntropyLoss for Classification Problems with
        # probability distributions)
        criterion = torch.nn.CrossEntropyLoss()

        def train():
              model.train()
              optimizer.zero_grad()
              # Use all data as input, because all nodes have node features
              out = model(data)
              # Only use nodes with labels available for loss calculation --> mask
              loss = criterion(out[data.train_mask], data.y[data.train_mask])
              loss.backward()
              optimizer.step()
              return loss

        def test():
              model.eval()
              out = model(data)
              # Use the class with highest probability.
              pred = out.argmax(dim=1)
              # Check against ground-truth labels.
              test_correct = pred[data.test_mask] == data.y[data.test_mask]
              # Derive ratio of correct predictions.
              test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
              return test_acc

        losses = []

        for epoch in range(0, args.epochs+1):
            loss = train()
            losses.append(loss)
            if epoch % 100 == 0:
              print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

        test_acc = test()
        print(f'***** Evaluating the test dataset ***** ')
        print(f'Test Accuracy: {test_acc:.4f}')
        results.append(test_acc)

    print(np.array(results).mean(), np.array(results).std())


if __name__ == '__main__':
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('--hc', type=int, default=64, help="number of hidden channels")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--decay', type=float, default=5e-4, help="decay rate")
    parser.add_argument('--epochs', type=int, default=1000, help="epochs")
    args = parser.parse_args()

    # storing the graph in the data variable
    data_before = load_data('data', 'Cora')

    # data_after = sdrf_w_cuda(data_before, 200)
    # data_after = sdrf_w_cuda(data_before, loops=50, remove_edges=False)
    data_after = sdrf(data_before, 50, is_undirected=True)
    data_after.y = data_before.y
    data_after.train_mask = data_before.train_mask
    data_after.val_mask = data_before.val_mask
    data_after.test_mask = data_before.test_mask

    # some statistics about the graph.
    # print(data_before)
    # print(to_networkx(data_after, to_undirected=True))
    print(f'Number of nodes: {data_after.num_nodes}')
    print(f'Number of edges: {data_after.num_edges}')
    print(data_after.train_mask, data_after.val_mask, len(data_after.train_mask), len(data_after.val_mask))
    print(f'Number of training nodes: {data_after.train_mask.sum()}')
    print(f'Training node label rate: {int(data_after.train_mask.sum()) / data_after.num_nodes:.2f}')
    print(f'Is undirected: {data_after.is_undirected()}')
    eval_performance(data_after, 5)
    # without linear layer, 1000 epochs: 80.955+-0.46
    # without linear layer, 1500 epochs: 80.925+-0.37
    # with linear layer, 1000 epochs: 81.2+-0.44
    # Citeseer: 71% with 200 rewiring
    # 85.4 +- 0.35
    #86 for 50 more training
    # 86.5 no sdrf more training
    # 81.18