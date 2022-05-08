import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, d_input, d_output, d_hidden=16):
        super(GCN, self).__init__()

        # Initialize the layers
        self.conv1 = GCNConv(d_input, d_hidden)
        self.conv2 = GCNConv(d_hidden, d_output)
        # self.out = Linear(d_hidden, d_output)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)

        # Output layer
        x = F.softmax(x, dim=1)
        return x
