import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, d_input, d_output, d_hidden=16):
        super(GCN, self).__init__()

        # Initialize the layers
        self.conv1 = GCNConv(d_input, d_hidden)
        self.conv2 = GCNConv(d_hidden, d_hidden)
        self.out = Linear(d_hidden, d_output)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        x = F.softmax(self.out(x), dim=1)
        return x
