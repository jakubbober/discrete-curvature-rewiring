import pickle
import torch
import argparse
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv #GATConv
from torch_geometric.transforms import NormalizeFeatures

from gcn import GCN


def test(model):
    model.eval()
    out = model(data)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


def train(model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    # Use all data as input, because all nodes have node features
    out = model(data)
    # Only use nodes with labels available for loss calculation --> mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--hc', type=int, default = 16, help = "number of hidden channels")
    parser.add_argument('--lr', type=float, default = 0.01, help = "learning rate")
    parser.add_argument('--decay', type=float, default = 5e-4, help = "decay rate")
    parser.add_argument('--epochs', type=int, default = 200, help = "epochs")
    args = parser.parse_args()

    # load pickle dataset
    # file = open('data/Planetoid/Cora/processed/data.pt', 'rb')
    # dataset = pickle.load(file)
    # file.close()

    with open('rewiring_edges', 'rb') as file:
        edges = pickle.load(file)

    dataset = Planetoid(root='data/Planetoid', name='Cora')

    # storing the graph in the data variable
    data = dataset[0]

    # data.edge_index = edges

    # Initialize model
    model = GCN(dataset.num_node_features, dataset.num_classes)

    # Use CPU
    device = torch.device("cpu")
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
    # criterion = F.nll_loss
    criterion = torch.nn.CrossEntropyLoss()
    losses = []

    for epoch in range(0, args.epochs + 1):
        loss = train(model, optimizer, criterion)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test(model)
    print(f'***** Evaluating the test dataset ***** ')
    print(f'Test Accuracy: {test_acc:.4f}')
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    #
    #
    # class GCNN(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.conv1 = GCNConv(dataset.num_node_features, 16)
    #         self.conv2 = GCNConv(16, dataset.num_classes)
    #
    #     def forward(self, data):
    #         x, edge_index = data.x, data.edge_index
    #
    #         x = self.conv1(x, edge_index)
    #         x = F.relu(x)
    #         x = F.dropout(x, training=self.training)
    #         x = self.conv2(x, edge_index)
    #
    #         return F.log_softmax(x, dim=1)
    #
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCNN().to(device)
    # data = dataset[0].to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    #
    # model.train()
    # for epoch in range(200):
    #     optimizer.zero_grad()
    #     out = model(data)
    #     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #     loss.backward()
    #     optimizer.step()
    #
    # model.eval()
    # pred = model(data).argmax(dim=1)
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = int(correct) / int(data.test_mask.sum())
    # print(f'Accuracy: {acc:.4f}')
