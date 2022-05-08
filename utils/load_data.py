import os

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T

from utils.TranductiveDataset import TranductiveDataset


def load_data(data_path: str, d_name: str) -> torch_geometric.data.data.Data:
    """
    Load torch_geometric data.
    :param data_path: string denoting the path to all the torch_geometric data.
    :param d_name: name of the specific dataset to be loaded.
    :return: given torch_geometric dataset.
    """
    if d_name == 'Cora' or d_name == 'Citeseer' or d_name == 'PubMed':
        d_loader = 'Planetoid'
    elif d_name == 'Computers' or d_name == 'Photo':
        d_loader = 'Amazon'
    elif d_name == 'CS' or d_name == 'Physics':
        d_loader = 'Coauthor'
    else:
        d_loader = 'TranductiveDataset'
    if d_loader == 'Planetoid':
        data = getattr(torch_geometric.datasets, d_loader)(data_path, d_name, transform=T.NormalizeFeatures())[0]
    elif d_loader == 'TranductiveDataset':
        data = TranductiveDataset(os.path.join(data_path, d_name))[0]
    else:
        data = getattr(torch_geometric.datasets, d_loader)(os.path.join(data_path, d_name), d_name)[0]
        labels = data.y.numpy()
        num_nodes = data.num_nodes
        # seed = 2018
        random_state = np.random.RandomState()
        remaining_indices = list(range(num_nodes))
        train_indices = _sample_per_class(random_state, labels, 20)
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, 500, replace=False)
        remaining_indices = np.setdiff1d(remaining_indices, val_indices)
        test_indices = random_state.choice(remaining_indices, 1000, replace=False)
        train_mask, val_mask, test_mask = _split_to_binary(num_nodes, train_indices), \
                                          _split_to_binary(num_nodes, val_indices), \
                                          _split_to_binary(num_nodes, test_indices)
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), \
                                                         torch.tensor(val_mask), \
                                                         torch.tensor(test_mask)
    return data


def _sample_per_class(random_state, labels, num_examples_per_class):
    """Helper function for load_data()."""
    num_classes = max(labels) + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for i, label in enumerate(labels):
            if class_index == label:
                sample_indices_per_class[class_index].append(i)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def _split_to_binary(node_num, indices):
    """Helper function for load_data()."""
    data = np.zeros(node_num, dtype=bool)
    data[indices] = True
    return data
