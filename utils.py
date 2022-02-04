# from https://github.com/GeoX-Lab/CGNN

import os

import numpy as np
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset


class TranductiveDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TranductiveDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        data = self.get_data()

        if self.pre_filter is not None:
            # data_list = [data for data in data_list if self.pre_filter(data)]
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            # data_list = [self.pre_transform(data) for data in data_list]
            data = self.pre_transform(data)

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def get_data(self):
        pass

    def read_file(self, path, name):
        pass


def sample_per_class(random_state, labels, num_examples_per_class):
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


def load_data(data_path, d_name):
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
        # CurvGN中的划分方法
        # index = [i for i in range(len(data.y))]
        # train_len = 20 * int(data.y.max() + 1)
        # train_mask = torch.tensor([i < train_len for i in index])
        # val_mask = torch.tensor([i >= train_len and i < 500 + train_len for i in index])
        # test_mask = torch.tensor([i >= len(data.y) - 1000 for i in index])
        # data.train_mask, data.val_mask, data.test_mask = train_mask, val_mask, test_mask
        labels = data.y.numpy()
        num_nodes = data.num_nodes
        seed = 2018
        random_state = np.random.RandomState(seed)
        remaining_indices = list(range(num_nodes))
        train_indices = sample_per_class(random_state, labels, 20)
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, 500, replace=False)
        remaining_indices = np.setdiff1d(remaining_indices, val_indices)
        test_indices = random_state.choice(remaining_indices, 1000, replace=False)
        train_mask, val_mask, test_mask = split_to_binary(num_nodes, train_indices), \
                                          split_to_binary(num_nodes, val_indices), \
                                          split_to_binary(num_nodes, test_indices)
        data.train_mask, data.val_mask, data.test_mask = torch.tensor(train_mask), \
                                                         torch.tensor(val_mask), \
                                                         torch.tensor(test_mask)
    return data


def split_to_binary(node_num, indices):
    data = np.zeros(node_num, dtype=bool)
    data[indices] = True
    return data
