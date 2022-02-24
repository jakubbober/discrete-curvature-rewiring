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




# array([294, 1013, 1074, 1675, 1675, 1675, 1675, 1675, 1676, 1914, 2045, 2047, 2047, 2047]),
# array([2047, 1675, 1675, 1013, 1074, 1676, 1914, 2047, 1675, 1675, 2047, 294, 1675, 2045]))
# 1675, 2047

# array([ 103,  109,  112,  124,  126,  138,  139,  153,  236,  294,  302,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  308,  329,  350,  408,  417,  426,  452,  476,  487,
#         519,  542,  554,  573,  581,  608,  655,  656,  719,  734,  736,
#         887,  910,  958,  973, 1009, 1013, 1045, 1072, 1158, 1193, 1245,
#        1251, 1346, 1367, 1483, 1490, 1551, 1572, 1584, 1640, 1651, 1656,
#        1705, 1770, 1771, 1772, 1775, 1779, 1781, 1782, 1787, 1797, 1798,
#        1799, 1802, 1804, 1805, 1841, 1841, 1841, 1841, 1841, 1841, 1856,
#        2045, 2046, 2048, 2056, 2078, 2080, 2084, 2085, 2086, 2087, 2088,
#        2089, 2090, 2091]),
# array([ 306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         103,  109,  112,  124,  126,  138,  139,  153,  236,  294,  302,
#         308,  329,  350,  417,  426,  452,  476,  487,  519,  542,  554,
#         573,  581,  608,  655,  656,  719,  887,  910,  958,  973, 1009,
#        1045, 1072, 1158, 1193, 1245, 1251, 1346, 1367, 1483, 1490, 1551,
#        1572, 1584, 1640, 1651, 1656, 1705, 1770, 1771, 1772, 1775, 1779,
#        1781, 1782, 1787, 1797, 1798, 1799, 1802, 1804, 1805, 1841, 1856,
#        2045, 2046, 2048, 2078, 2080, 2084, 2085, 2086, 2087, 2088, 2089,
#        2090, 2091,  306,  306,  306, 1841,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306, 1841, 1841,
#         306,  306,  306,  306,  306, 1841,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306,  306,  306,  408,  734,  736, 1013, 2056,  306,
#         306,  306,  306, 1841,  306,  306,  306,  306,  306,  306,  306,
#         306,  306,  306]))