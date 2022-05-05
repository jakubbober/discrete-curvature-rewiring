import torch
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
