import os

import torch
from torch_geometric.data import Dataset


class DAVIS2016(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DAVIS2016, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        # Download to `self.raw_dir`.

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
             # Read data from `raw_path`.
             data = Data(...)

             if self.pre_filter is not None and not self.pre_filter(data):
                 continue

            if self.pre_transform is not None:
                 data = self.pre_transform(data)

            torch.save(data, ops.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx))
        return data