import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from scipy.interpolate import interp1d
import pickle
import os
import numpy as np
import pandas as pd

class DowkerDataset(Dataset):
    def __init__(self, data_config):
        super(DowkerDataset, self).__init__()
        self.data_config = data_config
        self.data = []
        self.load_data()

    def load_data(self):
        if self.data_config.name in ['citation', 'bitcoin', 'question', 'social']:
            data_path = os.path.join('./data/dynamic_dataset', f'{self.data_config.name}_{self.data_config.dataset_size}.pkl')
        elif self.data_config.name in ['RedditB', 'Reddit5K', 'Reddit12K']:
            data_path = os.path.join('./data/static_dataset', f'{self.data_config.name}_{self.data_config.dataset_size}.pkl')
        else:
            raise ValueError("Invalid dataset name")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
