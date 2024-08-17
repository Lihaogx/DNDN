import torch
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
from data.dowker_data import DowkerDataset
from torch_geometric.loader import DataLoader

class DInterface(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.split = data_config.split
        self.batch_size = data_config.batch_size
        self.data_config = data_config
        self.train_dataset = None
        self.test_dataset = None
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            full_dataset = DowkerDataset(self.data_config)
            train_size = int(len(full_dataset) * self.split[0])
            test_size = int(len(full_dataset) * self.split[1])
            self.train_dataset, self.test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)