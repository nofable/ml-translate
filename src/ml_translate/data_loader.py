import torch
from torch.utils.data import DataLoader, random_split
from ml_translate.dataset import TranslateDataset
from ml_translate.config import ENG_FRA_TEXT_FILE


class TranslateDataLoader:
    def __init__(self):
        self.full_dataset = TranslateDataset(filepath=ENG_FRA_TEXT_FILE)

    def train_test_dataloaders(self, split=[0.8, 0.2]):
        g = torch.Generator().manual_seed(42)
        train_ds, test_ds = random_split(self.full_dataset, split, generator=g)
        train_dataloader = DataLoader(train_ds, batch_size=10, shuffle=True)
        test_dataloader = DataLoader(test_ds, batch_size=10)
        return train_dataloader, test_dataloader
