from torch.utils.data import DataLoader

from ml_translate.dataset import TranslateDataset

TEST_DATA_ENG_FRA_FILE = "tests/data/eng-fra-top-10.txt"


class TestDataLoader:
    def test_dataloader(self):
        dataset = TranslateDataset(
            filepath=TEST_DATA_ENG_FRA_FILE
        )
        dataloader = DataLoader(dataset, batch_size=2)
        x, y = next(iter(dataloader))
        assert x == ("Go.", "Run!")
        assert y == ("Va !", "Cours !")
