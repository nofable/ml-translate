import tempfile

from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from ml_translate.collate import collate
from ml_translate.dataset import TabSeparatedLineDelimTranslationPairsDataset
from ml_translate.tokenizer import BytePairTokenizer

TEST_DATA_ENG_FRA_FILE = "tests/data/eng-fra-top-10.txt"


class TestCollate:
    def test_collate(self):
        bytePairTokenizer = BytePairTokenizer()
        with tempfile.NamedTemporaryFile() as tmp_file:
            bytePairTokenizer.train(
                src_file=TEST_DATA_ENG_FRA_FILE, out_file=tmp_file.name
            )

            tokenizer = Tokenizer.from_file(tmp_file.name)
            dataset = TabSeparatedLineDelimTranslationPairsDataset(
                filepath=TEST_DATA_ENG_FRA_FILE
            )
            dataloader = DataLoader(dataset, batch_size=2)
            x, _ = next(iter(dataloader))
            output, mask, causal = collate(x, tokenizer)
            assert output.shape == (2, 6)
            assert mask.shape == (2, 6)
            assert causal.shape == (2, 6)
