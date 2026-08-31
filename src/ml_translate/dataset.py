import re

from torch.utils.data import Dataset

cap_lines = 100


class TranslateDataset(Dataset):
    def __init__(self, filepath: str):
        self.data = []
        with open(filepath) as file:
            count = 0
            for line in file:
                if count > cap_lines:
                    break
                else:
                    count += 1

                clean_line = line.strip()
                # Replaces NNBSP, which is used before punctuation in french
                clean_line = re.sub(r"\u202f", " ", clean_line)
                parts = clean_line.split("\t", maxsplit=1)
                assert len(parts) == 2
                self.data.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
