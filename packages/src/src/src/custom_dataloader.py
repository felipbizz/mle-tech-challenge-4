import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SequenceDataset(Dataset):
    def __init__(self, raw_data, input_len: int, target_len: int):
        """
        Args:
            raw_data (numpy.ndarray): Dados tabulares
            input_len (int): Número de "time steps" da sequência de entrada.
            target_len (int): Número de "time steps" da sequência de saída.
        """
        self.raw_data = raw_data
        self.input_len = input_len
        self.target_len = target_len
        self.data_length = len(raw_data) - (input_len + target_len)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        x = self.raw_data[idx : idx + self.input_len]
        y = self.raw_data[idx + self.input_len : idx + self.input_len + self.target_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


def main() -> None:

    # Example raw data (e.g., from a pandas DataFrame)
    raw_data = pd.DataFrame(
        {
            "feature1": np.arange(100),
            "feature2": np.arange(100, 200),
        }
    ).values

    input_len = 5
    target_len = 2

    # Create the Dataset
    dataset = SequenceDataset(raw_data, input_len, target_len)

    # Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Iterate through the DataLoader
    for x, y in dataloader:
        print("Input (x):", x)
        print("Target (y):", y)
        break

    return None


# Example Usage
if __name__ == "__main__":
    main()
