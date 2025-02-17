# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from typing import Optional, Type

import torch
from sklearn.base import TransformerMixin
from torch.utils.data import Dataset

# =============================================================================
# CLASSES
# =============================================================================


class SequenceDataset(Dataset):
    def __init__(
        self,
        raw_data,
        input_len: int,
        target_len: int,
        data_scaler: Optional[Type[TransformerMixin]] = None,
        num_stocks: Optional[int] = None,
    ):
        """
        Args:
            raw_data (numpy.ndarray): Dados tabulares
            input_len (int): Número de "time steps" da sequência de entrada.
            target_len (int): Número de "time steps" da sequência de saída.
        """
        self.raw_data = raw_data
        self.data_scaler = data_scaler() if data_scaler is not None else None
        self.data = (
            raw_data
            if self.data_scaler is None
            else self.data_scaler.fit_transform(raw_data)
        )
        self.input_len = input_len
        self.target_len = target_len
        self.data_length = len(raw_data) - (input_len + target_len)

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]
        y = self.data[idx + self.input_len : idx + self.input_len + self.target_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )
