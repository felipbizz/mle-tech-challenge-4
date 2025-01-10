from pathlib import Path
from typing import Callable, Union
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import torch
from typing import Union

load_dotenv(find_dotenv())

PROJECT_NAME = os.getenv("PROJECT_NAME", "mle-tech-challenge-4")


def wmape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()


class WMAPE(torch.nn.Module):
    def __init__(self):
        super(WMAPE, self).__init__()
        self.outputsize_multiplier = 1
        self.output_names = [""]
        self.is_distribution_output = False

    def domain_map(self, y_hat: torch.Tensor):
        return y_hat.squeeze(-1)

    def __call__(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        if mask is None:
            mask = torch.ones_like(y_hat)

        num = mask * (y - y_hat).abs()
        den = mask * y.abs()
        return num.sum() / den.sum()


def get_path_project(cwd: Path = Path.cwd()) -> Union[Callable, Path]:
    if cwd.name == PROJECT_NAME:
        return cwd
    return get_path_project(cwd.parent)
