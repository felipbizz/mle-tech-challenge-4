# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import pickle
from pathlib import Path

import torch
import yaml
from common.utils import get_path_project
from rede_neural.custom_neural_networks import (
    StockPredictionLoss,
    StockPredictionModel,
    TreinadorDeModelos,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# =============================================================================
# CONSTANTES
# =============================================================================

# Nome e versão do modelo
MODEL_NAME = "StockPredictionModel"

# Garantindo os paths corretos
DIR_PROJECT = get_path_project()
assert isinstance(DIR_PROJECT, Path)

RAW = DIR_PROJECT / "data/raw"
STAGED = DIR_PROJECT / "data/staged"

dataset_path = STAGED / "model_training/custom_dataset.pth"
scaler_path = STAGED / "model_training/data_scaler.pkl"

# Configurações necessárias
CONFIG_PATH = DIR_PROJECT / "config/custom_lstm_config.yaml"

with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as yaml_file:
    config = yaml.safe_load(yaml_file)

config_modelos = config["models"]
config_treino = config["training"]
config_dataloader = config["dataloader"]

# Criando os datasets de treino e de teste
dataset = torch.load(f=dataset_path, weights_only=False)

use_scaler = False
if scaler_path.exists() and use_scaler:
    with open(file=scaler_path, mode="rb") as pkl_f:
        scaler = pickle.load(file=pkl_f)
else:
    scaler = None

torch.manual_seed(42)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining 20% for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_dataset, **config_dataloader)
test_dataloader = DataLoader(dataset=test_dataset, **config_dataloader)

# =============================================================================
# MAIN
# =============================================================================


def main(model_name: str) -> None:

    ash = TreinadorDeModelos(
        modelo=StockPredictionModel,
        config_modelo=config_modelos[model_name],
        otimizador=Adam,
        fn_perda=StockPredictionLoss,
        lr=config_treino["learning_rate"],
        num_epochs=config_treino["num_epochs"],
    )

    ash.treina(dataloader=train_dataloader, scaler=scaler)
    ash.testa(dataloader=test_dataloader)

    return None


if __name__ == "__main__":
    main(model_name=MODEL_NAME)
