import pickle
from pathlib import Path

import torch
import yaml
from src.custom_neural_networks import LSTMModel_v2, TreinadorDeModelos
from src.utils import get_path_project
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Diretório do projeto
DIR_PROJECT = get_path_project()
assert isinstance(DIR_PROJECT, Path)

RAW = DIR_PROJECT / "data/raw"
STAGED = DIR_PROJECT / "data/staged"

dataset_path = STAGED / "model_training/custom_dataset.pth"
scaler_path = STAGED / "model_training/data_scaler.pkl"

# Configurações necessárias
CONFIG_PATH = DIR_PROJECT / "config/custom_lstm_config.yaml"

with open(CONFIG_PATH, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

config_modelo = config["model"]
config_treino = config["training"]
config_dataloader = config["dataloader"]

dataset = torch.load(f=dataset_path, weights_only=False)

if scaler_path.exists():
    with open(scaler_path, "rb") as pkl_f:
        scaler = pickle.load(file=pkl_f)
else:
    scaler = None

torch.manual_seed(42)

train_size = int(0.8 * len(dataset))  # 80% for training
test_size = len(dataset) - train_size  # Remaining 20% for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(dataset=train_dataset, **config_dataloader)
test_dataloader = DataLoader(dataset=test_dataset, **config_dataloader)


# Função main
def main(version: str = "v0.0") -> None:

    ash = TreinadorDeModelos(
        modelo=LSTMModel_v2,
        config_modelo=config_modelo,
        otimizador=Adam,
        fn_perda=nn.MSELoss,
        lr=config_treino["learning_rate"],
        num_epochs=config_treino["num_epochs"],
    )

    ash.treina(dataloader=train_dataloader, scaler=scaler, version=version)
    ash.testa(dataloader=test_dataloader, version=version)

    return None


version = "v0.1"
if __name__ == "__main__":
    main(version=version)
