import pickle
from pathlib import Path

import deltalake
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler
from src.custom_dataloader import SequenceDataset
from src.utils import get_path_project

# Diretório do projeto
DIR_PROJECT = get_path_project()
assert isinstance(DIR_PROJECT, Path)

RAW = DIR_PROJECT / "data/raw"
STAGED = DIR_PROJECT / "data/staged"

# Configurações necessárias
CONFIG_PATH = DIR_PROJECT / "config/custom_lstm_config.yaml"

with open(CONFIG_PATH, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

config_modelo = config["model"]
config_dataset = config["dataset"]

# Paths
datalake_path = RAW / "yfinance_api"
dataset_to_path = STAGED / "model_training/custom_dataset.pth"
scaler_to_path = STAGED / "model_training/data_scaler.pkl"

scaler_to_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = deltalake.DeltaTable(table_uri=datalake_path).to_pandas()

    df_v1 = df.copy()
    date_filter = (
        df_v1.groupby("ds")["unique_id"].count() == config_modelo["num_stocks"]
    )
    dates_to_consider = []
    for date, to_consider in date_filter.items():
        if to_consider:
            dates_to_consider.append(date)
    df_v1 = df_v1[df_v1["ds"].isin(dates_to_consider)].reset_index(drop=True)

    df_v2 = pd.pivot_table(
        data=df_v1, values="y", columns="unique_id", index="ds"
    ).sort_index()

    dataset = SequenceDataset(
        raw_data=df_v2.values, data_scaler=MinMaxScaler, **config_dataset
    )

    torch.save(
        obj=dataset,
        f=dataset_to_path,
    )

    scaler = dataset.data_scaler

    if scaler:
        with open(scaler_to_path, "wb") as pkl_f:
            pickle.dump(obj=scaler, file=pkl_f)

    return None


if __name__ == "__main__":
    main()
