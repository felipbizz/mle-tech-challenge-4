# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import pickle
from pathlib import Path

import deltalake
import pandas as pd
import torch
import yaml
from common.utils import get_path_project
from rede_neural.custom_dataloader import SequenceDataset

# =============================================================================
# CONSTANTES
# =============================================================================

# Diretório do projeto
DIR_PROJECT = get_path_project()
assert isinstance(DIR_PROJECT, Path)

RAW = DIR_PROJECT / "data/raw"
STAGED = DIR_PROJECT / "data/staged"

# Configurações necessárias
CONFIG_PATH = DIR_PROJECT / "config/custom_lstm_config.yaml"

with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as yaml_file:
    config = yaml.safe_load(yaml_file)

config_dataset = config["dataset"]

# Paths
datalake_path = RAW / "yfinance_api"
dataset_to_path = STAGED / "model_training/custom_dataset.pth"
scaler_to_path = STAGED / "model_training/data_scaler.pkl"

scaler_to_path.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    # Carrega os dados da tabela Delta como um DataFrame do Pandas.
    df = deltalake.DeltaTable(table_uri=datalake_path).to_pandas()

    # Cria uma cópia do DataFrame original para manipulação.
    df_v1 = df.copy()

    # Aplica um filtro para identificar as datas que têm o número esperado de ações ('num_stocks') nos dados.
    date_filter = (
        df_v1.groupby("ds")["unique_id"].count() == config_dataset["num_stocks"]
    )

    # Inicializa uma lista para armazenar as datas a serem consideradas.
    dates_to_consider = []
    for date, to_consider in date_filter.items():
        if to_consider:
            dates_to_consider.append(
                date
            )  # Adiciona a data à lista se atender ao critério.

    # Filtra o DataFrame para incluir apenas as datas consideradas.
    df_v1 = df_v1[df_v1["ds"].isin(dates_to_consider)].reset_index(drop=True)

    # Constrói uma tabela dinâmica onde as datas são as linhas, 'unique_id' são as colunas,
    # e os valores correspondem à coluna 'y', e ordena pelo índice (datas).
    df_v2 = pd.pivot_table(
        data=df_v1, values="y", columns="unique_id", index="ds"
    ).sort_index()

    # Cria um conjunto de dados de sequência a partir dos dados processados.
    dataset = SequenceDataset(raw_data=df_v2.values, data_scaler=None, **config_dataset)

    # Salva o conjunto de dados em um arquivo especificado.
    torch.save(
        obj=dataset,
        f=dataset_to_path,
    )

    # Obtém o scaler do conjunto de dados (se houver).
    scaler = dataset.data_scaler

    # Se um scaler foi gerado, salva-o em um arquivo utilizando pickle.
    if scaler:
        with open(scaler_to_path, "wb") as pkl_f:
            pickle.dump(obj=scaler, file=pkl_f)

    return None


if __name__ == "__main__":
    main()
