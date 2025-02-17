# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import joblib
import mlflow
import torch
from dotenv import find_dotenv, load_dotenv
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from sklearn.base import TransformerMixin
from torch import nn

# =============================================================================
# CONSTANTES
# =============================================================================

# Carregando as variáveis de ambiente
load_dotenv(find_dotenv())

# Obtendo o nome do projeto
PROJECT_NAME = os.getenv("PROJECT_NAME", "mle-tech-challenge-4-alecrim")

# Instanciando o cliente do MLFlow
mlflow_client = MlflowClient()

# Determinando qual dispositivo será utilizado
device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# FUNÇÕES
# =============================================================================

# -----------------------------------------------------------------------------
# Obtendo o path raiz do projeto
# -----------------------------------------------------------------------------


def get_path_project(
    project_name: str = PROJECT_NAME, cwd: Path = Path.cwd()
) -> Union[Callable, Path]:
    """
    Obtém o caminho do diretório do projeto a partir do diretório de trabalho atual.

    Args:
        project_name: Nome do projeto que estamos tentando localizar.
                      O valor padrão é definido por PROJECT_NAME.
        cwd: Diretório de trabalho atual a partir do qual a busca é iniciada.
             O valor padrão é o diretório atual (Path.cwd()).

    Returns:
        Path: O caminho para o diretório do projeto se encontrado,
              caso contrário, recursivamente busca no diretório pai.
    """
    # Verifica se o nome do diretório atual é o nome do projeto especificado.
    if cwd.name == project_name:
        return cwd  # Retorna o caminho do diretório do projeto encontrado.

    # Se não for encontrado, chama a função recursivamente no diretório pai.
    return get_path_project(project_name=project_name, cwd=cwd.parent)


# -----------------------------------------------------------------------------
# Carregando o modelo e o scaler
# -----------------------------------------------------------------------------


def load_model_and_scaler(
    model_name: str, version: Optional[str]
) -> Tuple[nn.Module, Optional[TransformerMixin]]:
    """
    Carrega um modelo PyTorch e um scaler opcional a partir do MLflow.

    Args:
        model_name: Nome do modelo a ser carregado.
        version: Versão do modelo a ser carregada; se None, a versão mais recente será utilizada.

    Returns:
        Tuple[nn.Module, Optional[TransformerMixin]]: O modelo carregado e o scaler opcional,
        ou None se não houver scaler.
    """
    # Obtém a versão mais recente do modelo registrado no MLflow.
    latest_mv = mlflow_client.get_latest_versions(model_name, stages=None)[0]

    # Define o URI do modelo a ser carregado, usando a versão especificada ou a mais recente.
    model_uri = (
        f"models:/{model_name}/{version}"
        if version
        else f"models:/{model_name}/{latest_mv.version}"
    )

    # Carrega o modelo PyTorch do MLflow, especificando o dispositivo de mapeamento.
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(device))

    # Caminho para o modelo baixado.
    model_path = download_artifacts(model_uri)
    # Define o caminho do scaler que deve ter sido salvo com o modelo.
    scaler_path = Path(model_path) / f"{model_name}_scaler.pkl"

    # Verifica se o arquivo do scaler existe.
    if scaler_path.exists():
        # Se existir, carrega o scaler usando joblib.
        scaler = joblib.load(download_artifacts(str(scaler_path)))
    else:
        # Caso contrário, informa que não foi usado um scaler e define como None.
        print("Não foi usado um scaler!")
        scaler = None

    # Retorna o modelo e o scaler (ou None).
    return model, scaler


def load_model_and_scaler_by_run(
    run_id: str, model_name: str
) -> Tuple[nn.Module, Optional[TransformerMixin]]:
    """
    Carrega o modelo e o scaler a partir de um ID de execução específica no MLflow.

    Args:
        run_id: Identificador da execução no MLflow.
        model_name: Nome do modelo a ser carregado.
        version: Versão do modelo a ser carregada.

    Returns:
        Uma tupla contendo o modelo carregado (como um objeto nn.Module) e o scaler (se disponível, ou None se não existir).
    """
    # Cria a URI do modelo utilizando o run_id, model_name e version fornecidos
    model_uri = f"runs:/{run_id}/{model_name}"
    # Carrega o modelo PyTorch usando a URI construída
    model = mlflow.pytorch.load_model(model_uri, map_location=torch.device(device))

    # Cria a URI do scaler, que é um arquivo .pkl associado à mesma execução
    scaler_uri = f"runs:/{run_id}/{model_name}/{model_name}_scaler.pkl"

    try:
        # Tenta carregar o scaler utilizando a biblioteca joblib
        scaler = joblib.load(download_artifacts(scaler_uri))
    except OSError:
        # Se ocorrer um erro ao carregar, informa que não há scaler e define como None
        print("Não foi usado um scaler!")
        scaler = None

    # Retorna o modelo e o scaler
    return model, scaler
