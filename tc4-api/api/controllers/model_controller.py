from fastapi import APIRouter, Body
from typing import Annotated, Any
from src.utils import setLog
from business_rules import model
from business_rules.tune_model import tuna_modelo_autolstm
from business_rules.train_model import train_model
import os

from prometheus_client import Summary, Counter

logger = setLog("model_controller", level=10)

router = APIRouter(prefix="/api/v1/model", tags=["Endpoints do Modelo"])

# Define Prometheus metrics
TUNE_REQUEST_TIME = Summary(
    "tune_request_processing_seconds", "Tempo gasto processando requisições de ajuste"
)
TRAIN_REQUEST_TIME = Summary(
    "train_request_processing_seconds",
    "Tempo gasto processando requisições de treinamento",
)
PREDICT_REQUEST_TIME = Summary(
    "predict_request_processing_seconds",
    "Tempo gasto processando requisições de previsão",
)
PREDICT_COUNT = Counter("request_count", "Número total de previsões")


@router.get("/list")
def list_models():
    """
    Lista os modelos treinados disponíveis para serem utilizados em previsões.

    Parameters:

        Nenhum parâmetro necessário.

    Returns:

        list : Lista com o nome dos arquivos de modelos disponíveis.
    """

    logger.info(
        "---------------------------------------------------------------------------------------------------"
    )
    logger.info("Listando modelos disponíveis para predição.")

    path = "ml_models"

    logger.info(f"Obtendo lista de arquivos em {path}")
    model_list = os.listdir(path)

    logger.debug(f"Retornando a lista de arquivos existentes: {model_list}")

    return model_list


@router.get("/tune")
@TUNE_REQUEST_TIME.time()
def tune():
    """
    Ajusta o modelo e retorna os melhores hiperparâmetros encontrados.

    Parameters:

        Nenhum parâmetro de entrada.

    Returns:

        dict : Dicionário com os melhores hiperparâmetros encontrados pelo tuning.
    """
    best_hp = tuna_modelo_autolstm()

    return best_hp


@router.post("/train")
@TRAIN_REQUEST_TIME.time()
def train(best_config: Annotated[dict | None, Body()]) -> str:
    """
    Treina o modelo utilizando os hiperparâmetros informados.

    Parameters:

        dict : Dicionário contendo os hiperparâmetros a serem utilizados no treinamento.

    Returns:

        str : O caminho onde foi salvo o modelo treinado.
    """
    return train_model(best_config)


@router.post("/predict")
@PREDICT_REQUEST_TIME.time()
def predict(model_file: Annotated[str | None, Body()], stock_option: str = "VALE3.SA"):
    """
    Lista os modelos treinados disponíveis para serem utilizados em previsões.

    Parameters:

        model_file (str) : Nome do modelo a ser usado na previsão.

    Returns:

        str: String contendo o caminho onde o gráfico da previsão foi salvo.

    Exceptions:

        FileNotFound : Disparado caso o arquivo do modelo não seja encontrado (a ser implementado).
    """
    logger.info(
        "---------------------------------------------------------------------------------------------------"
    )
    logger.info(f"Iniciando previsão utilizando o modelo {model_file}")
    PREDICT_COUNT.inc()
    return {"message": model.make_predictions(model_file, stock_option)}
