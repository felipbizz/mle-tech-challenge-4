from fastapi import APIRouter, Body
from typing import Annotated, Any
from src.utils import setLog
from neuralforecast import NeuralForecast
import joblib
import os

logger = setLog('model_controller', level=10)

router = APIRouter(prefix='/api/v1/model', tags=['Endpoints do Modelo'])

@router.get('/list')
def list_models():
    '''    
    Lista os modelos treinados disponíveis para serem utilizados em previsões.

    Parameters:

        Nenhum parâmetro necessário.

    Returns:
    
        list : Lista com o nome dos arquivos de modelos disponíveis.
    '''

    logger.info('---------------------------------------------------------------------------------------------------')
    logger.info('Listando modelos disponíveis para predição.')
    
    path = 'ml_models'
    
    logger.info(f'Obtendo lista de arquivos em {path}')    
    model_list = os.listdir(path)

    logger.debug(f'Retornando a lista de arquivos existentes: {model_list}')
    
    return model_list

@router.post('/predict')
def predict(model_file: Annotated[str | None, Body()]):
    '''    
    Lista os modelos treinados disponíveis para serem utilizados em previsões.

    Parameters:

        model_file (str) : Nome do modelo a ser usado na previsão.

    Returns:
    
        Ainda decidindo o retorno.

    Exceptions:

        FileNotFound : Disparado caso o arquivo do modelo não seja encontrado (a ser implementado).
    '''
    logger.info('---------------------------------------------------------------------------------------------------')
    logger.info(f'Iniciando previsão utilizando o modelo {model_file}')
    
    logger.info(f'Carregando o modelo {model_file}')
    try:
        model = load_model(model_file)
        logger.info(f'Modelo {model.__class__} carregado. Tipo do modelo {type(model)}')
        return { 'message': f'Modelo {model.__class__} carregado.' }
    except Exception as e:
        logger.error(e)
        return { 'message' : e.args}


def load_model(model_file : str) -> Any :
    '''    
    Carrega o modelo treinado a partir do arquivo informado.

    Parameters:

        model_file (str) : Nome do modelo a ser usado na previsão.

    Returns:
    
        model : Modelo carregado usando a biblioteca joblib.
    '''
    logger.info(f'Validando a existência do modelo treinado : {model_file}')
    
    if os.path.exists(f'ml_models/{model_file}'):
        logger.info(f'Modelo {model_file} encontrado na pasta ml_models.')
        return joblib.load(f'ml_models/{model_file}')
    else:
        logger.error(f'Arquivo {model_file} não encontrado na pasta ml_models. Verifique o nome do modelo.')
        raise ValueError(f'Arquivo {model_file} não encontrado! Verifique o nome do modelo e tente novamente!')