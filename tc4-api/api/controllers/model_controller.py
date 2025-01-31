from fastapi import APIRouter, Body
from typing import Annotated, Any
from src.utils import setLog
from business_rules import model
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
def predict(model_file: Annotated[str | None, Body()], stock_option : str = 'VALE3.SA'):
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
    
    return { 'message' : model.make_predictions(model_file, stock_option) } 
