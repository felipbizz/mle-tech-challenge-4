from src.utils import setLog
from neuralforecast import NeuralForecast
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta

logger = setLog('model')

def make_predictions(model_file, forecast_period) -> dict:

    logger.info(f'Carregando o modelo {model_file}')
    try:
        model = load_model(model_file)
        logger.info(f'Modelo {model_file} carregado. Tipo do modelo {type(model)}')

        predict_df = create_predict_dataframe(forecast_period)
        logger.info(f'Dataframe de previsão criado.\n{predict_df}')

        predict_result = model.predict(predict_df)
        logger.info(f'Resultado da previsão: {predict_result}')

        return { 'message': f'Previsões:\n{predict_result}' }
    except Exception as e:
        logger.error(e)
        data_returned = e.args[0]
        logger.error(f'Dado retornado: {data_returned}')
        raise
        # return { 'message' : e.args, 'data_returned' : data_returned }

def load_model(model_file : str) -> NeuralForecast :
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
    
def create_predict_dataframe(forecast_period: int):
    start_date = datetime.today()
    date_list = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(forecast_period)]

    return pd.DataFrame({'ds': date_list} )