from src.utils import setLog
from neuralforecast import NeuralForecast
import joblib
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

logger = setLog('model')

def make_predictions(model_file : str, stock_option : str) -> dict:

    logger.info(f'Carregando o modelo {model_file}')
    try:
        model : NeuralForecast = load_model(model_file)
        logger.info(f'Modelo {model_file} carregado. Tipo do modelo {type(model)}')

        logger.info('Realizando previsões utilizando o horizonte definido no treinamento do modelo.')
        predict_result : pd.DataFrame = model.predict()

        plot_df : pd.DataFrame = predict_result[predict_result.unique_id==stock_option].drop('unique_id', axis=1)
        
        logger.info(f'Resultado da previsão: {predict_result}')

        logger.info(f'Gerando plot para as previsões de {stock_option}')
        plt.plot(plot_df['ds'], plot_df['LSTM'], c='purple', label=f'Previsões para {stock_option}')
        plt.legend()
        plt.grid()
        plt.plot()

        plot_name : str = f'reports/neuralforecast_lstm_{stock_option}_{datetime.now().date()}.png'

        logger.info(f'Imagem do plot salva em {plot_name}')
        plt.savefig(plot_name, dpi=300)
        plt.close()

        return { 'message': f'Previsão concluída, consulte a imagem em {plot_name}' }
    
    except Exception as e:
        logger.error(e)
        raise

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
