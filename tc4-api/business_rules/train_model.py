from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from deltalake import DeltaTable
import joblib
from datetime import datetime, timedelta
from src.utils import setLog, WMAPE

logger = setLog('train_model', level=10)


def train_model(best_config) -> str:

    start_time : datetime = datetime.now()

    logger.info(f'Iniciando treinamento utilizandos os seguintes hiperparâmetros: {best_config}')

    models : list = [LSTM(loss=WMAPE(),**best_config)]
    logger.info('Modelo criado.')

    train = DeltaTable('deltalake').to_pandas()

    train = train.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    logger.info(f'Dados do deltalake carregados. Tamanho do dataset: {len(train)}')

    train.drop_duplicates(subset=['ds', 'unique_id'], inplace=True)
    logger.info(f'Dados após a remoção de duplicatas. Tamanho do dataset: {len(train)}')

    last_year = str((datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'))

    train = train[train['ds'] > last_year]
    logger.info(f'Restringindo dados de treinamento a um ano. Tamanho do dataset: {len(train)}')

    model = NeuralForecast(models=models, freq='D')
    model.fit(train)
    logger.info('Modelo treinado.')

    model_path = f'ml_models/neuralforecast_lstm_{datetime.now().strftime("%Y%m%d_%H%M")}.joblib'

    joblib.dump(model, model_path)
    logger.info(f'Modelo salvo em: {model_path}')

    end_time : datetime = datetime.now()
    total_time : datetime = end_time - start_time

    logger.info(f"Tempo total de execução: {total_time}")

    return model_path