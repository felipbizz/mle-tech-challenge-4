from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from deltalake import DeltaTable
from config.config import settings
import joblib
import torch

from datetime import datetime

from src.utils import WMAPE, setLog

logger = setLog('train_model', level=10)

start_time = datetime.now()
best_config = {
    "h": 77,
    "encoder_hidden_size": 300,
    "encoder_n_layers": 1,
    "context_size": 10,
    "decoder_hidden_size": 64,
    "learning_rate": 0.008279309926218455,
    "max_steps": 10000,
    "batch_size": 32,
    "loss": WMAPE(),
    "check_val_every_n_epoch": 100,
    "random_seed": 19,
    "input_size": 4928
}

logger.info(f'Iniciando treinamento utilizandos os seguintes hiperparâmetros: {best_config}')

models = [LSTM(scaler_type='robust',**best_config)]
logger.info('Modelo criado.')

train = DeltaTable('deltalake').to_pandas()

train = train.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
logger.info(f'Dados do deltalake carregados. Tamanho do dataset: {len(train)}')

train.drop_duplicates(subset=['ds', 'unique_id'], inplace=True)
logger.info(f'Dados após a remoção de duplicatas. Tamanho do dataset: {len(train)}')

# torch.set_float32_matmul_precision('high')

model = NeuralForecast(models=models, freq='D')
model.fit(train)
logger.info('Modelo treinado.')

model_path = f'ml_models/neuralforecast_lstm_{datetime.now().date()}.joblib'

joblib.dump(model, model_path)
logger.info(f'Modelo salvo em: {model_path}')

end_time = datetime.now()
total_time = end_time - start_time

logger.info(f"Tempo total de execução: {total_time}")