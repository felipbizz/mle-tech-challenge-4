from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from deltalake import DeltaTable
from config.config import settings
import joblib

from datetime import datetime

from src.utils import WMAPE

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

models = [LSTM(scaler_type='robust',**best_config)]

train = DeltaTable('deltalake').to_pandas()
train = train.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

model = NeuralForecast(models=models, freq='D')
model.fit(train)

joblib.dump(model, f"neuralforecast_lstm_{datetime.now().date()}.joblib")

end_time = datetime.now()
total_time = end_time - start_time

print(f"Tempo total de execução: {total_time}")