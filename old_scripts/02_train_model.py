from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM
from deltalake import DeltaTable
from config.config import settings

best_config = settings.best_config

models = [LSTM(scaler_type="robust", **best_config)]

df = DeltaTable("deltalake").to_pandas()
df = df.sort_values(by=["unique_id", "ds"]).reset_index(drop=True)

FILTRA_IDS = df["unique_id"].unique()[
    :2
]  # podemos analisar tirar esse filtro, eu coloquei por conta da memória da minha GPU

df = df[df["unique_id"].isin(FILTRA_IDS)]

train = df.loc[df["ds"] < "2024-09-01"]

model = NeuralForecast(models=models, freq="D")

if __name__ == "__main__":  
    
    model.fit(train)
