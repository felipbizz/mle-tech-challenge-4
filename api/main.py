# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

from pathlib import Path
from typing import List

import psutil
import torch
import uvicorn
import yaml
from common.utils import (
    get_path_project,
    load_model_and_scaler_by_run,
    load_model_and_scaler,
)
from fastapi import FastAPI
from prometheus_client import Gauge, make_asgi_app
from src import schemas

# =============================================================================
# CONSTANTES
# =============================================================================

# Garantindo os paths corretos
DIR_PROJECT = get_path_project(project_name="app")
assert isinstance(DIR_PROJECT, Path)

# Configurações necessárias
CONFIG_PATH = DIR_PROJECT / "config/custom_lstm_config.yaml"

with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as yaml_file:
    config = yaml.safe_load(yaml_file)

config_api = config["api"]

# =============================================================================
# Modelo - Rede Neural
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Modelo - Rede Neural
# =============================================================================

# modelo, scaler = load_model_and_scaler_by_run(**config_api["run"])
modelo, scaler = load_model_and_scaler(version=None, **config_api["model"])

modelo = modelo.to(device=device)
modelo.eval()

# =============================================================================
# API
# =============================================================================


def list_to_tensor(input: List[List[float]]) -> torch.Tensor:
    tensor = torch.Tensor(input).to(device=device)
    return tensor


def predict_request(request: schemas.TimeSeries) -> List[List[float]]:

    input = request.series

    if scaler:
        input = scaler.transform(input)

    tensor = list_to_tensor(input)
    output = modelo(tensor.unsqueeze(dim=0))
    output = output.squeeze(0).cpu().detach().numpy()

    if scaler:
        output = scaler.inverse_transform(output)

    return output.tolist()


# =============================================================================
# API
# =============================================================================


# -----------------------------------------------------------------------------
# Prometheus
# -----------------------------------------------------------------------------

# Define Prometheus metrics
cpu_usage_gauge = Gauge("cpu_usage", "CPU usage percentage")
gpu_usage_gauge = Gauge("gpu_usage", "GPU memory usage percentage")
predicted_value_gauge = Gauge("predicted_value", "Predicted value from model")


# Function to update system metrics
def update_metrics():
    cpu_usage_gauge.set(psutil.cpu_percent())

    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
        gpu_usage_gauge.set((gpu_memory_allocated / gpu_memory_total) * 100)
    else:
        gpu_usage_gauge.set(0)


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

app = FastAPI()

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/", response_model=schemas.Message)
async def root() -> schemas.Message:
    return schemas.Message(message="Backend da aplicação está funcionando!")


@app.post("/predict", response_model=schemas.Prediction)
async def predict(historico_uma_semana: schemas.TimeSeries) -> schemas.Prediction:
    prediction = predict_request(historico_uma_semana)
    return schemas.Prediction(
        message="Backend da aplicação está funcionando!", prediction=prediction
    )


metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


def main() -> None:
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
    return None


# if __name__ == "__main__":
#     main()
