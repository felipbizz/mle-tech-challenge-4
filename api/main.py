# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import logging
import os
from pathlib import Path
from typing import List

import torch
import uvicorn
import yaml
from common.metrics_utils import PrometheusMiddleware, metrics, setting_otlp
from common.utils import get_path_project, load_model_and_scaler
from fastapi import FastAPI
from src import schemas
from src.routers import metrics as metrics_router

# =============================================================================
# VARIÁVEIS DE AMBIENTE
# =============================================================================

APP_NAME = os.environ.get("APP_NAME", "app")
EXPOSE_PORT = os.environ.get("EXPOSE_PORT", 8000)
OTLP_GRPC_ENDPOINT = os.environ.get("OTLP_GRPC_ENDPOINT", "http://tempo:4317")


# Garantindo os paths corretos
DIR_PROJECT = get_path_project(project_name="app")
assert isinstance(DIR_PROJECT, Path)

# Configurações necessárias
CONFIG_PATH = DIR_PROJECT / "config/custom_lstm_config.yaml"

with open(file=CONFIG_PATH, mode="r", encoding="utf-8") as yaml_file:
    config = yaml.safe_load(yaml_file)

config_api = config["api"]

# =============================================================================
# REDE NEURAL
# =============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

modelo, scaler = load_model_and_scaler(version=None, **config_api["model"])

modelo = modelo.to(device=device)
modelo.eval()

# =============================================================================
# FUNÇÕES HELPER
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
# APP
# =============================================================================

app = FastAPI()
app.include_router(router=metrics_router.router)

# Setting metrics middleware
app.add_middleware(PrometheusMiddleware, app_name=APP_NAME)
app.add_route("/metrics", metrics)

# Setting OpenTelemetry exporter
setting_otlp(app, APP_NAME, OTLP_GRPC_ENDPOINT)


# Filter out /endpoint
class EndpointFilter(logging.Filter):
    # Uvicorn endpoint access log filter
    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage().find("GET /metrics") == -1


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/", response_model=schemas.Message)
async def root() -> schemas.Message:
    return schemas.Message(message="Backend da aplicação está funcionando!")


@app.post("/predict", response_model=schemas.Prediction)
async def predict(historico_uma_semana: schemas.TimeSeries) -> schemas.Prediction:
    prediction = predict_request(historico_uma_semana)
    return schemas.Prediction(
        message="Backend da aplicação está funcionando!", prediction=prediction
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    # update uvicorn access logger format
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [trace_id=%(otelTraceID)s span_id=%(otelSpanID)s resource.service.name=%(otelServiceName)s] - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=EXPOSE_PORT, log_config=log_config)

    return None


if __name__ == "__main__":
    main()
