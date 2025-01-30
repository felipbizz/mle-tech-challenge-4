from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.controllers import model_controller, datalake_controller
from src.utils import setLog
from src.setup import detect_routers
from prometheus_client import make_asgi_app
import logfire


logger = setLog('api', level=10)
logger.info('---------------------------------------------------------------------------------------------------')
logger.info('Iniciando API')
app = FastAPI()

logger.info('Configurando acesso ao Pydantic Logfire.')
logfire.configure()
logfire.instrument_fastapi(app)

logger.info('Adicionando rotas.')
detect_routers(app)
# app.include_router(model_controller.router)
# app.include_router(datalake_controller.router)

logger.info('Configurando o instrumentador de métricas do cliente do Prometheus.')
metrics_app = make_asgi_app()

logger.info('Definindo o endpoint de coleta de métricas a ser usado pelo servidor do Prometheus.')
app.mount('/metrics', metrics_app)

logger.info('Realizando configuração inicial da API')

logger.info('Adicionando middleware para CORS')
origins = ['*']
methods = ['*']
headers = ['*']

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_methods=methods,
                   allow_headers=headers,
                   allow_credentials=True)

