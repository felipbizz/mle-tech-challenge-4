import pandas as pd
from deltalake import DeltaTable
from src.utils import WMAPE, wmape
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from ray.tune.search.hyperopt import HyperOptSearch
from neuralforecast.auto import AutoLSTM
import datetime
import joblib
from src.utils import setLog
import mlflow
import mlflow.pyfunc
import psutil
import platform

logger = setLog('tune_model', level=10)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.enable_system_metrics_logging()
mlflow.set_experiment("autolstm_experiment")

def log_system_info():
    mlflow.log_param("system", platform.system())
    mlflow.log_param("release", platform.release())
    mlflow.log_param("version", platform.version())
    mlflow.log_param("machine", platform.machine())
    mlflow.log_param("processor", platform.processor())
    mlflow.log_param("cpu_count", psutil.cpu_count())
    mlflow.log_param("memory", psutil.virtual_memory().total / (1024 ** 3))


def tuna_modelo_autolstm():

    # Carregando dados armazenados no DeltaLake
    df = DeltaTable('deltalake').to_pandas()
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    df = df.loc[df['ds'] > '2024-01-01']
    logger.info(f'Dataframe filtrado: {df.shape}')

    logger.info(f'Dados carregados para os seguintes símbolos {df["unique_id"].unique()}')

    # # Separando dados de treinamento e testes
    train = df.loc[df['ds'] < '2024-10-01']
    valid = df.loc[(df['ds'] >= '2024-10-01') & (df['ds'] < '2025-01-31')]

    h = valid['ds'].nunique()
    logger.info(f'Horizonte de treinamento definido como: {h}')

    models = [AutoLSTM(h=h, 
                    num_samples=30, 
                    loss=WMAPE())]

    model = NeuralForecast(models=models, freq='D')
    logger.info('Modelo carregado.')


    with mlflow.start_run():
        run_name = f"fiap_mle_fase4_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
        mlflow.set_tag('mlflow.runName', run_name)
        logger.info(f'Definindo o nome da execução do experimento como : {run_name}')
        log_system_info()

        initial_config = models[0].config
        logger.debug(f'Salvando configuração inicial: {initial_config}')
        
        model.fit(train, val_size=30)
        mlflow.log_param('hparams', model.models[0].model.hparams)
        logger.info(f'Hiperparâmetros: {model.models[0].model.hparams}')
        # best_hp = models[0].results.get_best_result().metrics['config']
        # best_hp = model.best_hyperparameters_
        # logger.info(f'Melhores parâmetros encontrados:\n {best_hp}')
    
            # Log parameters
        mlflow.log_param("h", h)
        mlflow.log_param("num_samples", 3)
        mlflow.log_param("loss", "WMAPE")

        trained_config = models[0].config
        logger.debug(f'Configuração após o treino: {trained_config}')

        model_path = f"ml_models/neuralforecast_autolstm_{datetime.datetime.now().date()}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Salvando modelo em: {model_path}")

        p = model.predict().reset_index()
        p = p.merge(valid[["ds", "unique_id", "y"]], on=["ds", "unique_id"], how="left")

        # Log metrics
        wmape_value = wmape(p["y"], p["AutoLSTM"])
        mlflow.log_metric("wmape", wmape_value)
        logger.info(f"A avaliação do wmape é: {wmape_value}")

        logger.debug(f"Stocks previstos: {p['unique_id'].unique()}")

        # Plot and save the figure
        fig, ax = plt.subplots(2, 1, figsize=(1280 / 96, 720 / 96))
        fig.tight_layout(pad=7.0)
        for ax_i, unique_id in enumerate(["ABEV3.SA", "BBAS3.SA"]):
            plot_df = pd.concat(
                [
                    train.loc[train["unique_id"] == unique_id].tail(30),
                    p.loc[p["unique_id"] == unique_id],
                ]
            ).set_index("ds")
            plot_df[["y", "AutoLSTM"]].plot(ax=ax[ax_i], linewidth=2, title=unique_id)

        plot_path = f"reports/forecast_plot_{datetime.datetime.now().date()}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        logger.info(f'Plot salvo em : {plot_path}')

if __name__ == '__main__':
    tuna_modelo_autolstm()