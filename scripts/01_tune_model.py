import datetime
import platform

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import pandas as pd
import psutil
from deltalake import DeltaTable
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoLSTM
from src.utils import WMAPE, wmape

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
    df = DeltaTable("deltalake").to_pandas()
    df = df.sort_values(by=["unique_id", "ds"]).reset_index(drop=True)

    TEMPO_FILTRO_PARA_TESTE = "2018-01-01"

    df = df.loc[df["ds"] > TEMPO_FILTRO_PARA_TESTE]

    FILTRA_IDS = df["unique_id"].unique()[:2]

    df = df[df["unique_id"].isin(FILTRA_IDS)]

    train = df.loc[df["ds"] < "2024-09-01"]
    valid = df.loc[(df["ds"] >= "2024-09-01") & (df["ds"] < "2024-12-20")]
    h = valid["ds"].nunique()

    models = [AutoLSTM(h=h, num_samples=3, loss=WMAPE())]

    model = NeuralForecast(models=models, freq="D")

    with mlflow.start_run():
        log_system_info()

        model.fit(train, val_size=30)

        # Log parameters
        mlflow.log_param("h", h)
        mlflow.log_param("num_samples", 3)
        mlflow.log_param("loss", "WMAPE")

        # Save and log the model
        model_path = f"ml_models/neuralforecast_lstm_{datetime.datetime.now().date()}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        p = model.predict().reset_index()
        p = p.merge(valid[["ds", "unique_id", "y"]], on=["ds", "unique_id"], how="left")

        wmape_value = wmape(p["y"], p["AutoLSTM"])
        print("A avaliação do wmape é:", wmape_value)

        # Log metrics
        mlflow.log_metric("wmape", wmape_value)

        # Plot and save the figure
        fig, ax = plt.subplots(2, 1, figsize=(1280 / 96, 720 / 96))
        fig.tight_layout(pad=7.0)
        for ax_i, unique_id in enumerate(["ABEV3", "BBAS3"]):
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


if __name__ == "__main__":
    tuna_modelo_autolstm()
