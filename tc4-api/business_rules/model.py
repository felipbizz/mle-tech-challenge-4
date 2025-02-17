from src.utils import setLog
from neuralforecast import NeuralForecast
import joblib
import os
import pandas as pd
from datetime import datetime
from deltalake import DeltaTable
import matplotlib.pyplot as plt
import polars as pl

logger = setLog("model")


def make_predictions(model_file: str, stock_option: str) -> dict:
    logger.info(f"Carregando o modelo {model_file}")
    try:
        model: NeuralForecast = load_model(model_file)
        logger.info(f"Modelo {model_file} carregado. Tipo do modelo {type(model)}")

        if model_file.__contains__("autolstm"):
            y_pred_name: str = "AutoLSTM"
        else:
            y_pred_name: str = "LSTM"

        logger.debug(f"Chave utilizada na plotagem do resultado: {y_pred_name}")

        logger.info(
            "Realizando previsões utilizando o horizonte definido no treinamento do modelo."
        )
        predict_result: pd.DataFrame = model.predict()

        plot_df: pd.DataFrame = predict_result[
            predict_result.unique_id == stock_option
        ].drop("unique_id", axis=1)

        logger.info(f"Resultado da previsão: {predict_result}")

        df_hist = pl.read_delta("deltalake").to_pandas()
        df_hist = df_hist.sort_values(by=["unique_id", "ds"]).reset_index(drop=True)
        logger.info(
            f"Dados do deltalake carregados. Tamanho do dataset: {len(df_hist)}"
        )

        df_hist.drop_duplicates(subset=["ds", "unique_id"], inplace=True)
        logger.info(
            f"Dados após a remoção de duplicatas. Tamanho do dataset: {len(df_hist)}"
        )

        df_hist = df_hist[df_hist["ds"] >= "2024-06-01"]
        df_hist = df_hist[df_hist["unique_id"] == stock_option]
        logger.debug(f"Header histórico:\n{df_hist.shape}")
        logger.debug(f"Header previsto:\n{predict_result.shape}")
        logger.debug(f"Stock: {df_hist['unique_id'].unique()}")

        plot_df = pd.concat([df_hist, plot_df])
        logger.debug(f"Dados concatenados : \n{plot_df.shape}")

        logger.info(f"Gerando plot para as previsões de {stock_option}")
        plt.plot(
            df_hist["ds"],
            df_hist["y"],
            c="black",
            label=f"Valores históricos para {stock_option}",
        )
        plt.plot(
            plot_df["ds"],
            plot_df[y_pred_name],
            c="purple",
            label=f"Previsões para {stock_option}",
        )
        plt.legend()
        plt.grid()
        plt.plot()

        plot_name: str = f"reports/neuralforecast_lstm_{stock_option}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"

        logger.info(f"Imagem do plot salva em {plot_name}")
        plt.savefig(plot_name, dpi=300)
        plt.close()

        return {"message": f"Previsão concluída, consulte a imagem em {plot_name}"}

    except Exception as e:
        logger.error(e)
        raise


def load_model(model_file: str) -> NeuralForecast:
    """
    Carrega o modelo treinado a partir do arquivo informado.

    Parameters:

        model_file (str) : Nome do modelo a ser usado na previsão.

    Returns:

        model : Modelo carregado usando a biblioteca joblib.
    """
    logger.info(f"Validando a existência do modelo treinado : {model_file}")

    if os.path.exists(f"ml_models/{model_file}"):
        logger.info(f"Modelo {model_file} encontrado na pasta ml_models.")
        return joblib.load(f"ml_models/{model_file}")
    else:
        logger.error(
            f"Arquivo {model_file} não encontrado na pasta ml_models. Verifique o nome do modelo."
        )
        raise ValueError(
            f"Arquivo {model_file} não encontrado! Verifique o nome do modelo e tente novamente!"
        )
