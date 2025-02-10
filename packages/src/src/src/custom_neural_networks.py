import os
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Type

import joblib
import mlflow
import mlflow.pytorch
from mlflow.artifacts import download_artifacts
from mlflow.models import infer_signature
from sklearn.base import TransformerMixin
from torch import nn, optim
from torch.utils.data import DataLoader


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        output_sequence_length: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.output_sequence_length = output_sequence_length

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -self.output_sequence_length :, :])
        return out


class LSTMModel_v2(nn.Module):
    def __init__(
        self,
        num_stocks: int,
        num_lstm_layers: int,
        lstm_out_size: int,
        output_sequence_length: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_stocks,
            num_layers=num_lstm_layers,
            hidden_size=lstm_out_size,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_out_size, num_stocks)
        self.output_sequence_length = output_sequence_length
        return None

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -self.output_sequence_length :, :])
        return out


class TreinadorDeModelos:
    def __init__(
        self,
        modelo: Type[nn.Module],
        config_modelo: Dict,
        otimizador: Type[optim.Optimizer],
        fn_perda: Type[nn.modules.loss._Loss],
        lr: float = 0.01,
        num_epochs: int = 50,
    ) -> None:

        self.modelo = modelo(**config_modelo)

        self.fn_perda = fn_perda()

        self.otimizador = otimizador(params=self.modelo.parameters(), lr=lr)
        self.lr = lr

        self.num_epochs = num_epochs

        # MLflow
        self.nome_modelo = modelo.__name__
        self.nome_otimizador = otimizador.__name__
        self.nome_fn_perda = fn_perda.__name__

        return None

    def treina(
        self,
        dataloader: DataLoader,
        scaler: Optional[TransformerMixin] = None,
        version: str = "v0.0",
    ):

        experiment_name = f"Model {self.nome_modelo} - Experiment {version}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"{self.nome_modelo}_optimizer_{self.nome_otimizador}_lossfn_{self.nome_fn_perda}_{version}",
            log_system_metrics=True,
        ):
            mlflow.set_tags(
                {
                    "model": self.nome_modelo,
                    "version": version,
                    "optimizer": self.nome_otimizador,
                    "loss_fn": self.nome_fn_perda,
                }
            )
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("learning_rate", self.lr)

            self.modelo.train()
            for epoch in range(self.num_epochs):
                epoch_loss = 0.0
                num_losses = 0
                for x, y in dataloader:

                    if epoch == 0:
                        input_example = x.numpy()
                        signature = infer_signature(
                            input_example, self.modelo(x).detach().numpy()
                        )

                    h = self.modelo(x)
                    loss = self.fn_perda(h, y)
                    epoch_loss += loss.item()
                    num_losses += 1

                    self.otimizador.zero_grad()
                    loss.backward()
                    self.otimizador.step()

                avg_epoch_loss = epoch_loss / num_losses

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: Avg Loss = {avg_epoch_loss}")
                    mlflow.log_metric("avg_epoch_loss", avg_epoch_loss, step=epoch + 1)

            mlflow.pytorch.log_model(
                pytorch_model=self.modelo,
                artifact_path=f"{self.nome_modelo}_{version}",
                input_example=input_example,
                signature=signature,
            )

            if scaler:
                with TemporaryDirectory() as temp_dir:
                    scaler_path = os.path.join(
                        temp_dir, f"{self.nome_modelo}_{version}_scaler.pkl"
                    )
                    joblib.dump(scaler, scaler_path)
                    mlflow.log_artifact(
                        scaler_path, artifact_path=f"{self.nome_modelo}_{version}"
                    )

        return None

    def testa(
        self,
        dataloader: DataLoader,
        version: str = "v0.0",
    ):

        experiment_name = f"Model {self.nome_modelo} - Experiment {version}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(
            run_name=f"test_{self.nome_modelo}_optimizer_{self.nome_otimizador}_lossfn_{self.nome_fn_perda}_{version}",
            log_system_metrics=True,
        ):
            mlflow.set_tags(
                {
                    "model": self.nome_modelo,
                    "version": version,
                    "optimizer": self.nome_otimizador,
                    "loss_fn": self.nome_fn_perda,
                }
            )

            self.modelo.eval()
            losses = 0.0
            num_losses = 0
            for x, y in dataloader:

                h = self.modelo(x)
                loss = self.fn_perda(h, y)
                losses += loss.item()
                num_losses += 1

            avg_loss = losses / num_losses
            mlflow.log_metric("avg_loss", avg_loss)

            print(f"Testing completed. Average Loss: {avg_loss}")

        return None


def load_model_and_scaler(run_id: str, model_name: str, version: str):
    # Modelo
    model_uri = f"runs:/{run_id}/{model_name}_{version}"
    model = mlflow.pytorch.load_model(model_uri)

    # Scaler
    scaler_uri = (
        f"runs:/{run_id}/{model_name}_{version}/{model_name}_{version}_scaler.pkl"
    )
    try:
        scaler = joblib.load(download_artifacts(scaler_uri))
    except OSError:
        print("Não foi usado um scaler!")
        scaler = None

    return model, scaler
