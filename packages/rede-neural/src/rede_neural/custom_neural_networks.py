# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import os
from tempfile import TemporaryDirectory
from time import time
from typing import Dict, Optional, Type

import joblib
import mlflow
import mlflow.pytorch
import torch
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import TransformerMixin
from torch import nn, optim
from torch.utils.data import DataLoader

# =============================================================================
# CONSTANTES
# =============================================================================

# Instanciando o cliente do MLFlow
mlflow_client = MlflowClient()

# =============================================================================
# FUNÇÕES
# =============================================================================


def get_models_lastest_version(model_name: str) -> str:
    """
    Obtém a versão mais recente de um modelo registrado no MLflow.

    Args:
        model_name: Nome do modelo para o qual se deseja obter a versão mais recente.

    Returns:
        str: A versão mais recente do modelo como uma string. Caso não seja possível obter a versão, retorna "0".
    """
    try:
        # Obtém as versões mais recentes do modelo especificado.
        # stages=None indica que todas as versões devem ser consideradas, independentemente do estágio.
        latest_mv = mlflow_client.get_latest_versions(model_name, stages=None)[0]

        # A versão do modelo mais recente é armazenada na variável 'version'.
        version = latest_mv.version
    except:
        # Em caso de erro ao buscar a versão, retorna "0".
        version = "0"

    # Retorna a versão do modelo.
    return version


# =============================================================================
# CLASSES
# =============================================================================

# -----------------------------------------------------------------------------
# Modelos
# -----------------------------------------------------------------------------


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        output_sequence_length: int,
    ):
        """
        Modelo LSTM para previsão de sequências.

        Args:
            input_size: Número de características de entrada para cada passo da sequência.
            hidden_size: Número de unidades na camada oculta do LSTM.
            output_size: Número de características na saída do modelo (dimensão da previsão).
            output_sequence_length: Comprimento da sequência de saída que o modelo deve produzir.
        """
        super().__init__()

        # Inicializa a camada LSTM com 'input_size' unidades de entrada e 'hidden_size' unidades ocultas.
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Inicializa a camada totalmente conectada para mapear as saídas do LSTM para o tamanho de saída desejado.
        self.fc = nn.Linear(hidden_size, output_size)

        # Armazena o comprimento da sequência de saída, que será usado na camada final.
        self.output_sequence_length = output_sequence_length

    def forward(self, x):
        """
        Define a passagem para frente do modelo LSTM.

        Args:
            x: Um tensor de entrada com formato (batch_size, sequence_length, input_size).

        Returns:
            O tensor de saída com as previsões do modelo, com formato
            (batch_size, output_sequence_length, output_size).
        """
        # Passa a entrada pela camada LSTM, que retorna a saída e o estado oculto.
        out, _ = self.lstm(x)

        # Seleciona as saídas das últimas 'output_sequence_length' etapas,
        # e passa pela camada totalmente conectada.
        out = self.fc(out[:, -self.output_sequence_length :, :])

        # Retorna as previsões do modelo.
        return out


class LSTMModel_v2(nn.Module):
    def __init__(
        self,
        num_stocks: int,
        num_lstm_layers: int,
        lstm_out_size: int,
        output_sequence_length: int,
    ) -> None:
        """
        Modelo LSTM de segunda versão para previsão de séries temporais, adaptado para múltiplas ações.

        Args:
            num_stocks: Número de ações (features) que o modelo irá processar.
            num_lstm_layers: Número de camadas LSTM empilhadas.
            lstm_out_size: Número de unidades na camada oculta da LSTM.
            output_sequence_length: Comprimento da sequência na saída que o modelo deve produzir.
        """
        super().__init__()

        # Inicializa a camada LSTM com 'num_stocks' unidades de entrada,
        # 'num_lstm_layers' camadas LSTM, e 'lstm_out_size' unidades ocultas.
        self.lstm = nn.LSTM(
            input_size=num_stocks,
            num_layers=num_lstm_layers,
            hidden_size=lstm_out_size,
            batch_first=True,
        )

        # Inicializa a camada totalmente conectada para mapear a saída do LSTM
        # para o número de ações, 'num_stocks'.
        self.fc = nn.Linear(lstm_out_size, num_stocks)

        # Armazena o comprimento da sequência de saída para referência posterior.
        self.output_sequence_length = output_sequence_length

    def forward(self, x):
        """
        Realiza a passagem para frente do modelo LSTM.

        Args:
            x: Um tensor de entrada com formato (batch_size, sequence_length, num_stocks).

        Returns:
            O tensor de saída com previsões do modelo, com formato
            (batch_size, output_sequence_length, num_stocks).
        """
        # Passa os dados de entrada pela camada LSTM, que retorna a saída e o estado oculto.
        out, _ = self.lstm(x)

        # Seleciona as saídas das últimas 'output_sequence_length' etapas,
        # e as passa pela camada totalmente conectada.
        out = self.fc(out[:, -self.output_sequence_length :, :])

        # Retorna as previsões do modelo.
        return out


class LSTMModel_v3(nn.Module):
    def __init__(
        self,
        num_stocks: int,
        num_lstm_layers: int,
        lstm_out_size: int,
        output_sequence_length: int,
    ) -> None:
        """
        Modelo LSTM de terceira versão com mecanismos de encoder-decoder e atenção para previsão de séries temporais.

        Args:
            num_stocks: Número de ações (features) que o modelo irá processar.
            num_lstm_layers: Número de camadas LSTM empilhadas.
            lstm_out_size: Número de unidades na camada oculta da LSTM.
            output_sequence_length: Comprimento da sequência na saída que o modelo deve produzir.
        """
        super().__init__()
        self.output_sequence_length = output_sequence_length

        # Inicializa a camada LSTM do encoder.
        self.encoder = nn.LSTM(
            input_size=num_stocks,
            hidden_size=lstm_out_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=(
                0.2 if num_lstm_layers > 1 else 0
            ),  # Aplica dropout entre camadas se houver mais de uma camada
        )

        # Inicializa a camada LSTM do decoder.
        self.decoder = nn.LSTM(
            input_size=num_stocks,
            hidden_size=lstm_out_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Camada de atenção para combinar informações do encoder e do decoder.
        self.attention = nn.Linear(lstm_out_size * 2, 1)

        # Camadas totalmente conectadas para processamento adicional da saída.
        self.fc1 = nn.Linear(lstm_out_size, lstm_out_size)
        self.fc2 = nn.Linear(lstm_out_size, num_stocks)

        # Camada dropout para regularização.
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Realiza a passagem para frente do modelo com encoder-decoder.

        Args:
            x: Um tensor de entrada com formato (batch_size, sequence_length, num_stocks).

        Returns:
            Um tensor de saída com as previsões do modelo, com formato
            (batch_size, output_sequence_length, num_stocks).
        """
        # Passa os dados pela camada do encoder.
        encoder_out, (hidden, cell) = self.encoder(x)

        # Prepara a entrada inicial para o decoder usando o último passo da entrada conhecida.
        decoder_input = x[:, -1:, :]  # Formato (batch_size, 1, num_stocks)

        outputs = []  # Lista para armazenar as previsões ao longo do tempo
        for _ in range(self.output_sequence_length):
            # Passa a entrada do decoder e os estados ocultos para gerar saídas.
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            # Implementação de um mecanismo de atenção simples.
            attn_weights = torch.tanh(
                self.attention(torch.cat((decoder_out, encoder_out), dim=2))
            )
            attn_weights = torch.softmax(
                attn_weights, dim=1
            )  # Normaliza pesos de atenção
            context = torch.sum(
                encoder_out * attn_weights, dim=1, keepdim=True
            )  # Gera contexto

            # Concatena o contexto com a saída do decoder.
            decoder_out = torch.cat((decoder_out, context), dim=2)

            # Passa a saída concatenada pelas camadas totalmente conectadas.
            out = self.dropout(torch.relu(self.fc1(decoder_out)))
            pred = self.fc2(out)

            outputs.append(pred)  # Armazena a previsão
            decoder_input = pred  # Usa a previsão como a nova entrada do decoder (abordagem auto-regressiva)

        # Retorna as previsões concatenadas ao longo do tempo.
        return torch.cat(outputs, dim=1)


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Encoder do modelo sequence-to-sequence

        Args:
            input_size: Número de features de entrada (número de ações)
            hidden_size: Dimensão do estado oculto do LSTM
            num_layers: Número de camadas LSTM empilhadas
            dropout: Taxa de dropout para regularização
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,  # LSTM bidirecional para capturar padrões em ambas direções
        )

        # Camada de normalização para ajudar na estabilidade do treinamento
        self.layer_norm = nn.LayerNorm(
            hidden_size * 2
        )  # *2 devido ao LSTM bidirecional


class Decoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Decoder do modelo sequence-to-sequence

        Args:
            output_size: Dimensão da saída (número de ações)
            hidden_size: Dimensão do estado oculto do LSTM
            num_layers: Número de camadas LSTM empilhadas
            dropout: Taxa de dropout para regularização
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=output_size,  # A entrada será a previsão anterior
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Projeção do estado oculto para o espaço de saída
        self.fc = nn.Linear(hidden_size, output_size)

        # Camada de atenção para focar em partes relevantes da sequência de entrada
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)


class StockPredictionModel(nn.Module):
    def __init__(
        self,
        num_stocks: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        sequence_target_len: int = 7,
        dropout: float = 0.1,
        teacher_forcing_ratio: float = 0.5,
    ):
        """
        Modelo sequence-to-sequence completo para previsão de preços

        Args:
            num_stocks: Número de ações sendo previstas
            hidden_size: Dimensão do estado oculto do LSTM
            num_layers: Número de camadas LSTM
            dropout: Taxa de dropout
            teacher_forcing_ratio: Probabilidade de usar teacher forcing durante o treinamento
        """
        super().__init__()

        self.sequence_target_len = sequence_target_len
        self.encoder = Encoder(num_stocks, hidden_size, num_layers, dropout)
        self.decoder = Decoder(num_stocks, hidden_size, num_layers, dropout)
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Add a projection layer to map encoder outputs from 256 (bidirectional) to 128.
        self.encoder_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Inicialização dos pesos usando Kaiming initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, src, target=None, teacher_forcing_ratio=None):
        """
        Forward pass do modelo

        Args:
            src: Tensor de entrada [batch_size, seq_len, num_stocks]
            target: Tensor opcional com valores reais futuros para teacher forcing
            target_len: Número de steps futuros para prever
            teacher_forcing_ratio: Taxa de uso de teacher forcing
        """
        batch_size = src.size(0)
        num_stocks = src.size(2)

        # Validações de dimensionalidade
        if src.size(2) != num_stocks:
            raise ValueError(
                f"Input deve ter {num_stocks} features, mas tem {src.size(2)}"
            )

        encoder_outputs, (hidden, cell) = self.encoder.lstm(src)
        encoder_outputs = self.encoder.layer_norm(encoder_outputs)

        # Project encoder outputs to match decoder dimension
        encoder_outputs_proj = self.encoder_proj(encoder_outputs)

        # Inicializa a entrada do decoder e tensor de saída
        decoder_input = src[:, -1:, :]
        outputs = torch.zeros(batch_size, self.sequence_target_len, num_stocks).to(
            src.device
        )

        # Define teacher forcing ratio
        tf_ratio = (
            teacher_forcing_ratio
            if teacher_forcing_ratio is not None
            else self.teacher_forcing_ratio
        )

        hidden = self._transform_hidden_for_decoder(hidden)
        cell = self._transform_hidden_for_decoder(cell)

        for t in range(self.sequence_target_len):
            decoder_output, (hidden, cell) = self.decoder.lstm(
                decoder_input, (hidden, cell)
            )

            # Use the projected encoder outputs in the attention layer
            attn_output, _ = self.decoder.attention(
                decoder_output, encoder_outputs_proj, encoder_outputs_proj
            )

            decoder_output = self.decoder.layer_norm(decoder_output + attn_output)
            decoder_output = self.decoder.dropout(decoder_output)

            prediction = self.decoder.fc(decoder_output)
            outputs[:, t : t + 1, :] = prediction

            # Implementa teacher forcing
            if target is not None and torch.rand(1).item() < tf_ratio:
                decoder_input = target[:, t : t + 1, :]
            else:
                decoder_input = prediction

        return outputs

    def _transform_hidden_for_decoder(self, hidden):
        """
        Transforma os estados ocultos do encoder bidirecional para o formato do decoder
        """
        # Combina as direções do LSTM bidirecional
        num_directions = 2
        hidden = hidden.view(
            self.decoder.num_layers, num_directions, -1, self.decoder.hidden_size
        )
        return hidden.sum(dim=1)


# -----------------------------------------------------------------------------
# Funções de perda customizadas
# -----------------------------------------------------------------------------


class StockPredictionLoss(nn.Module):
    """
    Função de perda personalizada que combina MSE com penalização por mudanças bruscas
    """

    def __init__(self, smoothness_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smoothness_weight = smoothness_weight

    def forward(self, pred, target):
        # Erro de previsão básico
        mse_loss = self.mse(pred, target)

        # Penalização por mudanças bruscas entre timesteps consecutivos
        smoothness_loss = torch.mean(torch.abs(pred[:, 1:] - pred[:, :-1]))

        return mse_loss + self.smoothness_weight * smoothness_loss


# -----------------------------------------------------------------------------
# Classe para ajudar no treinamento dos modelos
# -----------------------------------------------------------------------------


class TreinadorDeModelos:
    def __init__(
        self,
        modelo: Type[nn.Module],
        config_modelo: Dict,
        otimizador: Type[optim.Optimizer],
        fn_perda: Type[nn.Module],
        lr: float = 0.01,
        num_epochs: int = 50,
    ) -> None:
        """
        Classe para treinar e testar modelos utilizando PyTorch.

        Args:
            modelo: Classe do modelo a ser treinado, deve herdar de nn.Module.
            config_modelo: Dicionário de configuração para inicializar o modelo.
            otimizador: Classe do otimizador a ser utilizado durante o treinamento.
            fn_perda: Classe da função de perda a ser utilizada.
            lr: Taxa de aprendizado para o otimizador (default: 0.01).
            num_epochs: Número de épocas para o treinamento (default: 50).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inicializa o modelo com as configurações fornecidas e move para o dispositivo apropriado (GPU ou CPU).
        self.modelo = modelo(**config_modelo).to(self.device)

        # Inicializa a função de perda.
        self.fn_perda = fn_perda()

        # Inicializa o otimizador com os parâmetros do modelo e a taxa de aprendizado especificada.
        self.otimizador = otimizador(params=self.modelo.parameters(), lr=lr)
        self.lr = lr

        # Armazena o número de épocas para o treinamento.
        self.num_epochs = num_epochs

        # Informações de MLflow para rastreamento de experimentos.
        self.nome_modelo = modelo.__name__
        self.nome_otimizador = otimizador.__name__
        self.nome_fn_perda = fn_perda.__name__

    def treina(
        self,
        dataloader: DataLoader,
        scaler: Optional[TransformerMixin] = None,
    ):
        """
        Treina o modelo utilizando os dados fornecidos através de um DataLoader.

        Args:
            dataloader: DataLoader contendo os dados de treinamento.
            scaler: Scaler opcional a ser utilizado para salvar as transformeções dos dados (default: None).
        """
        # Obtém a versão mais recente do modelo registrado.
        previous_version = get_models_lastest_version(model_name=self.nome_modelo)
        version = int(previous_version) + 1

        # Marca o início do treinamento.
        training_start = time()
        avg_epoch_duration = 0

        # Define o nome do experimento no MLflow.
        experiment_name = f"Model {self.nome_modelo} - Experiment {version}"
        mlflow.set_experiment(experiment_name)

        # Inicia a execução do experimento no MLflow.
        with mlflow.start_run(
            run_name=f"{self.nome_modelo}_optimizer_{self.nome_otimizador}_lossfn_{self.nome_fn_perda}_{version}",
            log_system_metrics=True,
        ):
            # Define tags para o rastreamento de informações.
            mlflow.set_tags(
                {
                    "model": self.nome_modelo,
                    "version": version,
                    "device": self.device,
                    "optimizer": self.nome_otimizador,
                    "loss_fn": self.nome_fn_perda,
                }
            )
            # Registra parâmetros do experimento.
            mlflow.log_param("num_epochs", self.num_epochs)
            mlflow.log_param("learning_rate", self.lr)

            # Coloca o modelo em modo de treinamento.
            self.modelo.train()
            for epoch in range(self.num_epochs):
                # Registra o tempo do início da época.
                t0_epoch = time()

                epoch_loss = 0.0
                num_losses = 0
                for x, y in dataloader:
                    # Move os dados para o dispositivo apropriado.
                    x, y = x.to(self.device), y.to(self.device)

                    if epoch == 0:
                        # Inferência de assinatura para rastreamento do modelo (apenas na primeira época).
                        input_example = x.cpu().numpy()
                        signature = infer_signature(
                            input_example, self.modelo(x).detach().cpu().numpy()
                        )

                    # Realiza a predição e calcula a perda.
                    h = self.modelo(x)
                    loss = self.fn_perda(h, y)
                    epoch_loss += loss.item()
                    num_losses += 1

                    # Zera os gradientes, realiza a retropropagação e atualiza os pesos.
                    self.otimizador.zero_grad()
                    loss.backward()
                    self.otimizador.step()

                # Registra a duração da época.
                epoch_duration = time() - t0_epoch
                avg_epoch_duration += epoch_duration

                # Calcula a perda média da época.
                avg_epoch_loss = epoch_loss / num_losses

                if (epoch + 1) % 10 == 0:
                    # Imprime e loga informações a cada 10 épocas.
                    print(
                        f"Epoch {epoch + 1}: Avg Loss = {avg_epoch_loss:.6f}\tt: {epoch_duration:.4f}s"
                    )
                    mlflow.log_metric("avg_epoch_loss", avg_epoch_loss, step=epoch + 1)
                    mlflow.log_metric("epoch_duration", epoch_duration, step=epoch + 1)

            # Registra a duração total do treinamento.
            training_duration = time() - training_start
            print(f"Training duration: {training_duration:.4f}s")
            mlflow.log_metric("training_duration", training_duration)

            # Calcula e registra a duração média das épocas.
            avg_epoch_duration /= self.num_epochs
            print(f"Average epoch duration: {avg_epoch_duration:.4f}s")
            mlflow.log_metric("avg_epoch_duration", avg_epoch_duration)

            # Loga o modelo treinado no MLflow.
            mlflow.pytorch.log_model(
                pytorch_model=self.modelo,
                artifact_path=self.nome_modelo,
                input_example=input_example,
                signature=signature,
                registered_model_name=self.nome_modelo,
            )

            # Se um scaler foi fornecido, salva o modelo de scaler.
            if scaler:
                with TemporaryDirectory() as temp_dir:
                    scaler_path = os.path.join(
                        temp_dir, f"{self.nome_modelo}_scaler.pkl"
                    )
                    joblib.dump(scaler, scaler_path)
                    mlflow.log_artifact(scaler_path, artifact_path=self.nome_modelo)

    def testa(
        self,
        dataloader: DataLoader,
    ):
        """
        Testa o modelo utilizando os dados fornecidos através de um DataLoader.

        Args:
            dataloader: DataLoader contendo os dados de teste.
        """
        # Obtém a versão mais recente do modelo registrado para teste.
        version = get_models_lastest_version(model_name=self.nome_modelo)

        # Marca o início do teste.
        testing_start = time()

        # Define o nome do experimento de teste no MLflow.
        experiment_name = f"Model {self.nome_modelo} - Experiment {version}"
        mlflow.set_experiment(experiment_name)

        # Inicia a execução do experimento de teste no MLflow.
        with mlflow.start_run(
            run_name=f"test_{self.nome_modelo}_optimizer_{self.nome_otimizador}_lossfn_{self.nome_fn_perda}_{version}",
            log_system_metrics=True,
        ):
            # Define tags para o rastreamento de informações.
            mlflow.set_tags(
                {
                    "model": self.nome_modelo,
                    "version": version,
                    "device": self.device,
                    "optimizer": self.nome_otimizador,
                    "loss_fn": self.nome_fn_perda,
                }
            )

            # Coloca o modelo em modo de avaliação.
            self.modelo.eval()
            losses = 0.0
            num_losses = 0
            for x, y in dataloader:
                # Move os dados para o dispositivo apropriado.
                x, y = x.to(self.device), y.to(self.device)

                # Realiza a predição e calcula a perda para os dados de teste.
                h = self.modelo(x)
                loss = self.fn_perda(h, y)
                losses += loss.item()
                num_losses += 1

            # Calcula a perda média nos dados de teste.
            avg_loss = losses / num_losses
            mlflow.log_metric("avg_loss", avg_loss)

            print(f"Testing completed. Average Loss: {avg_loss}")

            # Registra a duração total do teste.
            testing_duration = time() - testing_start
            print(f"Testing duration: {testing_duration:.4f}s")
            mlflow.log_metric("testing_duration", testing_duration)

        return None
