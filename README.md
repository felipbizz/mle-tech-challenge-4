## Setup

Esse projeto utiliza o [UV](https://docs.astral.sh/uv/) para gerenciamento do projeto.

Para instalar siga a documentação em: https://docs.astral.sh/uv/getting-started/

## Roadmap

As tarefas que devem ser executadas estão definidas em: https://github.com/felipbizz/mle-tech-challenge-4/issues/1

## Documentação

### scripts/00_download_files.py
Este script baixa os dados históricos de ações usando a biblioteca `yfinance` e os salva em um Delta Lake. 

#### Funções principais:
- `download_files(symbols: list)`: Faz o download dos dados para cada símbolo na lista e os salva no Delta Lake.

#### Como usar:
1. Defina a lista de símbolos das ações.
2. Execute o script para baixar e salvar os dados.

### scripts/01_tune_model.py
Este script ajusta um modelo AutoLSTM usando os dados do Delta Lake e salva o modelo treinado.

#### Funções principais:
- `tuna_modelo_autolstm()`: Ajusta o modelo AutoLSTM, salva o modelo treinado e gera previsões.

#### Como usar:
1. Execute o script para ajustar o modelo e salvar os resultados.

### scripts/02_train_model.py
Este script treina um modelo LSTM usando os dados do Delta Lake e a melhor configuração definida.

#### Como usar:
1. Execute o script para treinar o modelo com os dados disponíveis.

### Configurações
As configurações do projeto estão definidas no arquivo `config/config.py`.

### Dependências
- `neuralforecast`
- `deltalake`
- `yfinance`
- `joblib`
- `matplotlib`
- `pandas`
- `tqdm`

### Executando os scripts
1. Baixe os dados executando `00_download_files.py`.
2. Ajuste o modelo executando `01_tune_model.py`.
3. Treine o modelo final executando `02_train_model.py`.
