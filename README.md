# SOBRE O PROJETO

## Objetivo

O objetivo deste projeto é criar um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores de uma empresa à sua escolha e realizar toda a pipeline de desenvolvimento, desde a criação do modelo preditivo até o deploy do modelo em uma API que permita a previsão de preços de ações.

## Dados

O yfinance é uma biblioteca Python de código aberto que oferece uma maneira simples de baixar dados históricos de mercado do Yahoo Finance. Criada por Ran Aroussi como uma solução alternativa após a descontinuação da API do Yahoo Finance, o yfinance destaca-se por sua facilidade de uso e pela capacidade abrangente de recuperação de dados, tornando-se uma escolha popular entre as comunidades financeira e de ciência de dados.

Principais funcionalidades do yfinance:

- **Dados Históricos de Mercado**: Permite o download de preços históricos de ações, incluindo valores de abertura, máxima, mínima, fechamento e fechamento ajustado, além de volumes de negociação.
- **Ações Corporativas**: Fornece informações sobre ações corporativas, como dividendos e desdobramentos de ações, essenciais para análises financeiras precisas.
- **Demonstrações Financeiras**: Possibilita a obtenção de demonstrações financeiras, incluindo balanços patrimoniais, demonstrações de resultados e fluxos de caixa.
- **Dados de Resultados**: Oferece acesso a dados de resultados, como datas de divulgação, receitas e informações sobre lucro por ação (EPS).
- **Múltiplos Tickers**: Suporta a recuperação de dados para múltiplos tickers simultaneamente, permitindo uma análise eficiente de dados em um portfólio.

Neste projeto foram utilizados os dados de 13 ações conforme listado abaixo:

- DIS
- VALE3
- PETR4
- ITUB4
- ABEV3
- SANB11
- BBAS3
- JBSS3
- KLBN11
- BPAC11
- BBDC3
- ITSA4
- WEGE3

## Modelos

O objetivo é criar um modelo LSTM, que foi um dos modelos analisados por nós (`LSTMModel`, `LSTMModel_v2` e `LSTMModel_v3`), porém nos inspiramos nas arquiteturas chamadas *sequence to sequence*, propostas e utilizadas principalmente na área de Processamento de Linguagem Natural (NLP). Isso porque ganhamos uma flexibilidade no tamanho das sequências na entrada e na saída.

A arquitetura do modelo que foi escolhido para entrar em produção (`StockPredictionModel`) é um modelo sequence-to-sequence baseado em LSTM, projetado para prever valores futuros de ações a partir de dados históricos. Ela é dividida em duas partes principais: o encoder e o decoder.

A explicação da arquitetura está dada mais abaixo na documentação.

# CRIAÇÃO DO MODELO DE PREDIÇÃO

Antes de começar a mexermos no código, é importante que o **arquivo de configurações** (`./config/custom_lstm_config.yaml`) esteja com os devidos valores:

1. models: `**kwargs` dos modelos customizados que serão treinados posteriormente.
2. training: determina qual é a taxa de aprendizagem e o número de épocas do treinamento. 
3. dataset: determina quais são os tamanhos das sequências de entrada, saída e o número de ações dos dados de treinamento.
4. dataloader: determina o tamanho do lote (batch) e se os dados devem ser ordem aleatória, ou não.
5. api: define qual será o modelo, registrado pelo MLFlow, utilizado pela API.

## Dados

### Obtenção

Os dados foram obtidos através da biblioteca `yfinance` e salvos no diretório `./data/raw/` no formato *Delta Lake*.


> [!INFO] Para mais detalhes, olhar o código fonte em `./scripts/00_download_files.py`

### Preparação

O tratamento prévio dos dados foi bem simples:
- Cada linha representa uma observação com os valores de fechamento de um dia, e as colunas representam a qual ação o valor pertence.
- Considerou-se datas que possuem valores para **todas** as ações;

Criamos uma subclasse `Dataset` do módulo `torch.utils.data` para facilitar o seu uso durante o treinamento, pois pudemos customizar o método `__getitem__` de tal forma que retorna duas sequências, uma de entrada e outra de saída, com as dimensões estabelecidas no arquivo de configurações. Estes dados foram salvos em `./data/staged/`.

**CONSIDERANDO AS DIFERENÇAS ENTRE OS RESULTADOS DE MODELOS TREINADOS COM DADOS NORMALIZADOS E NÃO-NORMALIZADOS, OPTOU-SE POR NÃO UTILIZAR UM ESCALONADOR (*ex.: `MinMaxScaler`, `StandardScaler`, etc.*)**

> [!INFO] Para mais detalhes, olhar o código fonte em `./scripts/01_data_preparation.py`

## Modelos

### Sobre o uso do MLflow

Antes de falar sobre os modelos, vale a pena ressaltar que foi utilizado a biblioteca do MLflow para fazer todo o gerenciamento do ciclo de vida de modelos de machine learning, que abrange o rastreamento de experimentos, o registro de modelos e a organização de pipelines de treinamento. Em nosso código, o MLflow foi integrado à classe responsável pelo treinamento e teste dos modelos em PyTorch, permitindo que todas as informações importantes dos experimentos fossem registradas e armazenadas de forma centralizada.

Durante o treinamento, a função de treinamento inicia definindo um experimento com um nome que incorpora o nome do modelo e sua versão. Em seguida, é criada uma nova execução (run) no MLflow, onde são registradas tags e parâmetros relevantes, como o número de épocas e a taxa de aprendizado. Ao longo do treinamento, métricas como a perda média por época, a duração de cada época e o tempo total de treinamento são calculadas e logadas periodicamente usando funções como `mlflow.log_metric`. Após o término do treinamento, o modelo treinado é salvo e registrado no MLflow por meio do método `mlflow.pytorch.log_model`, que armazena não só o modelo, mas também um exemplo de entrada e a assinatura (`signature`) do modelo. Caso um `scaler` seja utilizado para normalizar os dados, ele também é salvo como um artefato, garantindo que todas as transformações aplicadas possam ser reproduzidas.

De maneira similar, durante o teste, o modelo é colocado em modo de avaliação e os dados de teste são processados para calcular a perda média e a duração total do teste, que também são registradas no MLflow. Assim, o MLflow permite acompanhar de forma detalhada cada etapa do processo, facilitando a comparação entre experimentos, o monitoramento do desempenho dos modelos e a gestão de diferentes versões do modelo. Essa integração é fundamental para manter a rastreabilidade e a reprodutibilidade em projetos de machine learning.

Os detalhes sobre a implementação do MLflow e todo  o processo de treino e de testes podem ser encontrados na classe `TreinadorDeModelos` em `rede_neural.custom_neural_networks`.

#### Armazenamento de modelos

Os modelos existentes estão registrados pela biblioteca MLFlow e podem ser consultados em um navegador via interface gráfica através do comando `mlflow ui`.

#### Carregando modelos registrados

Através da função `load_model_and_scaler` do nosso módulo customizado `common.utils`. Ela requer 3 informações/parâmetros de entrada:

1. `model_name`: O nome do modelo.
2. `version`: A versão do modelo.

Todas estas informações podem ser encontradas na interface gráfica do MLFlow.

- Exemplo de utilização:
	```python
	from common.utils import load_model_and_scaler
	
	model_name = "StockPredictionModel"
	version = "1"
	
	model, scaler = load_model_and_scaler(
	    model_name=model_name, version=version
	)
	```

> [!INFO] Sobre o `scaler`
> Se os dados usados no treinamento do modelo tenham sido modificados utilizando transformadores, como o `StandardScaler` ou o `MinMaxScaler` do Scikit Learn, esse transformador também será retornado pela função.

#### Criando um novo modelo

> [!WARNING] REQUISITOS OBRIGATÓRIOS
> 1. Ser uma **subclasse de `nn.Module`** do PyTorch.
> 2. A dimensão da **entrada** e da **saída** do modelo devem corresponder, respectivamente, às dimensões `dataset.input_len` e `dataset.target_len` que estão no arquivo `./config/custom_lstm_config.yaml`.

1. Crie o seu modelo
2. Insira ele em `./packages/rede-neural/src/rede_neural/custom_neural_networks.py`

Desta maneira o novo modelo já poderá ser importado via `rede_neural.custom_neural_networks` em outros códigos.

#### Treinando um novo modelo

Criei uma função que ajuda o processo de treinamento do modelo através de classe `TreinadorDeModelos`. Esta classe já se encarrega de utilizar o MLFlow para registrar os logs do sistema, parâmetros, artefatos, etc., facilitando bastante o fluxo.

Os parâmetros de entrada necessários para o `TreinadorDeModelos` são:

- `modelo` (**\***): O modelo, subclasse de `nn.Module`, que será utilizado para o treinamento.
- `config_modelo` (**\*\***):  São as `**kwargs` que serão passadas para a inicialização do `modelo`.
- `otimizador` (**\***):  O otimizador dos parâmetros do modelo.
- `fn_perda` (**\***):  A função de perda das predições do modelo.
- `lr` (**\*\***):  A taxa de aprendizado do otimizador.
- `num_epochs` (**\*\***):  Número de épocas do treinamento do modelo.

> [!WARNING] ATENÇÃO!
> (**\***) **SÃO CLASSES, E NÃO OBJETOS!!!**
> (**\*\***) *Recomendo utilizar o arquivo `./config/custom_lstm_config.yaml` para passar estas informações. Dê uma lida na seção sobre as configurações para saber mais*.

- Instanciando o `TreinadordeModelos`:
	```python
	from rede_neural.custom_neural_networks import (
	    StockPredictionLoss,
	    StockPredictionModel,
	    TreinadorDeModelos,
	)
	from torch.optim import Adam
	
	
	model_config = {
	    "num_stocks": 9,
	    "hidden_size": 128,
	    "num_layers": 4,
	    "dropout": 0.1,
	    "teacher_forcing_ratio": 0.25,
	}
	
	ash = TreinadorDeModelos(
		modelo=StockPredictionModel,
		config_modelo=model_config,
		otimizador=Adam,
		fn_perda=StockPredictionLoss,
		lr=0.01,
		num_epochs=100,
	)
	```

Para treinar, é preciso de mais três parâmetros:

- `dataloader`: Um objeto `DataLoader` do PyTorch contendo os dados de treino. É importante que o `__getitem__` do `Dataset` que está dentro do `DataLoader` retorne uma tupla onde o primeiro objeto são os dados de entrada (`X`) do modelo e o segundo é a saída desejada (`y`).
- `scaler`: É um parâmetro **opcional**, caso os dados tenham sido modificados utilizando transformadores como o `StandardScaler` ou o `MinMaxScaler` do Scikit Learn.
- `version`: É a versão do modelo que será registrado pelo MLFlow.

- Exemplo de treino e teste do modelo:
	```python
	ash.treina(dataloader=train_dataloader, scaler=scaler, version=version)
	ash.testa(dataloader=test_dataloader, version=version)
	```

Com isto, a rede neural terá sido treinada e catalogada pelo MLFlow, pronta para ser utilizada, caso queira, através da função `load_model_and_scaler`.


> [!INFO] Sobre o `ash`
> Referência ao treinador Pokémon chamado Ash.

### Detalhes da arquitetura *sequence to sequence*

Como mencionado na introdução, utilizamos uma arquitetura de modelo *sequence-to-sequence* baseado em LSTM como camada oculta, composta pelo encoder e pelo decoder.

No encoder, os dados de entrada – que contêm, por exemplo, os preços históricos de várias ações – são processados por uma rede LSTM bidirecional empilhada em múltiplas camadas. A característica bidirecional permite que a rede capture padrões temporais tanto no sentido cronológico normal quanto no reverso, enriquecendo a representação dos dados. Após a passagem pela LSTM, os outputs são normalizados usando uma camada de normalização (LayerNorm), o que ajuda a estabilizar o treinamento. Como a LSTM é bidirecional, a dimensão do estado oculto dobra; para alinhar essa saída com o que o decoder espera, é aplicada uma camada de projeção linear que mapeia a dimensão resultante para o tamanho desejado.

O decoder tem a função de gerar a sequência de previsões futuras. Ele também é composto por uma LSTM empilhada, mas, diferentemente do encoder, sua entrada a cada passo é a previsão anterior – ou, durante o treinamento, o valor real pode ser utilizado via técnica de teacher forcing, que ajuda na convergência do modelo. Além disso, o decoder incorpora um mecanismo de atenção multi-cabeça (MultiheadAttention) que permite que, em cada passo de previsão, ele foque nas partes mais relevantes da sequência processada pelo encoder. Essa atenção é somada à saída da LSTM, e o resultado passa por uma camada de normalização e dropout para evitar overfitting. Em seguida, uma camada linear mapeia o estado oculto para o espaço de saída, correspondendo ao número de ações que se deseja prever.

Outro ponto importante é a transformação dos estados ocultos do encoder para que se adequem ao formato esperado pelo decoder. Essa transformação envolve combinar (por exemplo, somando) as informações provenientes das duas direções da LSTM bidirecional, garantindo que o decoder receba uma representação unificada e coerente da sequência de entrada.

Em resumo, essa arquitetura combina a capacidade de capturar relações temporais complexas (por meio de LSTMs bidirecionais), a flexibilidade de focar em informações relevantes através do mecanismo de atenção e a eficácia do teacher forcing para melhorar o treinamento. Esses elementos, juntos, tornam o modelo especialmente adequado para a previsão de valores de ações, onde entender a dinâmica temporal dos dados é crucial para obter boas previsões.

# API

## Predição de valores

### Carregando o modelo

- Quando for carregar um modelo, preciso alterar **manualmente** todos os paths para que a raiz seja `/app`.
	- Quando eu crio um modelo, os arquivos `yaml` salvam o path absoluto e, ao usar o `load_model` em um código **dentro do Docker**, isso vai quebrar tudo.

### Formato das requisições

## Monitoramento da API

### Grafana + Prometheus
