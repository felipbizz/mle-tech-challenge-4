# FIAP - Tech Challenge - Machine Learning Engineering - Fase 4

## Grupo 9

<details open>

<summary> Expandir/Ocultar... </summary>

| **Nome**       | **RM** |
| :------------- | :----: |
| Diogo Padilha  | 357526 |
| Felipe Bizzo   | 356970 |
| Gabriel Rony   | 357376 |
| Lucas Alecrim  | 357415 |
| Thales Gomes   | 357646 |

</details>

## Setup

<details open>

<summary> Expandir/Ocultar... </summary>

Esse projeto utiliza o [UV](https://docs.astral.sh/uv/) para gerenciamento do projeto.  
Para instalar siga a documentação em: https://docs.astral.sh/uv/getting-started/  

</details>

## Roadmap

<details open>

<summary> Expandir/Ocultar... </summary>

As tarefas que devem ser executadas estão definidas em: https://github.com/felipbizz/mle-tech-challenge-4/issues/1

</details>

## Premissas do projeto

<details open>

<summary> Expandir/Ocultar... </summary>

Para limitar o escopo de treinamento do modelo restringimos as ações avaliadas às seguintes empresas:

|Cód.|Empresa|
|---|---|
|DIS|Disney|
|VALE3.SA|Vale SA|
|PETR4.SA|Petroleo Brasileiro SA Petrobras Preference Shares|
|ITUB4.SA|Itau Unibanco Holding SA Preference Shares|
|ABEV3.SA|Ambev SA|
|BBDC4.SA|Banco Bradesco SA Preference Shares|
|SANB11.SA|SANTANDER BR UNT|
|BBAS3.SA|Banco do Brasil SA|
|JBSS3.SA|JBS SA|
|KLBN11.SA|KLABIN S/A UNT N2|
|BPAC11.SA|BTG PACTUAL BANCO UNT|
|BBDC3.SA|BRADESCO ON EJ N1|
|ITSA4.SA|ITAUSA PN|
|WEGE3.SA|Weg SA|

Para o monitoramento online da API foi empregada a integração do FastAPI com o Pydantic Logfire.  
Vale ressaltar que para utilizar o Pydantic Logfire é necessário realizar o registro no portal (sendo possível utilizar as credenciais do GitHub como Single Sign On).  
Uma vez registrado, siga as instruções encontradas nas referências abaixo para configurar o projeto e criar as credenciais necessárias para o envio de métricas.  
Para a correta execução do ambiente do docker compose será necessário injetar as credenciais do Logfire na imagem da API.
Mais informações sobre como realizar a build se encontram em seções abaixo. 

> Referências (acessadas em 29/01/2025):  
> [Criando um projeto no Pydantic Logfire](https://logfire.pydantic.dev/docs/)  
> [Criando tokens de acesso ao projeto do Pydantic Logfire](https://logfire.pydantic.dev/docs/how-to-guides/create-write-tokens/)  
> [Integrando o FastAPI com o Pydantic Logfire](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/)  

# Definição do modelo
## Buscando os melhores hiperparâmetros utilizando o AutoLSTM

Para a execução deste trabalho fizemos uso do modelo LSTM da biblioteca NeuralForecast desenvolvida pela NIXTLA.  
Esta biblioteca possui uma função de autoajuste (AutoLSTM) que foi utilizada para a definição dos hiperparâmetros utilizados no treinamento do modelo produtivo.  

> **Referência**: https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html

Como função de erro, desenvolvemos uma variação da função MAPE que faz uso de valores ponderados no cálculo do erro.  
Esta função se chama WMAPE (Weighted Mean Absolute Percentage Error) e pode ser encontrada como função utilitária no projeto.

> **Referência**: https://lightning.ai/docs/torchmetrics/stable/regression/weighted_mean_absolute_percentage_error.html

![Função de erro WMAPE](readme_files/WMAPE.png)

</details>

## Imagens docker do projeto

<details open>

<summary> Expandir/Ocultar... </summary>

| Imagem Docker | Descrição |
| :---: | :--- |
| mle-api | API para execução das tarefas |
| prometheus | Servidor Prometheus |
| grafana | Servidor Grafana |
| mlflow | Servidor MLFlow |
| front | Frontend Streamlit |

</details>

### Gerando a imagem da API

<details open>

<summary> Expandir/Ocultar... </summary>

Garanta que o logfire está autenticado e que o arquivo de credencial exista no caminho <APP>/.logfire/logfire_credentials.json  

Caso decida por não utilizar o Logfire, execute o script shell abaixo na raiz da API (_tc4-api_) abaixo para criar o arquivo de credenciais vazio.

```bash
sh inicializarLogfireCredentials.sh
```

![Criando credenciais vazias](readme_files/EmptyCredentialsCreation.png)

> **Importante**  
> O comando abaixo deve ser executado na raiz da API (_tc4-api_), e não na raiz do projeto do GitHub.

```bash
docker build -f Dockerfile -t mle-api --secret id=logfire,src=.logfire/logfire_credentials.json .
```

Para iniciar todos os containers necessários para a execução do projeto, basta executar o comando a seguir:  

> **Importante**  
> Os modelos foram treinados em um ambiente com GPU e, portanto, podem gerar o erro abaixo caso não sejam executados em um ambiente que não a possua.

![Erro por falta de GPU](readme_files/NoGPU-Error.png)

```bash
docker-compose up -d
```

O Docker Compose fará a build da imagem do frontend na primeira execução.

Caso seja feita alguma alteração ao código, será necessário forçar o rebuild da imagem.

```bash
docker-compose build --no-cache
```

</details>

## Acessando a API

Acesse a URL : http://localhost:8000/docs para ter acesso ao SwaggerUI

### Atualizando o DeltaLake com dados do Yahoo! Finance

<details open>

<summary> Expandir/Ocultar... </summary>  

Ao executar o endpoint de download, os símbolos informados no corpo da requisição serão baixados e inseridos no DeltaLake.  
É importante notar que caso não haja valores para um certo símbolo ele será considerado como um erro de carga.

![Dados baixados](readme_files/DeltalakeDownload.png)

</details>


### Listando modelos treinados disponíveis para previsões

<details open>

<summary> Expandir/Ocultar... </summary>

Utilize o endpoint 'list' para obter os modelos disponíveis.  
Dentre os modelos, poderão ser listados tanto modelos gerados durante o ajuste de hiperparâmetros quanto modelos treinados com os parâmetros informados à API de treinamento.  

![Modelos disponíveis](readme_files/AvailableModels.png)

</details>

### Ajustando o modelo em busca dos melhores hiperparâmetros

<details open>

<summary> Expandir/Ocultar... </summary>

Este é processo que pode levar bastante tempo e, por isso, é recomendado que só seja realizado quando necessário.  
Ao final da execução serão retornados os melhores hiperparâmetros encontrados durante a fase de ajuste.  
O ajuste abaixo foi executado em 15m23s.

![Ajuste do modelo](readme_files/ModelTuning.png)

</details>

### Treinando o modelo

<details open>

<summary> Expandir/Ocultar... </summary>

De posse dos hiperparâmetros encontrados na fase de ajuste, é possível executar o treinamento do modelo que será utilizado para as previsões.  
Este endpoint retorna o nome do arquivo salvo contendo o modelo treinado no formato JOBLIB.

![Treinamento do modelo](readme_files/ModelTraining.png)

</details>

### Realizando previsões

<details open>

<summary> Expandir/Ocultar... </summary>

Informe a ação (símbolo) e o modelo a ser usado na previsão.

> Uma lista com os modelos treinados disponíveis para previsão pode ser obtida através de um endpoint da própria API.

![Endpoint de previsão](readme_files/PredictionEndpoint.png)

Ao final da previsão o endpoint irá gerar uma imagem na pasta _reports_ com o resultado similar à imagem abaixo:

![Resultado da previsão](readme_files/neuralforecast_lstm_VALE3.SA_20250216_1659.png)

</details>

# Visualizando métricas

<details open>

<summary> Expandir/Ocultar... </summary>

## Dados do TensorBoard

Durante a fase de ajuste, é possível visualizar os dados do Tensorboard que são expostos pelo container.  
Os arquivos de log a serem utilizados para a visualização do TensorBoard se encontram em: volumes/mle-api/ray/_id-da-sessao_/artifacts/_data-da-sessao_/__train_tune_data-de-execucao_/driver_artifacts

```bash
tensorboard --logdir volumes/mle-api/ray/_id-da-sessao_/artifacts/_data-da-sessao_/__train_tune_data-de-execucao_/driver_artifacts
```

![Executando o TensorBoard](readme_files/StartingTensorBoard.png)

O dashboard será carregado na URL http://localhost:6006/  
Neste dashboard é possível ver o comparativo dos gráficos gerados por cada amostra realizada pelo AutoLSTM.

![Dashboard do TensorBoard](readme_files/TensorBoardDashboard.png)

## Visualize as métricas da API online através do portal Logfire

O acesso ao dashboard do Logfire irá variar de acordo com o usuário e projetos utilizados na ferramenta.  
A URL abaixo é apenas uma referência, uma vez que será solicitada a credencial de acesso para o usuário.  
URL : https://logfire.pydantic.dev/_usuario_/_nome_do_projeto_

![Dashboard do Logfire](readme_files/LogfireDashboard.png)

## Obtendo métricas coletadas diretamente do cliente do Prometheus

Acesse a URL : http://localhost:8000/metrics/

ou, execute o comando abaixo em um terminal:

```bash
curl http://localhost:8000/metrics/
```

![Métricas do cliente do Prometheus](readme_files/PrometheusRawData.png)

## Acessando a interface do servidor Prometheus

Acesse a URL: http://localhost:9090/query

Através desta interface é possível consultar métricas diretamente do servidor do Prometheus.

![Servidor do Prometheus](readme_files/PrometheusServer.png)

## Acessando a interface do servidor Grafana

Acesse a URL : http://localhost:3000

> Acesse o dashboard utilizando as credenciais admin/admin.  
> É possível que seja solicitado que esta credencial seja alterada no primeiro acesso.

### Definição da fonte de dados

O Grafana foi configurado para receber dados coletados pelo servidor Prometheus.  
Detalhes desta configuração estão fora do escopo deste trabalho, mas podem ser encontrados na referência abaixo.  
Neste projeto há um dashboard pré-configurado chamado **FIAP MLE Fase 4**

[Suporte do Grafana à fonte de dados do Prometheus](https://prometheus.io/docs/visualization/grafana/)

### Visualizando o dashboard

![Dashboard Granfana](readme_files/GrafanaDashboard.png)

### Validando os dados do dashboard de acordo com o que foi registrado no log

Abaixo podemos fazer a correspondência entre o valor registrado no dashboard do Grafana com o valor calculado e registrado nos logs para o tempo consumido em uma requisição de treinamento.

![Tempo de treinamento no Grafana](readme_files/GrafanaTrainTime.png)

![Tempo de treinamento registrado nos logs](readme_files/LogTrainTime.png)

## Acessando os logs da aplicação

Informações relevantes são registradas nos logs correspondentes de cada módulo da aplicação.

![Pasta de logs](readme_files/LogDirectory.png)

Abaixo é exibido um exemplo de como estão estruturados os logs gerados.

![Exemplo do arquivo de logs](readme_files/LogExample.png)

# Testando o ambiente produtivo através do frontend

Acesse a URL: http://localhost:8501

Selecione o modelo (pré-treinado) da caixa de seleção.

Escolha quais ações devem executar a previsão e clique em **PREDICT**

Serão exibidos os gráficos com as previsões correspondentes a cada ação selecionada.

![Resultado do FrontEnd](readme_files/FrontEndResult.png)