#!/bin/bash

# Sai imediatamente se um comando retornar um status diferente de zero
set -e

# Passo 1: Sincronizar arquivos com o comando 'uv'
echo "Sincronizando arquivos com o grupo 'neuralnetwork'..."
uv sync --group=neuralnetwork

# Passo 2: Ativar o ambiente virtual
echo "Ativando o ambiente virtual..."
source .venv/bin/activate

# Passo 3: Executar o script para baixar arquivos
echo "Executando 00_download_files.py..."
python scripts/00_download_files.py

# Passo 4: Executar o script de preparação dos dados
echo "Executando 01_data_preparation.py..."
python scripts/01_data_preparation.py

# Passo 5: Executar o script de criação do modelo
echo "Executando 02_model_creation.py..."
python scripts/02_model_creation.py

echo "Pipeline concluído com sucesso!"

