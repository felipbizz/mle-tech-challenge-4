# syntax=docker/dockerfile:1

############################################################################
### Arquivo: Dockerfile                                                  ###
### Descrição: Conjunto de instruções para a construção da imagem da API ###
### Projeto: Tech Challenge - FIAP Machine Learning Engineering (Fase 4) ###
############################################################################

FROM python:3.11-slim

# Install uv (gerenciador de pacotes, similar ao pip ou poetry).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Cópia dos arquivos da aplicação.
COPY . /app

# Instalação de dependências.
WORKDIR /app

# Foi necessário recriar o lock file.
# Como sugestão, investigar como realizar a instalação de pacotes com um arquivo de lock existente.
RUN uv lock

RUN uv sync --frozen --no-cache

# Injeta as credenciais do logfire através de um secret.
# Talvez possa ser melhorado para criar uma variável de ambiente (LOGFIRE_TOKEN) no lugar do arquivo.
# FIca como sugestão de melhoria de segurança.

RUN mkdir -p /app/.logfire/

RUN --mount=type=secret,id=logfire,target=/root/logfire_credentials.json \
cp /root/logfire_credentials.json /app/.logfire/logfire_credentials.json


# Inicializa a API utilizando o FastAPI.
# Como sugestão de melhoria, pensar em implementar a customização da porta através de variável de ambiente.
CMD ["/app/.venv/bin/fastapi", "run", "api/app.py", "--port", "8000", "--host", "0.0.0.0"]

# A instrução abaixo pode ser usada para troubleshooting do container.
# É útil quando o container é finalizado antes mesmo de permitir o acesso ao shell através do docker exec.
# CMD [ "bash" ]