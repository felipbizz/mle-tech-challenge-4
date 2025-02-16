FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app

WORKDIR /app
RUN uv lock

RUN uv sync --frozen --no-cache

CMD ["streamlit", "run", "front/app.py"]
