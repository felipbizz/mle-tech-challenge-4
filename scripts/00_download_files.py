# =============================================================================
# BIBLIOTECAS E MÓDULOS
# =============================================================================

import shutil
from pathlib import Path

import yfinance as yf
from common.utils import get_path_project
from deltalake.writer import write_deltalake
from tqdm import tqdm

# =============================================================================
# CONSTANTES
# =============================================================================

# Garantindo os paths corretos
PROJECT_DIR = get_path_project()
assert isinstance(PROJECT_DIR, Path)
DATALAKE_PATH = PROJECT_DIR / "data/raw/yfinance_api"

if DATALAKE_PATH.exists() and DATALAKE_PATH.is_dir():
    shutil.rmtree(DATALAKE_PATH)

# Ações a serem baixadas
SYMBOLS = [
    "DIS",
    "VALE3",
    "PETR4",
    "ITUB4",
    "ABEV3",
    "SANB11",
    "BBAS3",
    "JBSS3",
    "KLBN11",
    "BPAC11",
    "BBDC3",
    "ITSA4",
    "WEGE3",
]


# =============================================================================
# FUNÇÕES
# =============================================================================


def download_files(symbols: list) -> None:
    """
    Baixa dados de ações para uma lista de símbolos e salva em um Delta Lake.

    Args:
        symbols: Lista de símbolos de ações a serem processados.
                 Os símbolos podem ser para ações dos EUA ou do Brasil (sufixo .SA).
    """
    # Itera sobre cada símbolo na lista, exibindo uma barra de progresso.
    for symbol in tqdm(symbols):
        print(f"Processando {symbol}")
        try:
            # Baixa os dados da ação usando yfinance. Se o símbolo for "DIS", baixa diretamente,
            # caso contrário, adiciona o sufixo ".SA" para ações brasileiras.
            if symbol == "DIS":
                df = yf.download(f"{symbol}")
            else:
                df = yf.download(f"{symbol}.SA")
        except Exception as e:
            # Em caso de erro durante o download, exibe uma mensagem e continua para o próximo símbolo.
            print(f"Não foi possível processar {symbol} por conta de {e}")
            continue

        # Verifica se o DataFrame não é None.
        assert df is not None

        # Reduz o nível da coluna para facilitar o manuseio.
        df.columns = df.columns.droplevel(1)
        # Reseta o índice, transformando o índice atual em uma coluna.
        df = df.reset_index()

        # Renomeia as colunas para as convenções desejadas.
        df = df.rename(
            columns={
                "Date": "ds",  # Renomeia a coluna de data para 'ds'
                "Close": "y",  # Renomeia a coluna de fechamento para 'y'
            }
        )
        # Seleciona apenas as colunas 'ds' e 'y' para o DataFrame final.
        df = df[["ds", "y"]]

        # Adiciona uma coluna 'unique_id' para identificar os dados baixados por símbolo.
        df["unique_id"] = symbol

        # Salva o DataFrame no Delta Lake, particionando pelo 'unique_id'.
        write_deltalake(DATALAKE_PATH, df, mode="append", partition_by=["unique_id"])

    return None


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    download_files(SYMBOLS)
    return None


if __name__ == "__main__":
    main()
