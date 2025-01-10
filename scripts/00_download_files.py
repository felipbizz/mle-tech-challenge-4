from tqdm import tqdm
import yfinance as yf
from deltalake.writer import write_deltalake
from src.utils import get_path_project

PROJECT_DIR = get_path_project()
DATALAKE_PATH = PROJECT_DIR / "data/raw/yfinance_api"
SYMBOLS = [
    "DIS",
    "VALE3",
    "PETR4",
    "ITUB4",
    "ABEV3",
    "BBDC4",
    "SANB11",
    "BBAS3",
    "JBSS3",
    "KLBN11",
    "BPAC11",
    "BBDC3",
    "ITSA4",
    "WEGE3",
]


def download_files(symbols: list) -> None:
    for symbol in tqdm(symbols):
        print(f"Processando {symbol}")
        try:
            if symbol == "DIS":
                df = yf.download(f"{symbol}")
            else:
                df = yf.download(f"{symbol}.SA")
        except Exception as e:
            print(f"Não foi possível processar {symbol} por conta de {e}")
            continue
        df.columns = df.columns.droplevel(1)
        df = df.reset_index()

        df = df.rename(
            columns={
                "Date": "ds",
                "Close": "y",
            }
        )  # "Price": "DIS"
        df = df[["ds", "y"]]

        df["unique_id"] = symbol
        write_deltalake(DATALAKE_PATH, df, mode="append", partition_by=["unique_id"])

    return None


def main() -> None:
    download_files(SYMBOLS)
    return None


if __name__ == "__main__":
    main()
