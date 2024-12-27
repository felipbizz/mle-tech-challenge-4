from tqdm import tqdm
import yfinance as yf
from deltalake.writer import write_deltalake


def download_files(symbols: list):
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

        df = df.rename(columns={"Date": "ds", "Close": "y", "Price": "DIS"})
        df = df[["ds", "y"]]

        df["unique_id"] = symbol
        write_deltalake("deltalake", df, mode="append", partition_by=["unique_id"])


if __name__ == "__main__":
    symbols = [
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

    download_files(symbols)
