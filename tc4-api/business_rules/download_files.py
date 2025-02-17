from tqdm import tqdm
import yfinance as yf
from deltalake.writer import write_deltalake
from src.utils import setLog

logger = setLog("download_files", level=10)


def download(symbols: list) -> dict:
    # Registra o resultado do download do símbolos para ser usado como retorno da função
    download_result: dict = {"successful_downloads": [], "failed_downloads": []}

    logger.info(
        "---------------------------------------------------------------------------------------------------"
    )
    logger.info(f"Iniciando a coleta de dados para os símbolos: {symbols}")

    for symbol in tqdm(symbols):
        logger.info(f"Processando {symbol}")

        try:
            df = yf.download(symbol)
        except Exception as e:
            logger.info(f"Não foi possível processar {symbol} por conta de {e}")
            download_result["failed_downloads"].append(symbol)
            continue

        logger.info("Ajustando dados para gravação no DeltaLake")
        df.columns = df.columns.droplevel(1)
        df = df.reset_index()

        df = df.rename(columns={"Date": "ds", "Close": "y"})
        df = df[["ds", "y"]]

        logger.info(f"Tamanho dos dados organizados de {symbol}: {len(df)}")

        if len(df) == 0:
            download_result["failed_downloads"].append(symbol)
            logger.warn(f"Não foram capturados dados para {symbol}")
            continue

        df["unique_id"] = symbol

        write_deltalake("deltalake", df, mode="append", partition_by=["unique_id"])
        logger.info(f"Dados para {symbol} gravados com sucesso no DeltaLake")

        download_result["successful_downloads"].append(symbol)
        logger.info(f"Dados de {symbol} carregados com sucessso.")

    return download_result


# if __name__ == '__main__':

#     # symbols : list = ['DIS', 'VALE3','PETR4','ITUB4','ABEV3','BBDC4','SANB11','BBAS3','JBSS3','KLBN11', 'BPAC11', 'BBDC3', 'ITSA4', 'WEGE3']
#     symbols: list = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'ABEV3.SA', 'BBDC4.SA', 'SANB11.SA', 'BBAS3.SA', 'JBSS3.SA', 'KLBN11.SA', 'BPAC11.SA', 'BBDC3.SA', 'ITSA4.SA', 'WEGE3.SA']
#     download_files(symbols)
