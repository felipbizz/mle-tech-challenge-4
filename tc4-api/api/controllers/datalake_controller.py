from fastapi import APIRouter, Body
from business_rules import download_files
from typing import Annotated
from src.utils import setLog

logger = setLog('datalake_controller', level=10)

router = APIRouter(prefix='/api/v1/datalake', tags=['Endpoints do Datalake'])


# Avaliar os retornos das funções e considerar o retorno de erro como um HTTPException a ser tratado no frontend.
@router.post('/download')
def download(symbols : Annotated[list[str] | None, Body()] = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'ABEV3.SA', 'BBDC4.SA', 'SANB11.SA', 'BBAS3.SA', 'JBSS3.SA', 'KLBN11.SA', 'BPAC11.SA', 'BBDC3.SA', 'ITSA4.SA', 'WEGE3.SA']): 
    '''    
    Carrega os símbolos usados na bolsa a partir da biblioteca Yahoo!Finance e os grava em um DataLake local.

    Parameters:

        symbols (list[str]): Lista de símbolos a serem baixados e carregados no DataLake.

    Returns:

        dict : Dicionário contento duas listas, uma de cargas com sucesso e outra com as cargas com falhas.
    '''

    logger.info('---------------------------------------------------------------------------------------------------')
    logger.info(f'Realizando o download dos símbolos: {symbols}')

    if(len(symbols) == 0):
        logger.error('Não foi definido nenhum símbolo para download.')
        return { 'Erro': 'Defina pelo menos um símbolo a ser carregado no DeltaLake' }
    else:
        download_result = download_files.download(symbols)
        logger.debug(f'Download concluído.\nResultado: {download_result}')

    return { 'Símbolos carregados com sucesso' : download_result['successful_downloads'],
             'Símbolos com erro na carga' : download_result['failed_downloads'] }