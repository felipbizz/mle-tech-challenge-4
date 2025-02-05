import pandas as pd
from deltalake import DeltaTable
from src.utils import WMAPE, wmape
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoLSTM
import datetime
import joblib
from src.utils import setLog

logger = setLog('tune_model', level=10)

def tuna_modelo_autolstm():

    # Carregando dados armazenados no DeltaLake
    df = DeltaTable('deltalake').to_pandas()
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    # # Separando dados de treinamento e testes
    # train = df.loc[df['ds'] < '2024-09-01']
    # test = df.loc[(df['ds'] >= '2024-09-01') & (df['ds'] < '2024-12-20')]

    h = test['ds'].nunique()

    models = [AutoLSTM(h=h, 
                    num_samples=30, 
                    loss=WMAPE())]

    model = NeuralForecast(models=models, freq='D')

    initial_config = model.config
    logger.debug(f'Salvando configuração inicial: {initial_config}')
    
    model.fit(train, val_size=30)
    best_hp = models[0].results.get_best_result().metrics['config']
    logger.info(f'Melhores parâmetros encontrados:\n {best_hp}')
    
    trained_config = model.config
    logger.debug(f'Configuração após o treino: {trained_config}')
    
    joblib.dump(model, f"neuralforecast_autolstm_{datetime.datetime.now().date()}.joblib")

    predictions = model.predict().reset_index()
    predictions = predictions.merge(test[['ds','unique_id', 'y']], on=['ds', 'unique_id'], how='left')
    
    print(f'A avaliação do wmape é:', wmape(predictions['y'], predictions['AutoLSTM']))

    fig, ax = plt.subplots(2, 1, figsize = (1280/96, 720/96))
    fig.tight_layout(pad=7.0)
    for ax_i, unique_id in enumerate(['ABEV3', 'BBAS3']):
        plot_df = pd.concat([train.loc[train['unique_id'] == unique_id].tail(30), 
                            predictions.loc[predictions['unique_id'] == unique_id]]).set_index('ds')
        plot_df[['y', 'AutoLSTM']].plot(ax=ax[ax_i], linewidth=2, title=unique_id)
        
        ax[ax_i].grid()
    fig.savefig(f'reports/neuralforecast_autolstm_{datetime.datetime.now().date()}.png', dpi=300)

if __name__ == '__main__':
    tuna_modelo_autolstm()