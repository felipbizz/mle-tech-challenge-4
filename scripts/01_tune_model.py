import pandas as pd
from deltalake import DeltaTable
from src.utils import WMAPE, wmape
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.auto import AutoLSTM
import datetime
import joblib

def tuna_modelo_autolstm():
    df = DeltaTable('deltalake').to_pandas()
    df = df.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    TEMPO_FILTRO_PARA_TESTE = '2018-01-01'

    df = df.loc[df['ds'] > TEMPO_FILTRO_PARA_TESTE]

    FILTRA_IDS = df['unique_id'].unique()[:2]

    df = df[df['unique_id'].isin(FILTRA_IDS)]

    train = df.loc[df['ds'] < '2024-09-01']
    valid = df.loc[(df['ds'] >= '2024-09-01') & (df['ds'] < '2024-12-20')]
    h = valid['ds'].nunique()

    models = [AutoLSTM(h=h, 
                    num_samples=30, 
                    loss=WMAPE())]

    model = NeuralForecast(models=models, freq='D')

    model.fit(train, val_size=30)
    
    joblib.dump(model, f"neuralforecast_lstm_{datetime.datetime.now().date()}.joblib")

    p = model.predict().reset_index()
    p = p.merge(valid[['ds','unique_id', 'y']], on=['ds', 'unique_id'], how='left')
    
    print(f'A avaliação do wmape é:', wmape(p['y'], p['AutoLSTM']))

    fig, ax = plt.subplots(2, 1, figsize = (1280/96, 720/96))
    fig.tight_layout(pad=7.0)
    for ax_i, unique_id in enumerate(['ABEV3', 'BBAS3']):
        plot_df = pd.concat([train.loc[train['unique_id'] == unique_id].tail(30), 
                            p.loc[p['unique_id'] == unique_id]]).set_index('ds')
        plot_df[['y', 'AutoLSTM']].plot(ax=ax[ax_i], linewidth=2, title=unique_id)
        
        ax[ax_i].grid()
    fig.savefig(f'reports/neuralforecast_lstm_{datetime.datetime.now().date()}.png', dpi=300)

if __name__ == '__main__':
    tuna_modelo_autolstm()