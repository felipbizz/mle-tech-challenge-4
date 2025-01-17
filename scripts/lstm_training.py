# Bibliotecas
import deltalake
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Carregando os dados
df = deltalake.DeltaTable(table_uri="data/raw/yfinance_api").to_pandas()
df.value_counts(subset=["unique_id"])

# Pegando apenas as datas onde todos possuem valor
df_v1 = df.copy()
date_filter = df_v1.groupby("ds")["unique_id"].count() == 14
dates_to_consider = []
for date, to_consider in date_filter.items():
    if to_consider:
        dates_to_consider.append(date)
df_v1 = df_v1[df_v1["ds"].isin(dates_to_consider)].reset_index(drop=True)

# Transformando cada ação em uma coluna com os valores 'y' e o índice será a data
df_v2 = pd.pivot_table(
    data=df_v1, values="y", columns="unique_id", index="ds"
).sort_index()

# Normalizando os dados usando o MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(X=df_v2.values)

# Criando as sequências (in e target)
num_features_in = df_v2.shape[1]
num_features_out = num_features_in
input_sequence_length = 7
output_sequence_length = 1
total_sequence_lenght = input_sequence_length + output_sequence_length

x = []
y = []

for i in range(data.shape[0] - total_sequence_lenght):
    x.append(data[i : i + input_sequence_length])
    y.append(data[i + input_sequence_length : i + total_sequence_lenght])

x = np.array(x)
y = np.array(y)

# Dividindo em dados de treino e de teste
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Transformando em tensores
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Criando os datasets (TensorDatasets)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Criando os  DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
