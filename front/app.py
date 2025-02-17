import streamlit as st
import matplotlib.pyplot as plt
import requests
import json

st.title("Previsão de Ações utilizando LSTM")

st.info("Selecione as ações e aperte o botão 'PREDICT'")

st.sidebar.header("Configurações")

url_list_models = "http://mle-api:8000/api/v1/model/list"
lista_de_modelos = json.loads(requests.request("GET", url_list_models).text)

# Lista de tipos de energia
stock_options = [
    "VALE3.SA",
    "PETR4.SA",
    "ITUB4.SA",
    "ABEV3.SA",
    "BBDC4.SA",
    "SANB11.SA",
    "BBAS3.SA",
    "JBSS3.SA",
    "KLBN11.SA",
    "BPAC11.SA",
    "BBDC3.SA",
    "ITSA4.SA",
    "WEGE3.SA",
]

modelo_selecionado = st.sidebar.selectbox("Tipo de Modelo", lista_de_modelos)

tipos_stock = st.sidebar.multiselect(
    "Selecione os tipos de energia para visualizar:", stock_options, default=[]
)

if st.sidebar.button("PREDICT"):
    for stock in tipos_stock:
        st.subheader(f"Previsão para {stock.capitalize()} ")

        url = f"http://mle-api:8000/api/v1/model/predict?stock_option={stock}"

        payload = json.dumps(modelo_selecionado)
        headers = {
            "Content-Type": "application/json",
            "Content-Type": "application/json",
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        path_imagem = json.loads(response.text)["message"]["message"].split("/")[1]
        array_img = plt.imread(f"reports/{path_imagem}")
        st.image(array_img)
