import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import os

st.title('Dashboard - Scoring crédit')

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURR_DIR, "data_streamlit.csv")
# MODEL_URL_MLFLOW = 'http://127.0.0.1:5001/invocations'
MODEL_URL_FLASK = 'https://ocjulienclaveau.eu.pythonanywhere.com/predict'

username = os.environ.get('PA_USERNAME')
password = os.environ.get('PA_PASSWORD')

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    st.write(data.shape)
    return data

def get_data_client(id_client):
    data = load_data()
    data_client = data.loc[data['SK_ID_CURR'] == id_client, :]
    data_client = data_client.replace(np.nan, None)
    st.write(data_client.shape)
    return data_client

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'dataframe_split' : data.to_dict(orient='split')}
    
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json, auth=(username, password))

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

                 
def main():
    # Récupération des données clients via SK_ID_CURR
    id_client = st.number_input('Id Client', value=0)
    data = get_data_client(id_client)
    
    # Filtre pour choisir quelles colonnes afficher 
    # de base les 10 variables les plus pertinentes sont séléctionnées
    selected_columns = st.multiselect('Choisissez les colonnes à afficher:',
                                      data.columns,
                                      default=data.iloc[:, 0:10])
    
    # Dataframe dynamique pour permettre à l'utilisateur de demander une
    # prédiction avec des changements sur ses infos
    edited_data = st.data_editor(data[selected_columns])
    
    col1, col2 = st.columns(2)
    
    # Bouton pour requete vers flask API servant le modèle
    predict_btn = col1.button('Obtenir Prédiction')
    if predict_btn:
        response = request_prediction(MODEL_URL_FLASK, data)
    
    # Affichage du résultat de la prédiction
    if int(response['prediction']) == 0 :
        col2.write('Accordé', 'green')
    else :
        col2.write('Refusé', 'red')
    
    # Affichage explication de la prédiction (avec LIME)
    components.html(response['explanation'], width=1000, height=200)
      
                 
if __name__ == '__main__':
    main()