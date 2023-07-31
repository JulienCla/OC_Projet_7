import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title('Dashboard - Scoring crédit')

DATA_URL = 'final_features.csv'
MODEL_URL = 'http://127.0.0.1:5001/invocations'
MODEL_URL_FLASK = 'flask_app'

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
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
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

                 
def main():

    id_client = st.number_input('Id Client', value=0)
    data = get_data_client(id_client)
    
    st.dataframe(data)
    
    predict_btn = st.button('Prédire')

    if predict_btn:
        response = request_prediction(MODEL_URL_FLASK + 'predict', data)
        
        st.write(response['prediction'])

        if int(response['prediction']) == '0' :
            st.write('Accordé')
        else :
            st.write('refusé')
      
                 
if __name__ == '__main__':
    main()