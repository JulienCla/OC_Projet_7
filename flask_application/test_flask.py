import unittest
import pandas as pd
import sys
import joblib
import mlflow
from custom_model import CustomModelWrapper
from unittest.mock import MagicMock
sys.modules['git'] = MagicMock()
from flask_app import app


class Testflaskapp(unittest.TestCase):
    
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app
        self.client = self.app.test_client()
        
    def test_loading_model(self):
        data_test = pd.read_csv('data_test.csv')
        data_test.drop('Unnamed: 0', axis=1, inplace=True)
        
        model = joblib.load('model.joblib')
        pred = model.predict(None, data_test)
        self.assertIn(pred, [0, 1], 'Incorrect model output ! : {}'.format(pred))
        
    def test_request_predict(self):
        with self.app.app_context():   
            headers = {"Content-Type": "application/json"}
            data_test = pd.read_csv('data_test.csv')
            data_test.drop('Unnamed: 0', axis=1, inplace=True)
            data_json = {'dataframe_split' : data_test.to_dict(orient='split')}
            response = self.client.post('/predict', headers=headers, json=data_json)
            response_data = response.get_json()

            self.assertEqual(response.status_code, 200, 'Erreur lors de la requete : {}'.format(response.status_code))
            self.assertIsNotNone(response_data, 'Erreur contenu reponse : {}'.format(response_data))
            self.assertIn(response_data['prediction'], [0, 1], 'Incorrect response output ! : {}'.format(response_data['prediction']))
            
    
if __name__ == '__main__':
    unittest.main()