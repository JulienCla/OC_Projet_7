import unittest
import pandas as pd
import sys
import joblib
from unittest.mock import MagicMock
sys.modules['git'] = MagicMock()
from flask_app import app

# Création d'une fonction personnalisée pour effectuer des prédictions en fonction d'un seuil
class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input):
        # Prédiction personnalisée avec le paramètre threshold
        probabilities = self.model.predict_proba(model_input)
        predictions = (probabilities[:, 1] >= self.threshold).astype(int)
        return predictions
    
    def predict_proba(self, model_input, context=None):
        return self.model.predict_proba(model_input)
    
    def model(self, context=None):
        return self.model


class Testflaskapp(unittest.TestCase):
    
    def setUp(self):
        app.config['TESTING'] = True
        self.app = app
        self.client = self.app.test_client()
        
    def test_loading_model(self):
        data_test = pd.read_csv('data_test.csv')
        data_test.drop('Unnamed: 0', axis=1, inplace=True)
        
        model = joblib.load('model.joblib')
        pred = model.predict(data_test)
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
            self.assertIn(response_data, [0, 1], 'Incorrect response output ! : {}'.format(response_data))
            
    
if __name__ == '__main__':
    unittest.main()