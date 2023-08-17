import unittest
import pandas as pd
import sys
from unittest.mock import MagicMock
sys.modules['git'] = MagicMock()
from flask_app import app, get_or_create_explainer


class Testflaskapp(unittest.TestCase):
    
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        
    
    def test_request_predict(self):
        with app.app_context():
            explainer = get_or_create_explainer()     
        
            headers = {"Content-Type": "application/json"}
            data_test = pd.read_csv('data_test.csv')
            data_json = {'dataframe_split' : data_test.to_dict(orient='split')}
            response = self.client.post('/predict', headers=headers, json=data_json)
            response_data = response.get_json()

            self.assertEqual(response.status_code, 200, 'Erreur lors de la requete : {}'.format(response.status_code))
            print(response_data)
            self.assertIn(response_data['prediction'], [0, 1], 'Incorrect model output !')
            self.assertEqual(response_data['explanation'], "Mock explanation")
        
if __name__ == '__main__':
    unittest.main()