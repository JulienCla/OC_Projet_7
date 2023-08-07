import unittest
import pandas as pd
from falsk_app import app


class Testflaskapp(unittest.TestCase):
    
    def SetUp(self):
        app.config['TESTING'] = True
        self.app = app.test_client()
        
    def test_request_predict(self):
        headers = {"Content-Type": "application/json"}
        data_test = pd.read_csv('data_test.csv')
        data_json = {'dataframe_split' : data_test.to_dict(orient='split')}
        rep = self.app.post('/predict', headers=headers, json=data_json)
        
        self.assertEqual(rep.status_code, 200, 'Erreur lors de la requete : {}'.format(rep.status_code))
        self.assertIn(rep['prediction'], [0, 1], 'Incorrect model output')
        
if __name__ == '__main__':
    unittest.main()