from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import git
from lime import lime_tabular

app = Flask(__name__)

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained ML model using the absolute path
MODEL_PATH = os.path.join(current_dir, 'model.joblib')
model = joblib.load(MODEL_PATH)

# Train lime explainer
X = os.path.join(current_dir, 'lime_data.csv')
cat_features = [i for i, c in enumerate(X.columns) if X.dtypes[i]!='float64']
class_names = ['accordé', 'refusé']

explainer = lime_tabular.LimeTabularExplainer(X_smpl_tr, mode="classification",
                                              class_names=class_names,
                                              feature_names=X.columns,
                                              categorical_features=cat_features)
                                          

# Automatic git pull via endpoint
@app.route('/git_update', methods=['POST'])
def git_update():
    repo = git.Repo('/home/OCJulienClaveau/OC_Projet_7')
    origin = repo.remotes.origin
    repo.create_head('main',
    origin.refs.main).set_tracking_branch(origin.refs.main).checkout()
    origin.pull()
    return '', 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get the data in JSON format and transform it in pd.DataFrame
        data_json = request.get_json()
        data = pd.DataFrame(data_json['dataframe_split']['data'], 
                            columns=data_json['dataframe_split']['columns'], 
                            index=data_json['dataframe_split']['index'])
        
        # Make predictions using the loaded model
        predictions = model.predict(data)
        
        # Make explanation with lime
        X_test_tr = model.named_steps['columntransformer'].transform(data)
        predict_fn = model.named_steps['estimator'].predict_proba

        explanation = explainer.explain_instance(X_test_tr[0], predict_fn,
                                                 num_features=5)

        explanation = explanation.as_html()
        
        return jsonify({'prediction' : predictions[0],
                        'explanation' : explanation})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()