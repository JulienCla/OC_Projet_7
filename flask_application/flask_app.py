from flask import Flask, request, jsonify, g
import pandas as pd
import numpy as np
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
DATA_PATH = os.path.join(current_dir, 'lime_data.csv')

# Function to create and return the explainer instance
def get_explainer():
    training_data = pd.read_csv(DATA_PATH)
    
    nb_num_feats = 104
    nb_total_feats = training_data.shape[1]
    cat_features = list(range(nb_num_feats, nb_total_feats))
    class_names = ['accordé', 'refusé']
    feat_names = training_data.columns
    training_data = training_data.to_numpy()
    
    explainer = lime_tabular.LimeTabularExplainer(
        training_data,
        mode="classification",
        class_names=class_names,
        feature_names=feat_names,
        categorical_features=cat_features
    )   
    return explainer
explainer = get_explainer()
# # Get or create the explainer instance within the application context
# def get_or_create_explainer():
#     if 'explainer' not in g:
#         g.explainer = get_explainer()
#     return g.explainer

# with app.app_context():
#     get_or_create_explainer()

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
        X_test_tr = model.named_steps['imputer_scaler'].transform(data)
        predict_fn = model.named_steps['estimator'].predict_proba

        explanation = explainer.explain_instance(X_test_tr[0], predict_fn,
                                                 num_features=5)

        explanation = explanation.as_html()

        response = {
            'prediction': int(predictions[0]), 
            'explanation': explanation
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()