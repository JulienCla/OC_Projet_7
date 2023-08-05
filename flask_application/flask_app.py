from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import git

app = Flask(__name__)

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained ML model using the absolute path
MODEL_PATH = os.path.join(current_dir, 'model.joblib')
model = joblib.load(MODEL_PATH)

# Automatic git pull
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
        data = pd.DataFrame(data_json['dataframe_split']['data'], columns=data_json['dataframe_split']['columns'], index=data_json['dataframe_split']['index'])
        
        # Make predictions using the loaded model
        predictions = model.predict(data)
        
        return jsonify({'prediction' : str(predictions[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()