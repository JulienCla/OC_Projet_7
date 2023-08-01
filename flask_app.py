from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load pre-trained ML model during app initialization
MODEL_PATH = 'LR_with_threshold.joblib'
model = joblib.load(MODEL_PATH)

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