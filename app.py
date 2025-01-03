from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

with open('models/model.pkl', 'rb') as f:
    model: RandomForestClassifier = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    try:
        area = float(data['area'])
        oxygen = float(data['oxygen'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
    except ValueError:
        return jsonify({'error': 'Invalid input. Please provide valid numeric values.'})

    input_df = pd.DataFrame([[area, oxygen, temperature, humidity]],
                            columns=['Area', 'Oxygen', 'Temperature', 'Humidity'])

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0]

    result = {
        'prediction': 'Fire Occurrence' if prediction == 1 else 'No Fire',
        'fire_percentage': round(prediction_proba[1] * 100, 2),  
    }

    return jsonify(result)


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    data = pd.read_csv(file)
    if not set(['Area', 'Oxygen', 'Temperature', 'Humidity']).issubset(data.columns):
        return jsonify({'error': 'Invalid file format. Ensure required columns are present.'})

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]

    data['Prediction'] = ['Fire Occurrence' if pred == 1 else 'No Fire' for pred in predictions]
    data['Confidence (%)'] = (probabilities * 100).round(2)

    return data.to_json(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
