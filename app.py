from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    try:
        area = float(data['area'])
    except ValueError:
        return jsonify({'error': 'Invalid input for area. Please enter a numeric value.'})

    try:
        oxygen = float(data['oxygen'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
    except ValueError:
        return jsonify({'error': 'Invalid input for one of the numeric fields. Please enter valid numbers.'})

    input_df = pd.DataFrame([[area, oxygen, temperature, humidity]],
                            columns=['Area', 'Oxygen', 'Temperature', 'Humidity'])

    prediction = model.predict(input_df)[0]
    result = 'Fire Occurrence' if prediction == 1 else 'No Fire'

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
