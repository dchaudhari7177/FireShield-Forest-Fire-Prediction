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
        oxygen = float(data['oxygen'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
    except ValueError:
        return jsonify({'error': 'Invalid input. Please ensure all fields have numeric values.'})

    input_df = pd.DataFrame([[area, oxygen, temperature, humidity]],
                            columns=['Area', 'Oxygen', 'Temperature', 'Humidity'])

    prediction = model.predict(input_df)[0]

    result = 'Fire Occurrence' if prediction == 1 else 'No Fire'

    fire_occurrence_percentage = 70.0 

    return jsonify({
    'prediction': result,
    'fire_percentage': fire_occurrence_percentage  
})

if __name__ == '__main__':
    app.run(debug=True)
