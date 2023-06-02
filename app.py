from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
from flask import Flask, render_template, request, send_from_directory
# Create Flask application
app = Flask(__name__)
temperature_data = pd.read_csv('static/temperature.csv')
# Load the Keras model
model = tf.keras.models.load_model('model.h5')
import csv
model1 = tf.keras.models.load_model('trans_model6.h5')
@app.route('/manual-prediction-trans', methods=['GET', 'POST'])
def manual_prediction_trans():
    if request.method == 'POST':
        inputs = [float(request.form[f'input{i+1}']) for i in range(5)]
        inputs = np.array(inputs).reshape(1, 5)
        prediction = model1.predict(inputs)
        result = prediction[0]
        return render_template('manualpredictiontrans.html', result=result)
    return render_template('manualpredictiontrans.html')

@app.route('/automatic-prediction-trans')
def automatic_prediction_trans():
    week_number = request.args.get('weekNumber')
    if week_number is None:
        return render_template('automaticpredictiontrans.html')

    week_number = int(week_number)
    prior_weeks = temperature_data[temperature_data['week'] < week_number].tail(5)['SST'].tolist()
    prediction = model1.predict([prior_weeks])[0][0]
    return render_template('automaticpredictiontrans.html', prediction=prediction)

@app.route('/automatic-prediction', methods=['GET'])
def automatic_prediction():
    week_number = request.args.get('weekNumber')
    if week_number is None:
        return render_template('automatic.html')

    week_number = int(week_number)
    prior_weeks = temperature_data[temperature_data['week'] < week_number].tail(5)['SST'].tolist()
    prediction = model.predict([prior_weeks])[0][0]
    return render_template('automatic.html', prediction=prediction)


# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the manual prediction page
@app.route('/manual-prediction')
def manual_prediction():
    return render_template('ManualPrediction.html')

# Define a route for the image visualization page
# ...
@app.route('/image-visualization', methods=['GET', 'POST'])
def image_visualization():
    if request.method == 'POST':
        week_number = int(request.form['weekNumber'])
        image_filename = f"Week{week_number}.png"
        image_path = os.path.join('static', 'images', image_filename)
        if os.path.isfile(image_path):
            time.sleep(3)
            return render_template('ImageVisualization.html', image_path=image_path,  week_number=week_number)
    return render_template('ImageVisualization.html')

# Define a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the request form
    data = [float(request.form[f'input{i}']) for i in range(1, 6)]
    
    # Perform prediction using the loaded model
    input_data = np.array([data])
    prediction = model.predict(input_data)

    # Postprocess the prediction (if needed)
    # ...

    # Render the prediction template with the prediction result
    return render_template('predict.html', prediction=prediction)
# Load temperature data from CSV file
def load_temperature_data():
    temperature_data = []
    with open('static/temperature.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            temperature_data.append(row)
    return temperature_data

# ... Existing code ...

@app.route('/table')
def table():
    temperature_data = load_temperature_data()
    return render_template('table.html', temperature_data=temperature_data)

if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
