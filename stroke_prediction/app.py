from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved label encoders, scaler, and trained model
with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the LSTM model
model = load_model('model/stroke_prediction_lstm_model.h5')

def predict_stroke_risk(new_data):
    # Prepare the features in the correct order
    features = []
    try:
        features.append(label_encoders['gender'].transform([new_data['gender']])[0])
        features.append(new_data['age'])
        features.append(new_data['hypertension'])
        features.append(new_data['heart_disease'])
        features.append(label_encoders['ever_married'].transform([new_data['ever_married']])[0])
        features.append(label_encoders['work_type'].transform([new_data['work_type']])[0])
        features.append(label_encoders['Residence_type'].transform([new_data['Residence_type']])[0])
        features.append(new_data['avg_glucose_level'])
        features.append(new_data['bmi'])
        features.append(label_encoders['smoking_status'].transform([new_data['smoking_status']])[0])
    except KeyError as e:
        raise KeyError(f"Label encoder for '{e.args[0]}' not found in the loaded encoders.")
    
    # Convert features to numpy array and scale
    features_array = np.array(features).reshape(1, -1)  # shape (1, 10)
    features_scaled = scaler.transform(features_array)
    
    # Reshape to add the time step dimension for LSTM (shape should be (1, 1, 10))
    features_scaled = np.expand_dims(features_scaled, axis=1)  # shape (1, 1, 10)
    
    # Make prediction using the model
    prediction = model.predict(features_scaled)
    
    # Return prediction probability
    return prediction[0][0]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        gender = request.form['gender']
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        Residence_type = request.form['Residence_type']
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = request.form['smoking_status']

        # Prepare input data
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        # Make prediction
        probability = predict_stroke_risk(input_data)

        # Determine risk level and redirect to appropriate page
        if probability >= 0.7:
            return redirect(url_for('stroke', severity="High Risk", probability=probability))
        elif 0.4 <= probability < 0.7:
            return redirect(url_for('stroke', severity="Medium Risk", probability=probability))
        else:
            return redirect(url_for('nostroke', probability=probability))

@app.route('/stroke')
def stroke():
    severity = request.args.get('severity')
    probability = request.args.get('probability')
    return render_template('stroke.html', severity=severity, probability=probability)

@app.route('/nostroke')
def nostroke():
    probability = request.args.get('probability')
    return render_template('nostroke.html', probability=probability)

if __name__ == '__main__':
    app.run(debug=True)














