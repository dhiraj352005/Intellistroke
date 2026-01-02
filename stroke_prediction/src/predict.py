import pickle
import numpy as np
from tensorflow.keras.models import load_model  # Import for loading Keras model

# Load the saved label encoders and scaler
with open('model/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_stroke_risk(new_data, model):
    # Assuming new_data is a dictionary containing the input features
    
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
    probability = prediction[0][0]
    
    # Define thresholds for severity levels
    if probability < 0.4:
        risk_class = "Low Risk"
    elif 0.4 <= probability < 0.7:
        risk_class = "Medium Risk"
    else:
        risk_class = "High Risk"
    
    # Return both the probability and the severity level
    return probability, risk_class


# Example usage:
if __name__ == "__main__":
    # Sample input data
    new_data = {
        'gender': 'Male',
        'age': 67,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }
    
    # Load the LSTM model
    model = load_model('model/stroke_prediction_lstm_model.h5')
    
    # Get prediction
    probability, risk_class = predict_stroke_risk(new_data, model)
    
    # Print the results
    print(f"Stroke risk probability: {probability:.2f}")
    print(f"Stroke risk level: {risk_class}")
