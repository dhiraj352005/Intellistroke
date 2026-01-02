import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Data Preprocessing
def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Fill missing BMI values with the mean
    df['bmi'].fillna(df['bmi'].mean(), inplace=True)

    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Select features and target variable
    X = df.drop(columns=['id', 'stroke'])
    y = df['stroke']

    # Scale the features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape input for LSTM [samples, time steps, features]
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    return X_scaled, y, scaler, label_encoders

# Create LSTM Model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=(1, input_shape)))  # Assuming 1 timestep
    model.add(LSTM(units=64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))  # Output layer with sigmoid for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Load and preprocess data
file_path = r'A:\project_x\stroke_prediction\data\healthcare-dataset-stroke-data.csv'
X, y, scaler, label_encoders = preprocess_data(file_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get the input shape for the model (number of features)
input_shape = X_train.shape[2]

# Create the LSTM model
model = create_lstm_model(input_shape)

# Define EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model
model_path = 'model/stroke_prediction_lstm_model.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# Save the scaler and label encoders
try:
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(label_encoders, 'model/label_encoders.pkl')
    print("Scaler and label encoders saved successfully.")
except Exception as e:
    print(f"Error saving scaler or label encoders: {e}")
