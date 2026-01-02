import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(r'A:\project_x\stroke_prediction\data\healthcare-dataset-stroke-data.csv')
    
    # Check for missing values across the dataset
    print("Missing values in each column:\n", data.isnull().sum())
    
    # Fill missing BMI values with the mean and round to 2 decimal places
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)
    data['bmi'] = data['bmi'].round(2)
    
    # Convert categorical features to numerical (Label Encoding)
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    
    # Dictionary to store label encoders for each categorical column
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save the label encoder for future use
    
    # Create 'model/' directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save label encoders to a pickle file
    with open('model/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    print(type(label_encoders))  # Should print: <class 'dict'>
    print(label_encoders.keys())  # Should print the keys: ['gender', 'ever_married', etc.]
    print(label_encoders['gender'].classes_)
    print(label_encoders['ever_married'].classes_)
    print(label_encoders['work_type'].classes_)
    print(label_encoders['Residence_type'].classes_)
    print(label_encoders['smoking_status'].classes_)

    
    # Features and target variable
    X = data.drop(['id', 'stroke'], axis=1)  # Remove 'id' and target 'stroke'
    y = data['stroke']
    
    # Normalize the numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save the scaler for future use
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = r'A:\project_x\stroke_prediction\data\healthcare-dataset-stroke-data.csv'
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    
    print("Training data shape: ", X_train.shape)
    print("Test data shape: ", X_test.shape)
    
