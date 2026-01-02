import os
import numpy as np
from model import preprocess_data, create_lstm_model  # Import functions from model.py
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Ensure reproducibility
np.random.seed(42)

# Load and preprocess the data
file_path = r'A:\project_x\stroke_prediction\data\healthcare-dataset-stroke-data.csv'
X, y, scaler, label_encoders = preprocess_data(file_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get the input shape for the LSTM model (number of features)
input_shape = X_train.shape[2]

# Create the LSTM model
model = create_lstm_model(input_shape)

# Define EarlyStopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
print("Starting model training...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, 
                    callbacks=[early_stopping], verbose=1)  # Verbose option added

# Evaluate the model on the test data
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Save the trained model to a file
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save(os.path.join(model_dir, 'stroke_prediction_lstm_model.h5'))  # Adjusted to avoid repetitive 'model'
print(f"Model saved at: {os.path.join(model_dir, 'stroke_prediction_lstm_model.h5')}")

# Optionally, save the training history if needed (accuracy, loss over epochs)
np.save(os.path.join(model_dir, 'training_history.npy'), history.history)
print("Training history saved.")
