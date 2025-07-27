import joblib
import pandas as pd

# Load the trained model
model = joblib.load('crop_model.pkl')

# Define input with proper column names
input_data = pd.DataFrame([{
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.87974371,
    'humidity': 82.00274423,
    'ph': 6.5,
    'rainfall': 202.9355362
}])

# Make prediction
prediction = model.predict(input_data)

print("Recommended Crop:", prediction[0])
