import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("results/model_primary.pkl")
scaler = joblib.load("results/scaler.pkl")

# 👇 Sample new data (1 row of 41 features - just for demo)
# Replace these numbers with actual sensor values or real-time input
new_data = np.array([[0.5]*41])  # dummy row with all 0.5s

# Scale the input
new_data_scaled = scaler.transform(new_data)

# Predict
prediction = model.predict(new_data_scaled)

# Display results
print("\n🎯 Predicted Outputs (Stage 1 - ~.C.Actual):")
for i, value in enumerate(prediction[0]):
    print(f"  Output {i+1}: {value:.2f}")
