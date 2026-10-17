from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Healthcare Triage API")

# 1. Load the trained model into memory when the server starts
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Define the expected data format
class PatientSymptoms(BaseModel):
    age: int
    heart_rate: int
    blood_pressure_systolic: int

# 3. The Health Check for Zero-Downtime Deployment
@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# 4. The Real Prediction Endpoint
@app.post("/predict")
def predict_triage(symptoms: PatientSymptoms):
    # Convert the incoming JSON into a format the model understands
    input_data = np.array([[
        symptoms.age, 
        symptoms.heart_rate, 
        symptoms.blood_pressure_systolic
    ]])
    
    # Ask the model for a prediction
    prediction = model.predict(input_data)[0]
    
    severity = "Critical" if prediction == 1 else "Stable"
    
    return {
        "symptoms_received": symptoms,
        "predicted_severity": severity
    }