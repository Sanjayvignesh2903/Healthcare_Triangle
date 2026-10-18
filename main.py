from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Healthcare Triage API")

# Load the newly trained expanded model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the required incoming data format (now matching NEWS2 parameters)
class PatientVitals(BaseModel):
    age: int
    heart_rate: int
    blood_pressure_systolic: int
    respiratory_rate: int
    spo2: float
    temperature: float
    supplemental_oxygen: int
    consciousness_alert: int

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/predict")
def predict_triage(vitals: PatientVitals):
    # Extract the features in the EXACT same order used during training
    features = np.array([[
        vitals.age,
        vitals.heart_rate,
        vitals.blood_pressure_systolic,
        vitals.respiratory_rate,
        vitals.spo2,
        vitals.temperature,
        vitals.supplemental_oxygen,
        vitals.consciousness_alert
    ]])
    
    prediction = model.predict(features)[0]
    status = "Critical" if prediction == 1 else "Stable"
    
    return {"prediction": int(prediction), "triage_level": status}