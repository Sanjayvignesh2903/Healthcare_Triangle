from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Healthcare Triage API")

# Configure CORS to allow your frontend to communicate with this backend
origins = [
    "*"  # Use "*" for testing, replace with your exact GitHub Pages URL later for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

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