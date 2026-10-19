from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Healthcare Triage API")

# Configure CORS to allow your frontend to communicate with this backend
origins = [
    "*" 
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
def predict_triage(data: PatientVitals):
    # Pass all 8 vital signs directly to the new model in the exact trained order
    input_features = np.array([[
        data.age, 
        data.heart_rate, 
        data.blood_pressure_systolic,
        data.respiratory_rate,
        data.spo2,
        data.temperature,
        data.supplemental_oxygen,
        data.consciousness_alert
    ]])
    
    # Run the prediction
    prediction = model.predict(input_features)[0]
    
    # Format the response
    triage_status = "Critical" if prediction == 1 else "Stable"
    
    return {
        "prediction": int(prediction),
        "triage_level": triage_status
    }