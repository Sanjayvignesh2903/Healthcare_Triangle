import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Generate 1,000 synthetic patient records with expanded clinical vitals
np.random.seed(42)
n_samples = 1000

data = {
    "age": np.random.randint(18, 90, n_samples),
    "heart_rate": np.random.randint(40, 150, n_samples),
    "blood_pressure_systolic": np.random.randint(80, 200, n_samples),
    "respiratory_rate": np.random.randint(10, 35, n_samples),
    "spo2": np.random.randint(85, 100, n_samples),
    "temperature": np.random.uniform(35.0, 40.0, n_samples),
    "supplemental_oxygen": np.random.randint(0, 2, n_samples), # 0 = No, 1 = Yes
    "consciousness_alert": np.random.randint(0, 2, n_samples)  # 1 = Alert, 0 = Not Alert
}
df = pd.DataFrame(data)

# Medical logic based loosely on clinical deterioration thresholds
# Critical (1) if: SpO2 < 92 OR heart rate > 130 OR respiratory rate > 25 OR not alert
df['severity'] = np.where(
    (df['spo2'] < 92) | 
    (df['heart_rate'] > 130) | 
    (df['respiratory_rate'] > 25) | 
    (df['consciousness_alert'] == 0), 1, 0
)

# 2. Train the Machine Learning Model
feature_cols = [
    'age', 'heart_rate', 'blood_pressure_systolic', 
    'respiratory_rate', 'spo2', 'temperature', 
    'supplemental_oxygen', 'consciousness_alert'
]
X = df[feature_cols]
y = df['severity']

print("Training expanded Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Success! Expanded model.pkl has been created.")