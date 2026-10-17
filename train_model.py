import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Generate 1,000 synthetic patient records
np.random.seed(42)
n_samples = 1000

data = {
    "age": np.random.randint(18, 90, n_samples),
    "heart_rate": np.random.randint(60, 160, n_samples),
    "blood_pressure_systolic": np.random.randint(80, 200, n_samples)
}
df = pd.DataFrame(data)

# Medical logic: High heart rate + high BP = Critical (1), else Stable (0)
df['severity'] = np.where((df['heart_rate'] > 120) & (df['blood_pressure_systolic'] > 150), 1, 0)

# 2. Train the Machine Learning Model
X = df[['age', 'heart_rate', 'blood_pressure_systolic']]
y = df['severity']

print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Save the trained model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
print("Success! model.pkl has been created.")