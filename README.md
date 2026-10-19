# 🏥 Medical Triage API (NEWS2 Protocol)

A containerized REST API built with FastAPI and Python that serves a Machine Learning inference engine for clinical triage.

## 🚀 Overview
This backend service processes real-time patient vital signs (Age, Heart Rate, Systolic BP, Respiratory Rate, SpO2, Temperature, Breathing Support, and Consciousness) and uses a trained **Random Forest Classifier** to instantly predict patient health risk levels based on the NEWS2 protocol. 

The model was trained on a comprehensive 100,000-record clinical dataset using Scikit-Learn to ensure high-accuracy threshold detection.

## 🛠️ Tech Stack
* **Framework:** FastAPI
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Server:** Uvicorn ASGI
* **Deployment:** Docker, Render
* **Validation:** Pydantic

## 🌐 Live Application
The frontend client for this API is built with HTML/CSS/JS and is hosted on GitHub Pages:
https://sanjayvignesh2903.github.io/healthcare-triage-ui/

## ⚙️ CI/CD & Architecture
* **Containerization:** The API is fully containerized using Docker for consistent cross-environment execution.
* **Continuous Deployment:** Integrated with Render for automated builds and deployments upon pushing to the main branch.
* **Security:** Strict CORS middleware is configured to securely accept asynchronous fetch requests exclusively from the designated GitHub Pages client.

## 🧠 Model Training
The inference model (`model.pkl`) was serialized after offline training. It analyzes 8 distinct clinical parameters simultaneously to catch edge cases of patient deterioration that simple rule-based systems might miss.
