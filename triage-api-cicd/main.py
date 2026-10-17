from fastapi import FastAPI

app = FastAPI(title="Healthcare Triage API")

@app.get("/health")
def health_check():
    """
    The load balancer will hit this endpoint. 
    If it returns 200 OK, the new container is marked 'Healthy' 
    and traffic is switched over.
    """
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/")
def read_root():
    return {"message": "Triage API is running"}