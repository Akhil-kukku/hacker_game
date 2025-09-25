"""
Simple test API to verify basic functionality
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Cybersecurity Engine Test API")

@app.get("/")
def read_root():
    return {"message": "Cybersecurity Engine Test API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is working"}

if __name__ == "__main__":
    print("Starting test API server...")
    print("Access at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000)
