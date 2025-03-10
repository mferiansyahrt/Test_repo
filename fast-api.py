from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("regression_model.pkl")

# Initialize FastAPI
app = FastAPI()

# Define request model
class InputData(BaseModel):
    feature: float

# Define API endpoint for prediction
@app.post("/predict")
def predict(data: InputData):
    # Convert input to NumPy array
    X_new = np.array([[data.feature]])
    
    # Make prediction
    prediction = model.predict(X_new)
    
    return {"prediction": prediction[0][0]}

# After that, try run this command below on terminal

# cd C:\Users\Asus\Documents\API_Test\
# uvicorn fast-api:app --host 0.0.0.0 --port 8000 --reload

