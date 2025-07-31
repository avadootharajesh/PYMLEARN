# Real_Time_Inference.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Define input data schema
class Features(BaseModel):
    features: list[float]

app = FastAPI()

# Load model and scaler
model, scaler = joblib.load('model.joblib')

@app.post("/predict")
async def predict(data: Features):
    try:
        x = np.array(data.features).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred = model.predict(x_scaled)[0]
        return {"prediction": int(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# send to api request
{
  "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
}