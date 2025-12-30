from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI(title="ML Cancer Prediction API - Both Models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load BOTH models
dt_model = joblib.load("models/decision_tree_model.joblib")
lr_model = joblib.load("models/logistic_regression_model.joblib")
dt_features = joblib.load("models/dt_feature_names.joblib")
lr_features = joblib.load("models/lr_feature_names.joblib")

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/")
async def root():
    return {
        "message": "ML Cancer Prediction API",
        "models": ["Decision Tree", "Logistic Regression"],
        "expected_features": len(dt_features)
    }

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        if len(request.features) != len(dt_features):
            raise HTTPException(status_code=400, detail=f"Expected {len(dt_features)} features")
        
        features = np.array(request.features).reshape(1, -1)
        
        # Decision Tree
        dt_pred = dt_model.predict(features)[0]
        dt_proba = dt_model.predict_proba(features)[0]
        
        # Logistic Regression
        lr_pred = lr_model.predict(features)[0]
        lr_proba = lr_model.predict_proba(features)[0]
        
        return {
            "decision_tree": {
                "prediction": "Malignant" if dt_pred == 0 else "Benign",
                "malignant_prob": float(dt_proba[0]),
                "benign_prob": float(dt_proba[1]),
                "confidence": float(max(dt_proba))
            },
            "logistic_regression": {
                "prediction": "Malignant" if lr_pred == 0 else "Benign",
                "malignant_prob": float(lr_proba[0]),
                "benign_prob": float(lr_proba[1]),
                "confidence": float(max(lr_proba))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
