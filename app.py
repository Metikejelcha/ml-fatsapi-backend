from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import uvicorn

app = FastAPI(title="ML Cancer Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

dt_model = None
lr_model = None
dt_features = None  # list of names from joblib

class PredictionRequest(BaseModel):
    features: list[float]

def load_models():
    global dt_model, lr_model, dt_features
    dt_model = joblib.load("decision_tree_model.joblib")
    lr_model = joblib.load("logistic_regression_model.joblib")
    dt_features = joblib.load("dt_feature_names.joblib")

@app.on_event("startup")
async def startup_event():
    load_models()

@app.get("/")
async def root():
    return {
        "message": "ML Cancer Prediction API",
        "models": ["Decision Tree", "Logistic Regression"],
        "status": "ok" if dt_model and lr_model else "error"
    }

@app.post("/predict")
async def predict(req: PredictionRequest):
    if dt_model is None or lr_model is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    if len(req.features) != len(dt_features):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(dt_features)} features, got {len(req.features)}"
        )

    x = np.array(req.features).reshape(1, -1)

    dt_pred = dt_model.predict(x)[0]
    dt_proba = dt_model.predict_proba(x)[0]

    lr_pred = lr_model.predict(x)[0]
    lr_proba = lr_model.predict_proba(x)[0]

    return {
        "decision_tree": {
            "prediction": "Malignant" if dt_pred == 0 else "Benign",
            "confidence": float(max(dt_proba)),
            "malignant_prob": float(dt_proba[0]),
            "benign_prob": float(dt_proba[1])
        },
        "logistic_regression": {
            "prediction": "Malignant" if lr_pred == 0 else "Benign",
            "confidence": float(max(lr_proba)),
            "malignant_prob": float(lr_proba[0]),
            "benign_prob": float(lr_proba[1])
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
