from typing import List

import numpy as np
import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from tensorflow.keras.models import load_model

WINDOW_SIZE = 60
MODELS_DIR = "../models"

model_path = Path(MODELS_DIR) / "lstm_model.h5"
scaler_path = Path(MODELS_DIR) / "scaler.pkl"

if not model_path.exists() or not scaler_path.exists():
    raise RuntimeError("Modelo ou scaler não encontrados. Rode train_lstm.py primeiro.")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

app = FastAPI(title="LSTM Stock Price Predictor")

""" @app.on_event("startup")
async def _startup():
    Instrumentator().instrument(app).expose(app) """

class PredictionRequest(BaseModel):
    prices: List[float]  # históricos de fechamento
    n_days: int = 1      # quantos dias prever

class PredictionResponse(BaseModel):
    predicted_prices: List[float]

@app.get("/")
def read_root():
    return {"message": "API LSTM de previsão de preços de ações online."}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if len(req.prices) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"É necessário pelo menos {WINDOW_SIZE} preços históricos."
        )

    # usamos apenas os últimos WINDOW_SIZE
    last_window = np.array(req.prices[-WINDOW_SIZE:], dtype=float).reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window)

    # prevendo iterativamente
    window = last_window_scaled.copy()
    preds_scaled = []

    for _ in range(req.n_days):
        x_input = window.reshape((1, WINDOW_SIZE, 1))
        pred_scaled = model.predict(x_input, verbose=0)
        preds_scaled.append(pred_scaled[0, 0])

        # atualiza janela: remove 1º, adiciona pred
        window = np.vstack([window[1:], pred_scaled])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds = scaler.inverse_transform(preds_scaled).ravel().tolist()

    return PredictionResponse(predicted_prices=preds)
