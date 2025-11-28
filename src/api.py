from typing import List
from pathlib import Path
import time
import threading

import numpy as np
import joblib

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from tensorflow.keras.models import load_model

# Opcional: tentar importar psutil para métricas de CPU/Memória
try:
    import psutil
except ImportError:
    psutil = None

WINDOW_SIZE = 60
MODELS_DIR = "../models"

model_path = Path(MODELS_DIR) / "lstm_model.h5"
scaler_path = Path(MODELS_DIR) / "scaler.pkl"

if not model_path.exists():
    raise RuntimeError("Modelo não encontrado. Rode o train_lstm.py primeiro.")

if not scaler_path.exists():
    raise RuntimeError("Scaler não encontrado. Rode o train_lstm.py primeiro.")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

app = FastAPI(title="LSTM Stock Price Predictor API")

# ============================
#   MODELOS DE REQUEST/RESPONSE
# ============================

class PredictionRequest(BaseModel):
    prices: List[float]
    n_days: int = 1


class PredictionResponse(BaseModel):
    predicted_prices: List[float]


# ============================
#   ESTADO DE MÉTRICAS (IN-MEMORY)
# ============================

metrics_lock = threading.Lock()
metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "avg_response_time_ms": 0.0,
    "max_response_time_ms": 0.0,
}


# ============================
#   MIDDLEWARE DE MONITORAMENTO
# ============================

@app.middleware("http")
async def add_monitoring(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        success = response.status_code < 500
    except Exception:
        success = False
        with metrics_lock:
            metrics["total_errors"] += 1
        # re-levanta a exceção para FastAPI tratar
        raise
    finally:
        end = time.perf_counter()
        elapsed_ms = (end - start) * 1000.0

        with metrics_lock:
            metrics["total_requests"] += 1
            # média móvel simples
            n = metrics["total_requests"]
            old_avg = metrics["avg_response_time_ms"]
            metrics["avg_response_time_ms"] = old_avg + (elapsed_ms - old_avg) / n
            if elapsed_ms > metrics["max_response_time_ms"]:
                metrics["max_response_time_ms"] = elapsed_ms

            if not success:
                metrics["total_errors"] += 1

    return response


# ============================
#   ENDPOINTS
# ============================

@app.get("/health")
def health_check():
    """
    Endpoint simples de health-check para ver se a API está de pé.
    """
    return {"status": "ok"}


@app.get("/metrics-summary")
def metrics_summary():
    """
    Endpoint de monitoramento simples com:
    - total de requests
    - total de erros
    - tempo médio de resposta
    - tempo máximo de resposta
    - uso de CPU/Memória (se psutil estiver instalado)
    """
    with metrics_lock:
        data = metrics.copy()

    # adiciona métricas de recurso, se possível
    if psutil is not None:
        process = psutil.Process()
        cpu_percent = psutil.cpu_percent(interval=0.0)
        mem_info = process.memory_info()
        data.update(
            {
                "cpu_percent": cpu_percent,
                "memory_rss_mb": mem_info.rss / (1024 * 1024),
            }
        )

    return data


@app.get("/")
def root():
    return {"message": "API LSTM de previsão de preços de ações online."}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if len(req.prices) < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Envie ao menos {WINDOW_SIZE} preços históricos.",
        )

    last_window = np.array(req.prices[-WINDOW_SIZE:], dtype=float).reshape(-1, 1)
    last_window_scaled = scaler.transform(last_window)

    window = last_window_scaled.copy()
    preds_scaled = []

    for _ in range(req.n_days):
        x_input = window.reshape(1, WINDOW_SIZE, 1)
        pred_scaled = model.predict(x_input, verbose=0)
        preds_scaled.append(pred_scaled[0, 0])
        window = np.vstack([window[1:], pred_scaled])

    preds = (
        scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1))
        .ravel()
        .tolist()
    )

    return PredictionResponse(predicted_prices=preds)
