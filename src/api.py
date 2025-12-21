from typing import List, Optional, Dict, Tuple
from pathlib import Path
import time
import threading
import math
from datetime import date, datetime
import os

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError

from dotenv import load_dotenv

# ============================
#   CONSTANTES E DIRETÓRIOS
# ============================

# Carrega as variáveis do arquivo .env para o ambiente
load_dotenv()

WINDOW_SIZE = 60
DATA_DIR = Path("../data")
MODELS_DIR = Path("../models")

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

HF_REPO_ID = os.getenv("HF_REPO_ID", "seu-usuario/nome-do-repo")
HF_TOKEN = os.getenv("HF_TOKEN") # O Token será injetado pelo Render

app = FastAPI(title="LSTM Stock Predictor API - Multi-Assets")


# ============================
#   MODELOS Pydantic
# ============================

class DownloadRequest(BaseModel):
    start_date: Optional[str] = "2018-01-01"
    end_date: Optional[str] = None


class TrainResponse(BaseModel):
    symbol: str
    started: bool
    status_url: str


class PredictRequest(BaseModel):
    symbol: str
    last_n_days: int = WINDOW_SIZE
    n_days: int = 1


class PredictionResponse(BaseModel):
    symbol: str
    predicted_prices: List[float]


class SymbolStatusResponse(BaseModel):
    symbol: str
    has_data: bool
    has_model: bool
    ready_for_prediction: bool
    data_path: Optional[str]
    model_path: Optional[str]
    scaler_path: Optional[str]
    data_last_modified: Optional[str]
    model_last_modified: Optional[str]

    training_status: Optional[str]
    training_started_at: Optional[str]
    training_finished_at: Optional[str]
    training_metrics: Optional[dict]
    training_error: Optional[str]


# ============================
#   AUXILIARES
# ============================

def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper()


def get_data_path(symbol: str) -> Path:
    return DATA_DIR / f"{normalize_symbol(symbol)}_history.csv"


def get_model_paths(symbol: str) -> Tuple[Path, Path]:
    sym = normalize_symbol(symbol)
    return (
        MODELS_DIR / f"{sym}_lstm_model.h5",
        MODELS_DIR / f"{sym}_scaler.pkl"
    )


def create_sequences(series: np.ndarray, window_size: int):
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i, 0])
        y.append(series[i, 0])
    return np.array(X), np.array(y)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def upload_model_to_hf(symbol: str):
    """Sobe o modelo (.h5) e o scaler (.pkl) para o Hugging Face"""
    if not HF_TOKEN:
        print("HF_TOKEN não configurado. Upload pulado.")
        return

    sym = normalize_symbol(symbol)
    model_path, scaler_path = get_model_paths(sym)

    api = HfApi(token=HF_TOKEN)

    # Upload do Modelo
    if model_path.exists():
        print(f"Subindo modelo de {sym} para o Hugging Face...")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=f"models/{sym}_lstm_model.h5",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )

    # Upload do Scaler
    if scaler_path.exists():
        print(f"Subindo scaler de {sym} para o Hugging Face...")
        api.upload_file(
            path_or_fileobj=scaler_path,
            path_in_repo=f"models/{sym}_scaler.pkl",
            repo_id=HF_REPO_ID,
            repo_type="model"
        )


def download_model_from_hf(symbol: str):
    """Tenta baixar o modelo e o scaler do Hugging Face se não existirem localmente"""
    sym = normalize_symbol(symbol)
    model_path, scaler_path = get_model_paths(sym)

    # Se já temos os arquivos, não faz nada
    if model_path.exists() and scaler_path.exists():
        return True

    print(f"Arquivos locais de {sym} ausentes. Tentando baixar do Hugging Face...")

    try:
        # Baixa Modelo
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/{sym}_lstm_model.h5",
            local_dir=BASE_DIR,  # Salva na estrutura de pastas correta
            token=HF_TOKEN
        )
        # Baixa Scaler
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"models/{sym}_scaler.pkl",
            local_dir=BASE_DIR,
            token=HF_TOKEN
        )
        return True
    except (RepositoryNotFoundError, RevisionNotFoundError, Exception) as e:
        print(f"Não foi possível baixar modelo de {sym} do HF: {e}")
        return False

# ============================
#   CACHE DE MODELOS
# ============================

model_cache: Dict[str, Tuple[tf.keras.Model, MinMaxScaler]] = {}
model_cache_lock = threading.Lock()


def load_model_for_symbol(symbol: str):
    sym = normalize_symbol(symbol)
    with model_cache_lock:
        if sym in model_cache:
            return model_cache[sym]

        # Tenta garantir que os arquivos existam (Local ou Baixando do HF)
        download_model_from_hf(sym)

        model_path, scaler_path = get_model_paths(sym)

        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"Modelo ou scaler não encontrado para {sym} (nem local, nem no HF)")

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        model_cache[sym] = (model, scaler)
        return model, scaler


# ============================
#   JOBS DE TREINO
# ============================

training_jobs_lock = threading.Lock()
training_jobs: Dict[str, Dict] = {}


def _train_symbol_model_task(sym: str):
    """
    Função de treino executada em background.
    """
    global training_jobs
    data_path = get_data_path(sym)

    try:
        df = pd.read_csv(data_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df.sort_values("Date", inplace=True)

        price_col = "Close" if "Close" in df.columns else "Adj Close"

        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df.dropna(subset=[price_col], inplace=True)

        if len(df) <= WINDOW_SIZE + 1:
            raise RuntimeError(f"Dados insuficientes para treinar {sym}")

        close_prices = df[[price_col]].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_prices)

        X, y = create_sequences(scaled, WINDOW_SIZE)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        total_samples = X.shape[0]
        train_len = int(total_samples * 0.7)
        val_len = int(total_samples * 0.15)

        X_train = X[:train_len]
        y_train = y[:train_len]
        X_val = X[train_len:train_len + val_len]
        y_val = y[train_len:train_len + val_len]
        X_test = X[train_len + val_len:]
        y_test = y[train_len + val_len:]

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )

        y_pred_scaled = model.predict(X_test)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_inv = scaler.inverse_transform(y_pred_scaled).ravel()

        metrics = {
            "mae": float(mean_absolute_error(y_test_inv, y_pred_inv)),
            "rmse": float(math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))),
            "mape": float(mape(y_test_inv, y_pred_inv))
        }

        model_path, scaler_path = get_model_paths(sym)
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

        try:
            upload_model_to_hf(sym)
        except Exception as e:
            print(f"Erro ao subir para o Hugging Face: {e}")

        with model_cache_lock:
            if sym in model_cache:
                del model_cache[sym]

        with training_jobs_lock:
            training_jobs[sym]["status"] = "success"
            training_jobs[sym]["finished_at"] = datetime.utcnow().isoformat()
            training_jobs[sym]["metrics"] = metrics
            training_jobs[sym]["error"] = None

    except Exception as e:
        with training_jobs_lock:
            training_jobs[sym]["status"] = "error"
            training_jobs[sym]["finished_at"] = datetime.utcnow().isoformat()
            training_jobs[sym]["error"] = str(e)


# ============================
#   MIDDLEWARE DE MONITORAMENTO
# ============================

metrics_lock = threading.Lock()
metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "avg_response_time_ms": 0.0,
    "max_response_time_ms": 0.0,
}

try:
    import psutil
except ImportError:
    psutil = None


@app.middleware("http")
async def add_monitoring(request: Request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)
        success = response.status_code < 500
    except Exception:
        success = False
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        with metrics_lock:
            metrics["total_requests"] += 1
            n = metrics["total_requests"]
            avg = metrics["avg_response_time_ms"]
            metrics["avg_response_time_ms"] = avg + (elapsed_ms - avg) / n
            metrics["max_response_time_ms"] = max(metrics["max_response_time_ms"], elapsed_ms)
            if not success:
                metrics["total_errors"] += 1

    return response


# ============================
#   ENDPOINTS
# ============================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics-summary")
def metrics_summary():
    data = metrics.copy()
    if psutil:
        p = psutil.Process()
        data["cpu_percent"] = psutil.cpu_percent(interval=0.0)
        data["memory_rss_mb"] = p.memory_info().rss / (1024 * 1024)
    return data


@app.post("/stocks/{symbol}/download")
def download_data(symbol: str, req: DownloadRequest):
    sym = normalize_symbol(symbol)

    start = req.start_date or "2018-01-01"
    end = req.end_date or date.today().isoformat()

    df = yf.download(sym, start=start, end=end)
    if df.empty:
        raise HTTPException(404, f"Nenhum dado encontrado para {sym}.")

    df.reset_index(inplace=True)
    path = get_data_path(sym)
    df.to_csv(path, index=False)

    return {
        "symbol": sym,
        "rows": len(df),
        "data_path": str(path.resolve())
    }


@app.get("/stocks/{symbol}/status", response_model=SymbolStatusResponse)
def stock_status(symbol: str):
    sym = normalize_symbol(symbol)

    data_path = get_data_path(sym)
    model_path, scaler_path = get_model_paths(sym)

    has_data = data_path.exists()
    has_model = model_path.exists() and scaler_path.exists()
    ready = has_data and has_model

    with training_jobs_lock:
        job = training_jobs.get(sym, {})

    return SymbolStatusResponse(
        symbol=sym,
        has_data=has_data,
        has_model=has_model,
        ready_for_prediction=ready,
        data_path=str(data_path.resolve()) if has_data else None,
        model_path=str(model_path.resolve()) if has_model else None,
        scaler_path=str(scaler_path.resolve()) if has_model else None,
        data_last_modified=datetime.fromtimestamp(data_path.stat().st_mtime).isoformat() if has_data else None,
        model_last_modified=datetime.fromtimestamp(model_path.stat().st_mtime).isoformat() if has_model else None,

        training_status=job.get("status", "idle"),
        training_started_at=job.get("started_at"),
        training_finished_at=job.get("finished_at"),
        training_metrics=job.get("metrics"),
        training_error=job.get("error"),
    )


@app.post("/stocks/{symbol}/train", response_model=TrainResponse)
def train_model(symbol: str, request: Request, background_tasks: BackgroundTasks):
    sym = normalize_symbol(symbol)
    data_path = get_data_path(sym)

    if not data_path.exists():
        raise HTTPException(400, f"Dados não encontrados para {sym}. Use /stocks/{sym}/download primeiro.")

    with training_jobs_lock:
        job = training_jobs.get(sym)
        if job and job.get("status") == "running":
            raise HTTPException(409, f"Treino já em execução para {sym}.")

        training_jobs[sym] = {
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "metrics": None,
            "error": None,
        }

    background_tasks.add_task(_train_symbol_model_task, sym)

    return TrainResponse(
        symbol=sym,
        started=True,
        status_url=str(request.base_url) + f"stocks/{sym}/status"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest, request: Request):
    sym = normalize_symbol(req.symbol)

    data_path = get_data_path(sym)
    if not data_path.exists():
        raise HTTPException(404, f"Sem dados para {sym}. Faça o download primeiro.")

    model_path, scaler_path = get_model_paths(sym)
    if not model_path.exists():
        raise HTTPException(
            404,
            {
                "message": f"Não existe modelo treinado para {sym}.",
                "docs": str(request.base_url) + "docs"
            }
        )

    df = pd.read_csv(data_path)
    price_col = "Close" if "Close" in df.columns else "Adj Close"
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df.dropna(subset=[price_col], inplace=True)

    if req.last_n_days < WINDOW_SIZE:
        raise HTTPException(400, f"last_n_days deve ser >= {WINDOW_SIZE}")

    if len(df) < req.last_n_days:
        subset = df
    else:
        subset = df.tail(req.last_n_days)

    if len(subset) < WINDOW_SIZE:
        raise HTTPException(400, f"Amostra insuficiente. Necessário pelo menos {WINDOW_SIZE} registros.")

    last_prices = subset[price_col].values[-WINDOW_SIZE:]

    model, scaler = load_model_for_symbol(sym)

    last_window_scaled = scaler.transform(last_prices.reshape(-1, 1))
    window = last_window_scaled.copy()
    preds_scaled = []

    for _ in range(req.n_days):
        x = window.reshape(1, WINDOW_SIZE, 1)
        pred_scaled = model.predict(x, verbose=0)
        preds_scaled.append(pred_scaled[0, 0])
        window = np.vstack([window[1:], pred_scaled])

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel().tolist()

    return PredictionResponse(symbol=sym, predicted_prices=preds)
