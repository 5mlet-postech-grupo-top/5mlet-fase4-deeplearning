import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Caminho do CSV (mesmo usado no download_data.py)
DATA_PATH = "../data/aapl_history.csv"
MODELS_DIR = "../models"

WINDOW_SIZE = 60  # dias usados para predizer o próximo


def create_sequences(series: np.ndarray, window_size: int):
    """
    Transforma a série em pares (janela, próximo valor).
    series deve ter shape (N, 1)
    """
    X, y = [], []
    for i in range(window_size, len(series)):
        X.append(series[i - window_size:i, 0])
        y.append(series[i, 0])
    return np.array(X), np.array(y)


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def main():
    data_path = Path(DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"Arquivo de dados não encontrado em: {data_path.resolve()}")

    print(f"Lendo dados de: {data_path.resolve()}")
    df = pd.read_csv(data_path)

    # Mostra um preview pra debug
    print("Colunas encontradas:", list(df.columns))
    print("Primeiras linhas do CSV:")
    print(df.head())

    # Garante coluna de data se existir
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)

    # Escolhe qual coluna de preço usar
    price_col = None
    if "Close" in df.columns:
        price_col = "Close"
    elif "Adj Close" in df.columns:
        price_col = "Adj Close"

    if price_col is None:
        raise RuntimeError(
            "Nenhuma coluna de preço encontrada ('Close' ou 'Adj Close'). "
            "Confira o formato do arquivo CSV."
        )

    # Converte para numérico e remove qualquer lixo (ex.: 'AAPL' virando NaN)
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])

    if len(df) <= WINDOW_SIZE + 1:
        raise RuntimeError(
            f"Dados insuficientes após limpeza. Linhas: {len(df)} (precisa de > {WINDOW_SIZE + 1})."
        )

    close_prices = df[[price_col]].values  # shape (N, 1)

    # Escalonamento
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    # Criação das sequências
    X, y = create_sequences(scaled, WINDOW_SIZE)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

    total_samples = X.shape[0]
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    print(f"Total de amostras: {total_samples}")
    print(f"Treino: {X_train.shape[0]} | Validação: {X_val.shape[0]} | Teste: {X_test.shape[0]}")

    # Modelo LSTM
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    print("Iniciando treinamento...")
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    print("Treinamento finalizado. Avaliando no conjunto de teste...")

    # Predição no teste
    y_pred_scaled = model.predict(X_test)

    # Desfaz o scaling
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled).ravel()

    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mape_value = mape(y_test_inv, y_pred_inv)

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape_value:.2f}%")

    # Salvar modelo e scaler
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "lstm_model.h5"
    scaler_path = models_dir / "scaler.pkl"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Modelo salvo em: {model_path.resolve()}")
    print(f"Scaler salvo em: {scaler_path.resolve()}")


if __name__ == "__main__":
    main()
