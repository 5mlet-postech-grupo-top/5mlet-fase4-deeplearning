import yfinance as yf
import pandas as pd
from pathlib import Path

def download_stock(symbol: str, start_date: str, end_date: str, out_path: str):
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)  # Date vira coluna
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Dados salvos em {out_path}, linhas: {len(df)}")

if __name__ == "__main__":
    symbol = "AAPL"  # troque aqui se quiser
    start_date = "2018-01-01"
    end_date = "2024-07-20"

    out_path = "../data/aapl_history.csv"
    download_stock(symbol, start_date, end_date, out_path)
