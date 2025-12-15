import yfinance as yf
import pandas as pd
import os
from pathlib import Path

class LMEDataLoader:
    """
    Responsable du téléchargement et du cache des données brutes (Extract).
    """
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self, ticker: str, start_date: str = "2020-01-01", end_date: str = None) -> pd.DataFrame:
        """
        Télécharge les données depuis Yahoo Finance ou les charge depuis le disque si déjà présentes aujourd'hui.
        """
        file_path = self.data_dir / f"{ticker}_{start_date}.parquet"
        
        # Optionnel : Cache simple pour éviter de spammer l'API
        if file_path.exists():
            print(f"📦 [Data] Chargement depuis le cache : {file_path}")
            return pd.read_parquet(file_path)
            
        print(f"⬇️ [Data] Téléchargement {ticker} depuis Yahoo Finance...")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Gestion multi-index (spécifique aux nouvelles versions de yfinance)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, level=1, axis=1)
            except KeyError:
                pass # Parfois le format diffère selon la version

        # Sauvegarde pour la prochaine fois
        df.to_parquet(file_path)
        return df