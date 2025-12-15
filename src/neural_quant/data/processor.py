import pandas as pd
import numpy as np
import torch
from typing import Tuple

class DataProcessor:
    """
    Responsable du nettoyage et du Feature Engineering (Transform).
    Prépare les tenseurs PyTorch.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.stats = {} # Pour stocker moyenne/std (pour dé-normaliser plus tard)

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute Log-Returns et Volatilité Réalisée."""
        df = df.copy()
        
        # 1. Sélection de la colonne Prix (Adj Close ou Close)
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        
        # 2. Log Returns : r_t = ln(S_t / S_{t-1})
        df['Log_Returns'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # 3. Target Volatility (Rolling Std Dev annualisée)
        # C'est ce que le Neural SDE devra apprendre à reproduire
        df['Realized_Vol'] = df['Log_Returns'].rolling(window=self.window_size).std() * np.sqrt(252)
        
        # Nettoyage des NaN créés par le shift et le rolling
        df.dropna(inplace=True)
        return df

    def to_tensors(self, df: pd.DataFrame, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convertit le DataFrame en Tensors PyTorch normalisés.
        Returns: (Returns_Tensor, Vol_Tensor) au format [Batch, Time, 1]
        """
        if 'Log_Returns' not in df.columns:
            df = self.add_features(df)
            
        returns = df['Log_Returns'].values.astype(np.float32)
        vols = df['Realized_Vol'].values.astype(np.float32)
        
        # Normalisation (Standard Scaling)
        # Très important pour que le réseau de neurones converge vite
        ret_mean, ret_std = returns.mean(), returns.std()
        vol_mean, vol_std = vols.mean(), vols.std()
        
        self.stats = {
            'ret_mean': ret_mean, 'ret_std': ret_std,
            'vol_mean': vol_mean, 'vol_std': vol_std
        }
        
        # On normalise les inputs
        returns_norm = (returns - ret_mean) / (ret_std + 1e-8)
        
        # On garde la vol brute comme target souvent, ou normalisée. 
        # Pour simplifier ici, on normalise tout.
        vols_norm = (vols - vol_mean) / (vol_std + 1e-8)
        
        # Reshape pour PyTorch : [Batch=1, Time, Dim=1]
        # On considère toute la série comme un seul batch ici pour l'entraînement séquentiel
        t_returns = torch.tensor(returns_norm, device=device).view(1, -1, 1)
        t_vols = torch.tensor(vols_norm, device=device).view(1, -1, 1)
        
        return t_returns, t_vols