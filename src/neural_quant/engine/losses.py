import torch
import torch.nn as nn

class VolatilityMatchLoss(nn.Module):
    """
    Fonction de perte spécialisée pour calibrer la volatilité stochastique.
    Objectif : Minimiser l'écart entre la Volatilité Générée par le SDE et la Volatilité Réalisée historique.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, predicted_vols: torch.Tensor, target_vols: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predicted_vols: [Batch, Time, 1] issue du SDE
            target_vols: [Batch, Time, 1] issue du DataProcessor (Realized Vol)
        """
        # 1. On s'assure que les tailles correspondent (parfois le SDE simule t+1)
        min_len = min(predicted_vols.shape[1], target_vols.shape[1])
        pred = predicted_vols[:, :min_len, :]
        targ = target_vols[:, :min_len, :]
        
        # 2. Calcul de l'erreur quadratique moyenne (MSE)
        # Si le modèle prédit 20% de vol et la réalité est 10%, il est pénalisé.
        loss = self.mse(pred, targ)
        
        return self.alpha * loss