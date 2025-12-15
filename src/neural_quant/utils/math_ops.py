import torch
import torch.nn.functional as F

def softplus_stable(x: torch.Tensor, min_val: float = 1e-6, scale: float = 1.0) -> torch.Tensor:
    """
    Une version sécurisée de Softplus pour garantir la positivité (Volatilité).
    x: Input tensor
    min_val: Valeur minimale absolue (pour éviter la division par zéro)
    scale: Facteur d'échelle pour l'activation
    """
    return scale * F.softplus(x) + min_val

def build_grid(T: float, steps: int, device: str = 'cpu') -> torch.Tensor:
    """
    Génère une grille temporelle standardisée.
    """
    return torch.linspace(0, T, steps, device=device)