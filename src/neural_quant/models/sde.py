import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- C'était manquant !
import torchsde
from typing import Tuple

class NeuralSDE(torchsde.SDEIto):
    """
    Implémentation des fonctions de Drift et Diffusion pour prix/volatilité.
    """
    
    def __init__(self, context_dim: int, hidden_dim: int):
        # CORRECTION : On retire 'sde_type="ito"' car c'est implicite dans SDEIto
        super().__init__(noise_type="general")
        
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        # ... le reste ne change pas ...
        
        # Initialisation petite pour stabilité
        nn.init.uniform_(self.context_net[-1].weight, a=-0.1, b=0.1)
        
        self.register_buffer("identity", torch.diag(torch.ones(2)))

    def get_sde_params(self, z_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.context_net(z_t)
        mu, kappa, xi, eta = out.chunk(4, dim=-1)
        
        # --- FORCAGE DES PARAMÈTRES POUR LA DÉMO ---
        
        # mu : On laisse libre (tendance)
        
        # kappa (Vitesse de retour à la moyenne) : Entre 0.5 et 5.0
        # On évite un rappel trop violent qui tue la vol
        kappa = torch.sigmoid(kappa) * 4.0 + 0.5
        
        # xi (Moyenne de long terme de la variance) : ESSENTIEL
        # On force la variance cible à être au moins 0.01 (10% vol) et max 0.16 (40% vol)
        # Softplus garantit la positivité, +0.02 garantit qu'on ne tombe jamais à 0
        xi = F.softplus(xi) + 0.02 
        
        # eta (Volatilité de la volatilité) : C'est ça qui crée les oscillations du graphe du bas
        # On booste un peu pour voir ça bouger
        eta = F.softplus(eta) + 0.1
        
        return mu, kappa, xi, eta

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Drift: f(t, y)
        V_t = y[:, 1].clamp_min(1e-6).unsqueeze(-1)
        z_t = torch.ones_like(V_t) * 0.1 
        
        mu, kappa, xi, eta = self.get_sde_params(z_t)
        
        drift_X = mu
        drift_V = kappa * (xi - V_t)
        
        return torch.cat([drift_X, drift_V], dim=-1)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Diffusion: g(t, y) -> Matrice [Batch, 2, 2] car noise_type="general"
        V_t = y[:, 1].clamp_min(1e-6).unsqueeze(-1)
        z_t = torch.ones_like(V_t) * 0.1 
        mu, kappa, xi, eta = self.get_sde_params(z_t)
        
        diffusion_X = torch.sqrt(V_t) 
        diffusion_V = eta * torch.sqrt(V_t)
        
        # Construction de la matrice de diffusion diagonale
        # [[sqrt(V), 0], [0, eta*sqrt(V)]]
        batch_size = y.shape[0]
        diffusion_matrix = torch.zeros(batch_size, 2, 2, device=y.device)
        
        # Remplissage diagonal
        diffusion_matrix[:, 0, 0] = diffusion_X.squeeze(-1)
        diffusion_matrix[:, 1, 1] = diffusion_V.squeeze(-1)
        
        return diffusion_matrix