import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class NeuralHawkes(nn.Module):
    """
    Processus de Hawkes Neural Conditionné.
    
    Modélise l'intensité des sauts lambda(t) comme :
    lambda(t) = mu(Z_t) + sum_{t_i < t} alpha(Z_t) * exp(-beta * (t - t_i))
    
    Où Z_t est le vecteur de contexte (sorties du LLM / Market features).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, decay_beta: float = 1.0):
        super().__init__()
        
        self.decay_beta = decay_beta  # Vitesse d'oubli du marché
        
        # Le réseau qui traduit le contexte (LLM) en paramètres de Hawkes
        # Il prédit mu (baseline intensity) et alpha (jump excitement)
        self.context_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(), # Activation bornée pour stabilité
            nn.Linear(hidden_dim, 2) # Sortie: [mu, alpha]
        )
        
        # Initialisation soignée (Biais positif pour éviter lambda=0 au début)
        self.context_net[-1].bias.data.fill_(0.5)

    def get_params(self, context_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transforme le contexte en paramètres positifs mu et alpha.
        Utilisation de Softplus pour garantir la positivité (contrainte mathématique forte).
        """
        out = self.context_net(context_embedding)
        mu, alpha = out.chunk(2, dim=-1)
        
        # Softplus(x) = log(1 + exp(x)) -> Toujours positif
        mu = F.softplus(mu) + 1e-6       # Baseline intensity
        alpha = F.softplus(alpha) * 0.9  # Branching ratio (< 1 pour stationnarité stricte, mais peut dépasser localement)
        
        return mu, alpha

    def forward(self, 
                context_embedding: torch.Tensor, 
                dt: float, 
                past_jumps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcul l'intensité lambda(t) pour le pas de temps actuel.
        
        Args:
            context_embedding: [Batch, Dim] (Signal LLM)
            dt: pas de temps depuis la dernière update
            past_jumps: [Batch, History_Len] (1 si saut, 0 sinon) - Simplification pour grille discrète
        
        Returns:
            intensity: [Batch, 1]
        """
        mu, alpha = self.get_params(context_embedding)
        
        # Composante auto-excitante (Hawkes)
        # Approximation discrète pour le trading haute fréquence / daily
        hawkes_component = torch.zeros_like(mu)
        
        if past_jumps is not None and past_jumps.sum() > 0:
            # Calcul vectorisé de la somme exponentielle pondérée
            # (Dans une version optimisée C++, on utiliserait un état récurrent récursif)
            batch_size, seq_len = past_jumps.shape
            
            # Création grille temporelle relative
            time_grid = torch.arange(seq_len, device=past_jumps.device).float() * dt
            decay_kernel = torch.exp(-self.decay_beta * time_grid) # [Seq_Len]
            
            # Convolution temporelle : Sauts * Kernel
            # On inverse decay_kernel pour que t=0 soit le saut le plus récent
            decay_kernel = decay_kernel.flip(0)
            
            # Produit scalaire par batch
            hawkes_integral = (past_jumps * decay_kernel).sum(dim=1, keepdim=True)
            hawkes_component = alpha * hawkes_integral

        lambda_t = mu + hawkes_component
        return lambda_t

    def compute_log_likelihood(self, 
                             event_times: torch.Tensor, 
                             context_embeddings: torch.Tensor, 
                             t_max: float) -> torch.Tensor:
        """
        Fonction de perte pour entraîner le modèle sur des données historiques.
        Log-Likelihood = sum(log(lambda(t_i))) - int_0^T lambda(u) du
        
        Note: C'est la version simplifiée. Pour le niveau PhD, on utilisera
        l'intégrale exacte du processus de Hawkes.
        """
        # Placeholder pour le Sprint 3 (Calibration)
        pass