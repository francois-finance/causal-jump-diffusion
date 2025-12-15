import torch
import torch.nn as nn
from typing import Tuple

from neural_quant.models.sde import NeuralSDE
from neural_quant.models.jumps import NeuralHawkes

class HybridJumpDiffusion(nn.Module):
    """
    Le Modèle Complet : Neural Jump-Diffusion (NJD).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.sde = NeuralSDE(context_dim=input_dim, hidden_dim=hidden_dim)
        self.hawkes = NeuralHawkes(input_dim=input_dim, hidden_dim=hidden_dim)
        
    def forward_simulation(self, 
                           z_series: torch.Tensor, 
                           dt: float = 1.0/252.0, 
                           y0: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        T_steps, batch_size, _ = z_series.shape
        device = z_series.device
        
        if y0 is None:
            # On part de log-price=4.6 (environ 100$) et vol=0.04
            y0 = torch.tensor([4.605, 0.04], device=device).repeat(batch_size, 1)
            
        current_y = y0
        
        prices_hist = [current_y[:, 0]]
        vols_hist = [current_y[:, 1]]
        jumps_hist = [torch.zeros(batch_size, device=device)]
        
        past_jumps_memory = torch.zeros(batch_size, 0, device=device)
        
        for t in range(T_steps - 1):
            z_t = z_series[t]
            
            # 1. Diffusion Continue (Le "bruit" normal du marché)
            f_val = self.sde.f(t, current_y)    
            g_val = self.sde.g(t, current_y)    
            dW = torch.randn(batch_size, 2, device=device) * torch.sqrt(torch.tensor(dt))
            diffusion_term = torch.bmm(g_val, dW.unsqueeze(-1)).squeeze(-1)
            dy_continuous = f_val * dt + diffusion_term
            
            # 2. Calibration de l'Impact du Signal (Le "Fine Tuning")
            lambda_t = self.hawkes(z_t, dt=dt, past_jumps=past_jumps_memory)
            
            # --- CURSEUR 1 : SENSIBILITÉ ---
            # Avant : * 50.0 (Apocalypse) -> Maintenant : * 5.0 (Crise réaliste)
            # Cela augmente la proba de saut, mais ne la garantit pas à 100% à chaque pas.
            signal_impact = torch.sum(torch.abs(z_t), dim=1, keepdim=True) * 5.0
            total_lambda = lambda_t + signal_impact
            
            p_jump = 1 - torch.exp(-total_lambda * dt)
            jump_occurred = (torch.rand(batch_size, 1, device=device) < p_jump).float()
            
            # --- CURSEUR 2 : MAGNITUDE DES SAUTS ---
            # Avant : Force -15% fixe.
            # Maintenant : Distribution Normale centrée sur -2% (drift négatif) avec écart-type 4%.
            # Cela permet des sauts de -10% (rares) ou de -1% (fréquents).
            # C'est beaucoup plus organique.
            random_jump_size = torch.randn(batch_size, 1, device=device) * 0.04 - 0.02
            
            total_jump = jump_occurred * random_jump_size
            
            past_jumps_memory = torch.cat([past_jumps_memory, jump_occurred], dim=1)[:, -50:]
            
            # Update
            dy_jump = torch.zeros_like(dy_continuous)
            dy_jump[:, 0] = total_jump.squeeze(-1)
            
            current_y = current_y + dy_continuous + dy_jump
            prices_hist.append(current_y[:, 0])
            vols_hist.append(current_y[:, 1])
            jumps_hist.append(jump_occurred.reshape(-1))

        return (torch.stack(prices_hist), 
                torch.stack(vols_hist), 
                torch.stack(jumps_hist))