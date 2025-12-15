import torch
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict, List

class SDEModelTrainer:
    """
    Gère le cycle de vie de l'entraînement : Forward -> Loss -> Backward -> Update.
    """
    def __init__(self, model, learning_rate: float = 0.005, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        
        # On optimise les poids du SDE (Neural Networks mu, kappa, xi, eta)
        # On peut aussi optimiser le NLP, mais commençons par figer le NLP pour stabiliser le SDE.
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # Scheduler : Réduit le learning rate si ça stagne (très utile pour les SDE)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
    def train_step(self, z_signal: torch.Tensor, target_vol: torch.Tensor, loss_fn, dt: float) -> float:
        """Une étape d'optimisation (Gradient Descent)."""
        self.model.train() # Mode entraînement
        self.optimizer.zero_grad()
        
        # 1. Forward Pass : Le modèle génère une trajectoire
        # Output actuel du modèle : [Time, Batch] (ex: [1500, 1])
        _, pred_vols, _ = self.model.forward_simulation(z_signal, dt=dt)
        
        # --- CORRECTION DE DIMENSION ICI ---
        # La Loss attend : [Batch, Time, 1]
        # On permute (Time, Batch) -> (Batch, Time) puis on ajoute la dimension 1
        pred_vols = pred_vols.permute(1, 0).unsqueeze(-1)
        
        # 2. Compute Loss
        loss = loss_fn(pred_vols, target_vol)
        
        # 3. Backward Pass
        loss.backward()
        
        # 4. Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 5. Update Weights
        self.optimizer.step()
        
        return loss.item()

    def fit(self, 
            z_train: torch.Tensor, 
            vol_train: torch.Tensor, 
            epochs: int = 100, 
            loss_fn = None) -> List[float]:
        
        if loss_fn is None:
            # Import local pour éviter les cycles
            from neural_quant.engine.losses import VolatilityMatchLoss
            loss_fn = VolatilityMatchLoss()
            
        print(f"🚀 [Engine] Démarrage de l'entraînement sur {epochs} époques...")
        dt = 1.0 / 252.0
        history = []
        
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            loss_val = self.train_step(z_train, vol_train, loss_fn, dt)
            history.append(loss_val)
            
            # Update scheduler
            self.scheduler.step(loss_val)
            
            # Affichage tqdm
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss_val:.6f} | LR: {self.optimizer.param_groups[0]['lr']:.5f}")
            
        return history