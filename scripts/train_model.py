import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 1. Setup des imports (Pour que Python trouve 'src')
# On remonte d'un cran depuis 'scripts/' vers la racine
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from neural_quant.data import LMEDataLoader, DataProcessor
from neural_quant.models.hybrid import HybridJumpDiffusion
from neural_quant.engine import SDEModelTrainer, VolatilityMatchLoss
from neural_quant.utils import set_seed

def main():
    # --- A. Configuration ---
    set_seed(42)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Sur Mac M1/M2, on peut tenter "mps" mais "cpu" est plus stable pour les SDE complexes
    DEVICE = "cpu" 
    
    print(f"⚙️  Démarrage du Training Pipeline sur {DEVICE}...")
    
    # --- B. Chargement des Données (ETL) ---
    ticker = "HG=F" # Contrat Futur Cuivre
    loader = LMEDataLoader(data_dir=project_root / "data" / "raw")
    df_raw = loader.fetch_data(ticker, start_date="2020-01-01")
    
    print(f"📊 Données chargées : {len(df_raw)} jours de trading pour {ticker}")
    
    # Transformation en Tenseurs
    processor = DataProcessor(window_size=5)
    # t_returns: [1, T, 1], t_vols: [1, T, 1]
    t_returns, t_vols = processor.to_tensors(df_raw, device=DEVICE)
    
    # --- C. Préparation du Modèle ---
    # Pour l'entraînement, on n'a pas encore l'historique complet des News sur 4 ans.
    # On va simuler un signal Z "neutre" bruité pour calibrer la dynamique interne du SDE.
    # Dans la V2, on remplacera ça par le vrai signal NLP historique.
    T_steps = t_returns.shape[1]
    input_dim = 1
    
    # On crée un signal Z synthétique (bruit blanc) pour l'entraînement
    # Le modèle va apprendre à générer de la vol stochastique même sans signal fort
    z_train = torch.randn(T_steps, 1, input_dim, device=DEVICE) # [Time, Batch=1, Dim]
    
    print("🧠 Initialisation du modèle Hybride (SDE + Hawkes)...")
    model = HybridJumpDiffusion(input_dim=input_dim, hidden_dim=32)
    
    # --- D. Entraînement (Calibration) ---
    trainer = SDEModelTrainer(model, learning_rate=0.01, device=DEVICE)
    loss_fn = VolatilityMatchLoss(alpha=10.0) # Alpha booste l'importance de la loss
    
    print("🚀 Lancement de la calibration...")
    history = trainer.fit(
        z_train=z_train, 
        vol_train=t_vols, # On veut que le SDE retrouve la vol réalisée du Cuivre
        epochs=50,        # 50 époques pour tester (mettre 200 pour vrai résultat)
        loss_fn=loss_fn
    )
    
    # --- E. Sauvegarde ---
    save_dir = project_root / "models" / "saved"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "copper_sde_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Modèle sauvegardé sous : {save_path}")
    
    # --- F. Visualisation des Résultats ---
    # 1. Courbe d'apprentissage
    plt.figure(figsize=(10, 4))
    plt.plot(history, label='Training Loss (MSE)')
    plt.title('Calibration du SDE sur le Cuivre (HG=F)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("✅ Pipeline terminé avec succès.")

if __name__ == "__main__":
    main()