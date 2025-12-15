import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional

def plot_hybrid_simulation(
    prices: np.ndarray, 
    vols: np.ndarray, 
    jumps: np.ndarray, 
    signals: Optional[np.ndarray] = None,
    title: str = "Simulation Neural Jump-Diffusion"
):
    """
    Génère le graphique standard 'Hedge Fund Style' pour visualiser une trajectoire.
    """
    rows = 3 if signals is not None else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 4 * rows), sharex=True)
    
    # 1. Signal (Optionnel)
    if signals is not None:
        ax_sig = axes[0]
        ax_sig.fill_between(range(len(signals)), signals, color='green', alpha=0.3, label='Signal ($Z_t$)')
        ax_sig.set_title("Signal Exogène (News/Macro)", fontweight='bold')
        ax_sig.legend(loc='upper left')
        ax_sig.grid(True, alpha=0.3)
        ax_price = axes[1]
        ax_vol = axes[2]
    else:
        ax_price = axes[0]
        ax_vol = axes[1]

    # 2. Prix + Sauts
    ax_price.plot(prices, color='#2980b9', linewidth=1.5, label='Log-Prix ($X_t$)')
    
    # Détection des sauts pour lignes rouges
    jump_indices = np.where(jumps > 0)[0]
    if len(jump_indices) > 0:
        ax_price.vlines(jump_indices, np.min(prices), np.max(prices), 
                       colors='#c0392b', linestyles='--', alpha=0.5, label='Sauts ($dN_t$)')
    
    ax_price.set_title(f"{title} - Trajectoire de Prix", fontweight='bold')
    ax_price.set_ylabel("Log-Prix")
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)

    # 3. Volatilité
    ax_vol.plot(vols, color='#e67e22', label='Volatilité Stochastique ($V_t$)')
    ax_vol.set_title("Régime de Volatilité", fontweight='bold')
    ax_vol.set_ylabel("Variance")
    ax_vol.set_xlabel("Temps (Pas)")
    ax_vol.legend(loc='upper left')
    ax_vol.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig