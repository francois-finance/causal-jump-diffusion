import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from neural_quant.nlp.extractor import CausalEventExtractor
from neural_quant.models.hybrid import HybridJumpDiffusion
from neural_quant.utils import set_seed

def calculate_metrics(prices_real, jumps):
    """Calcule les KPIs de risque sur les chemins réels (base 100)."""
    initial_price = prices_real[0, 0]
    final_prices = prices_real[-1, :]
    returns = (final_prices - initial_price) / initial_price
    
    var_95 = np.percentile(returns, 5)
    es_95 = returns[returns <= var_95].mean()
    
    cummax = np.maximum.accumulate(prices_real, axis=0)
    drawdowns = (prices_real - cummax) / cummax
    max_dd_dist = drawdowns.min(axis=0)
    avg_max_dd = max_dd_dist.mean()
    
    has_jumped = (np.sum(jumps, axis=0) > 0)
    prob_jump = np.mean(has_jumped)
    
    return {
        "VaR 95%": var_95,
        "Expected Shortfall": es_95,
        "Avg Max Drawdown": avg_max_dd,
        "Crash Probability": prob_jump
    }

def run_strategy_backtest(median_prices, signals, threshold=2.0):
    """
    Stratégie Long / Short.
    - Si Signal < Threshold : On est LONG (on suit le marché).
    - Si Signal > Threshold : On est SHORT (on gagne quand le marché perd).
    """
    # Calcul des rendements du benchmark
    asset_returns = np.diff(median_prices) / median_prices[:-1]
    
    strategy_returns = np.zeros_like(asset_returns)
    positions = np.ones_like(asset_returns) # 1 = Long, -1 = Short

    for t in range(len(asset_returns)):
        prev_signal = signals[t]
        
        # --- RÈGLE AGRESSIVE (Long/Short) ---
        if prev_signal > threshold:
            # ALERTE CRISE : On parie à la baisse (Short)
            positions[t] = -1.0 
            strategy_returns[t] = -1.0 * asset_returns[t]
        else:
            # CALME : On suit la hausse (Long)
            positions[t] = 1.0 
            strategy_returns[t] = asset_returns[t]
            
    # Reconstruction de la courbe
    strat_curve = [100.0]
    for r in strategy_returns:
        strat_curve.append(strat_curve[-1] * (1 + r))
        
    return np.array(strat_curve), positions

def main():
    set_seed(42)
    DEVICE = "cpu"
    print("--- 🔮 Démarrage du Risk & Trading Engine ---")

    # 1. Load Model
    model_path = project_root / "models" / "saved" / "copper_sde_v1.pth"
    if not model_path.exists():
        print("❌ Erreur: Modèle non trouvé.")
        return
    model = HybridJumpDiffusion(input_dim=1, hidden_dim=32).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Scenario Definition
    print("📰 Analyse du scénario 'Mining Strike'...")
    news_scenario = [
        "Copper market stable, demand steady",
        "Minor delays in shipping reported",
        "Rumors of union strikes in Chile mines",
        "BREAKING: Major strike halts Escondida mine production", # Crisis start
        "Violence erupts, military intervention discussed",
        "Supply shortage fears drive market volatility",
        "Negotiations begin between union and government",
        "Strike ends, production resumes slowly", # Recovery
        "Market stabilizes as supply chain restores",
        "Copper prices normalize"
    ]
    extended_scenario = []
    for news in news_scenario: extended_scenario.extend([news] * 5)
        
    nlp = CausalEventExtractor()
    z_signal_one = nlp.process_news_batch(extended_scenario) # [50, 1]
    signal_np = z_signal_one.squeeze().numpy()

    # 3. Monte Carlo Simulation
    N_PATHS = 1000
    print(f"🎲 Monte Carlo ({N_PATHS} chemins)...")
    z_batch = z_signal_one.unsqueeze(1).repeat(1, N_PATHS, 1).to(DEVICE)
    
    with torch.no_grad():
        prices_log, vols, jumps = model.forward_simulation(z_batch, dt=1/252)

    # Conversion Base 100
    prices_real = np.exp(prices_log.numpy())
    prices_real = prices_real / prices_real[0, :] * 100.0
    median_path = np.median(prices_real, axis=1)
    
    # 4. Backtest Strategy
    print("📈 Exécution de la stratégie 'Smart Hedging'...")
    strat_curve, positions = run_strategy_backtest(median_path, signal_np, threshold=2.0)
    
    # Calcul PnL final
    pnl_bh = (median_path[-1] - 100)
    pnl_strat = (strat_curve[-1] - 100)

    # 5. Visualization Dashboard (2x2 Grid)
    fig = plt.figure(figsize=(14, 10))
    grid = plt.GridSpec(2, 2)

    # --- Plot A: Monte Carlo Risks ---
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.plot(prices_real[:, :50], color='gray', alpha=0.1)
    ax1.plot(median_path, color='blue', linewidth=2, label='Benchmark (Median)')
    ax1.plot(np.percentile(prices_real, 5, axis=1), color='red', linestyle='--', label='Worst Case 5%')
    
    crisis_mask = signal_np > 1.5
    if np.any(crisis_mask):
        ax1.axvspan(np.where(crisis_mask)[0][0], np.where(crisis_mask)[0][-1], color='red', alpha=0.1, label='Zone de Risque (IA)')
    
    ax1.set_title("1. Simulation de Risque (Monte Carlo)", fontweight='bold')
    ax1.set_ylabel("Prix Base 100")
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot B: Volatility Regime ---
    ax2 = fig.add_subplot(grid[0, 1], sharex=ax1)
    mean_vol = np.mean(vols.numpy(), axis=1)
    ax2.fill_between(range(len(signal_np)), signal_np, color='green', alpha=0.3, label='Signal IA ($Z_t$)')
    ax2.plot(mean_vol * 10, color='orange', linewidth=2, label='Volatilité Modèle (x10)')
    ax2.axhline(1.5, color='red', linestyle=':', label='Seuil de Hedging')
    ax2.set_title("2. Signal IA vs Volatilité du Modèle", fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Plot C: PnL Strategy Backtest ---
    ax3 = fig.add_subplot(grid[1, :])
    ax3.plot(median_path, color='grey', linestyle='--', label=f'Buy & Hold (PnL: {pnl_bh:+.1f}%)')
    ax3.plot(strat_curve, color='purple', linewidth=2.5, label=f'AI Smart Hedging (PnL: {pnl_strat:+.1f}%)')
    
    # Colorier les zones où on est CASH
    cash_zones = np.where(positions == 0)[0]
    if len(cash_zones) > 0:
        ax3.fill_between(range(1, len(positions)+1), min(strat_curve), max(strat_curve), 
                         where=(positions==0), color='purple', alpha=0.1, label='Position: CASH (Protégé)')

    ax3.set_title("3. Backtest de Stratégie : Protection contre le Crash", fontweight='bold')
    ax3.set_ylabel("Valeur du Portfolio (Base 100)")
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel("Jours de Trading")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()