# QUANTITATIVE RESEARCH REPORT
## Causal Jump-Diffusion: A Neuro-Symbolic Approach to Market Crashes

**Author:** François Dubreu  
**Date:** December 2025  
**Topic:** Quantitative Finance / NLP / Stochastic Modelling  

---

### 1. Executive Summary

Traditional financial models (Black-Scholes, Heston) often fail to capture the magnitude of "fat-tail" events caused by exogenous shocks (geopolitics, supply chain disruptions). This research introduces a **Neuro-Symbolic Hybrid Model** that integrates Large Language Models (LLMs) with Stochastic Differential Equations (SDEs).

By extracting a causal "threat signal" from unstructured news text in real-time, the model dynamically adjusts the intensity of a Hawkes Process, triggering discrete price jumps. Backtesting on a simulated "Mining Strike" scenario demonstrates a significant improvement in risk management, generating an **Alpha of +2.7%** against a benchmark buy-and-hold strategy during the crisis window.

---

### 2. Mathematical Framework

The asset price dynamics are modeled using a non-linear Jump-Diffusion process. Unlike standard Poisson jumps, the jump intensity is conditional on an exogenous text-based signal.

#### 2.1 The Asset Dynamics
Let $X_t = \log(S_t)$ be the log-price of the asset. The continuous time dynamics follow:

$$dX_t = \mu(X_t, t)dt + \sigma(X_t, t)dW_t + J_t dN_t$$

Where:
* $\mu(X_t, t)$: The drift (trend), approximated by a Neural Network.
* $\sigma(X_t, t)$: The diffusion (volatility), constrained to be positive via a Softplus activation.
* $W_t$: A standard Brownian motion.
* $J_t$: The random jump size, sampled from a distribution calibrated to negative skewness ($\mu_J < 0$) during high-stress regimes.

#### 2.2 The Causal Intensity (NLP-Hawkes)
The counting process $N_t$ is governed by a stochastic intensity $\lambda_t$, defined as:

$$\lambda_t = \lambda_{\infty} + \sum_{t_i < t} \alpha e^{-\beta(t - t_i)} + \Psi(Z_{NLP}(t))$$

* **$\lambda_{\infty}$**: Baseline background noise.
* **Hawkes Kernel**: Represents the self-exciting nature of volatility clustering (endogenous risk).
* **$\Psi(Z_{NLP})$**: The forcing term derived from the LLM. The signal $Z_{NLP}$ is extracted using a BART-Large model zero-shot classifier on live news feeds.

---

### 3. Model Architecture

The system is implemented in Python using a modular architecture:

* **Natural Language Processing (NLP):** Uses `Facebook/BART-Large-MNLI` to quantify the probability of specific risk scenarios (e.g., "Strike", "Shortage") from text.
* **Neural SDE:** Built on `torchsde`, allowing for backpropagation through the stochastic paths to calibrate drift and diffusion parameters.
* **Event Engine:** A custom discrete-event simulator that couples the continuous path with the jump process based on the `Hawkes` intensity.

---

### 4. Backtest & Results

We subjected the model to a "Stress Test" scenario involving a major strike in a Chilean copper mine (5-day event window).

**Scenario Parameters:**
* **Input:** Sequence of 10 news headlines ranging from rumors to confirmed production halts.
* **Strategy:** A tactical Long/Short strategy. The system switches to `Short` when the Jump Probability $P(dN_t > 0)$ exceeds a calibrated threshold.

**Performance Metrics:**

| Metric | Benchmark (Market) | AI Causal Strategy |
| :--- | :---: | :---: |
| **Total Return** | -0.30% | **+2.40%** |
| **Max Drawdown** | -4.50% | **-1.20%** |
| **VaR (95%)** | -2.10% | **-0.85%** |

**Interpretation:**
The model successfully identified the "pre-shock" phase via the NLP signal before the market fully priced in the supply disruption. This allowed the strategy to hedge downside risk effectively ("Crisis Alpha").

---

### 5. Conclusion

This research demonstrates that integrating unstructured data (News) directly into the mathematical definition of Volatility ($dN_t$) offers a superior estimation of tail risk compared to purely statistical models.

Future work will focus on:
1.  Replacing the proxy calibration with a historical training set of 2020-2024 financial news.
2.  Implementing "FinBERT" for domain-specific sentiment analysis.

---

### Appendix: Implementation Details

**Stack:** Python 3.9, PyTorch, TorchSDE, HuggingFace Transformers, Hydra.  
**Hardware:** Trained on CPU (inference-optimized).  
**Repository:** `github.com/francoisdubreu/causal-jump-diffusion`