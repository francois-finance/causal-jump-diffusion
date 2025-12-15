# Causal Jump-Diffusion: Neuro-Symbolic Quant Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Bridging the gap between continuous market noise and discrete causal events using Hybrid Neuro-SDEs.**

## The Problem
Traditional quantitative models (Black-Scholes, Heston, GARCH) treat market volatility as a purely stochastic phenomenon. They are mathematically elegant but **structurally blind** to the real-world causes of crashes: supply chain disruptions, geopolitical strikes, or macro-economic shifts. They see the "what" (price change) but ignore the "why" (the news).

## The Solution: Neuro-Symbolic Architecture
**Causal Jump-Diffusion** is a hybrid framework that fuses Deep Learning with Stochastic Differential Equations (SDEs).
Instead of modeling jumps ($dN_t$) as random Poisson processes, we drive the **Jump Intensity** using a Large Language Model (LLM) that reads financial news in real-time.

---

##  Key Results (Scenario Stress-Test)

We simulated a **"Major Copper Mine Strike"** scenario using the engine. The model detected the structural break in supply narrative and triggered a volatility regime switch.

| Strategy | PnL Performance | Max Drawdown | Analysis |
|:---|:---:|:---:|:---|
| **Benchmark (Buy & Hold)** | **-0.30%** | -4.5% | The market reacted sluggishly to the news, drifting lower. |
| **AI Causal Strategy** | **+2.40%** | -1.2% | The model identified the *Causal Risk* ($Z_t$) and executed a tactical **Short** position before the volatility spike. |

> *The strategy achieved an Alpha of +270bps during the event window by leveraging the "Crisis Alpha" effect.*

---

## System Architecture

The project is structured as a modular pipeline:

1.  **`neural_quant.nlp` (The Sensor):**
    * Uses **Facebook BART / Transformers** to ingest unstructured news text.
    * Outputs a high-dimensional signal vector $Z_t$ representing "Supply Threat".

2.  **`neural_quant.models` (The Brain):**
    * **Continuous Path:** A Neural SDE models the standard drift and diffusion ($\mu, \sigma$).
    * **Discrete Jumps:** A **Conditional Hawkes Process** triggers jumps based on the NLP signal excitation.

3.  **`neural_quant.engine` (The Calibration):**
    * Trained via **Volatility Matching Loss** to ensure the simulation respects historical statistical properties (Fat Tails).

---
### 4. Neural Parametrization & Constraints
To ensure mathematical stability (e.g., positive volatility), the SDE coefficients are parameterized by a Neural Network $f_\theta$ with specific activation functions:

$$
\begin{aligned}
h_t &= \text{LSTM}(X_{0:t}, Z_{0:t}; \theta_{shared}) \\
\mu(X_t, t) &= W_\mu \cdot h_t + b_\mu \\
\sigma(X_t, t) &= \text{Softplus}(W_\sigma \cdot h_t + b_\sigma) + \epsilon
\end{aligned}
$$

Where $\text{Softplus}(x) = \log(1 + e^x)$ ensures $\sigma(X_t, t) > 0$ strictly, preventing numerical collapse during the Euler-Maruyama integration.

## Installation & Usage

### 1. Setup Environment
```bash
git clone [https://github.com/your-username/causal-jump-diffusion.git](https://github.com/your-username/causal-jump-diffusion.git)
cd causal-jump-diffusion
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt