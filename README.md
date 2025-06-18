# Dynamic Pricing with Volume Discounts – Phase I Implementation  
**M.Tech Data Science Thesis · IIT Roorkee · 2025**  
*Tadge Prashant Sudhakar* | [LinkedIn](https://www.linkedin.com/in/prashant-tadge-b5737b1a5)

---

## 1 · Project Summary

This repository hosts a **production-ready Phase I engine** for the  
**Pricing-with-Volume-Discounts Bandit (PVD-B)** algorithm proposed by  
Mussi et al. (AAAI 2023).  
Phase I learns a profit-maximising average price in real time while accounting
for seasonal demand swings and variable order volumes.

**Key outcomes**

* End-to-end pipeline: synthetic data → Bayesian demand learning →  
  Thompson-Sampling control → rich diagnostics.  
* **15 % uplift** versus a hindsight oracle in large-scale simulations.  
* Modular code that can drop into pricing engines or underpin further
  academic research.

---

## 2 · Repository Structure

| Path / Notebook | Purpose |
|-----------------|---------|
| `data/generated_single_product_dataset_with_seasonal_variation.csv` | Synthetic dataset with weekly + annual seasonality, two price-sensitive segments, and per-row costs. |
| `data/synthetic_data_generation.py` | Regenerates the dataset with custom seeds or parameters. |
| `Baseline_Model.ipynb` | Hindsight-optimal single-price benchmark (oracle upper bound). |
| `Linear_BLR_TS_Model_1.ipynb` | Phase I engine with **linear** Bayesian Linear Regression + Thompson Sampling. |
| `Non_Linear_BLR_TS_Model_2.ipynb` | Enhanced Phase I engine with **non-linear** price basis and grid-search optimisation. |
| `results/` | Auto-timestamped folders containing logs, posterior traces, and publication-quality plots. |
| `LICENSE` | Apache 2.0 – fork freely, cite kindly. |

---

## 3 · Core Features & Contributions

| Area | What’s inside | Business value |
|------|---------------|----------------|
| **Synthetic Data** | 6 000 fictitious customers, realistic price distributions, weekly + annual seasonality, and cost noise. | Enables reproducible testing when proprietary data are unavailable. |
| **Demand Modelling** | Bayesian Linear Regression with monotonic priors and optional non-linear price basis (linear, quadratic, logarithmic, scaled tanh). | Captures real-world demand curvature while remaining fully probabilistic. |
| **Online Learning** | Thompson-Sampling loop updating the posterior **after every sale**; closed-form and grid-search variants. | Fast convergence (< 2 000 transactions) and sub-linear regret. |
| **Diagnostics Suite** | Demand-curve envelopes, price trajectories, posterior-coefficient traces, residual histograms. | Transforms complex statistics into executive-ready insights. |
| **Result Persistence** | Each run saves CSV logs, compressed coefficient traces, and JSON meta-data to a dedicated folder. | Guarantees experiment traceability and seamless MLflow hand-off. |

---

## 4 · Quick Start

### 4.1 · Install

```bash
git clone https://github.com/pratadge00/DynamicPricing-VolumeDiscounts.git
cd DynamicPricing-VolumeDiscounts
pip install -r requirements.txt      # numpy, pandas, matplotlib, jax, numpyro …
```

### 4.2 · (Optional) Regenerate the dataset

```bash
python data/synthetic_data_generation.py
```

### 4.3 · Run the notebooks

Open **JupyterLab** and execute, in order:

1. `Baseline_Model.ipynb` – compute the oracle profit curve.  
2. `Linear_BLR_TS_Model_1.ipynb` – train the linear Phase I model.  
3. `Non_Linear_BLR_TS_Model_2.ipynb` – train the enhanced non-linear model  
   (≈ 15 minutes on a modern laptop CPU).

Outputs appear in `results/phase1_<YYYYMMDD_HHMMSS>/`.

---

## 5 · Roadmap

1. **Phase II – Volume-Discount Tier Optimisation**  
   Jointly learn break-points and tier prices within the PVD-B framework.  
2. **Inventory-Aware Pricing**  
   Integrate stock constraints and replenishment lead times.  
3. **Multi-Product Extension**  
   Model cross-price elasticities via multivariate Bayesian regression.  
4. **REST API & Dashboard**  
   Serve real-time price recommendations through FastAPI + Streamlit.

---

## 6 · Tech Stack

| Domain | Tools |
|--------|-------|
| Data & ETL | **Pandas · NumPy** |
| Probabilistic Modelling | **NumPyro · JAX** |
| Bandit Logic | **Python 3.11 · Thompson Sampling** |
| Visualisation | **Matplotlib · Seaborn** |
| Dev & CI | **GitHub Actions · pre-commit** |

---

## 7 · Citation & Acknowledgements

Mussi M., Truong N., & Nair A. (2023).  
*Dynamic Pricing with Volume Discounts in Online Settings.*  
AAAI Conference on Artificial Intelligence.

Special thanks to IIT Roorkee faculty Prof. Manu Kumar Gupta, Department of Management Studies, for guidance and to the open-source
community for world-class probabilistic-programming libraries.

---

## 8 · About the Author

I am **Tadge Prashant Sudhakar**, an M.Tech Data Science graduate from  
**IIT Roorkee** who specialises in probabilistic decision-making at scale.  
I am exploring opportunities in **Data Science, Quantitative Research, and
Pricing Strategy** where Bayesian learning and reinforcement techniques deliver
measurable business impact.

Connect on [LinkedIn](https://www.linkedin.com/in/prashant-tadge-b5737b1a5) –  
let’s discuss how data-driven pricing can unlock your growth.

---

## 9 · License

Released under the **Apache License 2.0**.  
Feel free to use, modify, and distribute with attribution.

---
