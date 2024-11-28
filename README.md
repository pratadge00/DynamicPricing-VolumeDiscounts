# **Dynamic Pricing with Volume Discounts in Online Settings**

## **Project Overview**
This repository implements the **Optimal Price Estimation** phase of the **Pricing with Volume Discounts Bandit (PVD-B)** algorithm. The project features advanced dynamic pricing strategies for e-commerce applications, leveraging Bayesian inference, Thompson Sampling, and synthetically generated datasets.

### **The implementation includes:**
- **Synthetic dataset generation** to simulate e-commerce transactions.
- **A Bayesian Linear Regression model** with enhanced monotonicity constraints and basis functions.
- **Thompson Sampling** for iterative price optimization.
- **Visualizations** for insights into demand and profit trends.
- The **Volume Discount Learning** phase is planned for future work.

---

## **Repository Contents**
1. **`generated_single_product_dataset_with_seasonal_variation.csv`**: Synthetic dataset with seasonal trends, demand decay, and customer segmentation.
2. **`synthetic_data_generation.py`**: Script for generating the dataset.
3. **`optimal_price_estimation.py`**: Code for Bayesian demand modeling and Thompson Sampling optimization.
4. **`visualization.py`**: Script for visualizing demand curves, profit trends, and optimization results.

---

## **Features**

### **1. Synthetic Dataset Generation**
- Generates realistic transaction data:
  - Gaussian price distributions for distinct customer segments.
  - Demand decay functions to simulate price sensitivity.
  - Annual and weekly seasonal trends.
  - 6,000 unique customers with variable transaction frequencies.

### **2. Demand Modeling**
- **Bayesian Linear Regression (BLR):**
  - Price and time-related features with expanded basis functions.
  - Monotonicity constraints to reflect real-world economic principles.
  
- **Enhanced Basis Functions:**
  - Logarithmic, quadratic, and exponential decay functions.
  - Annual and weekly sinusoidal trends.
  - Day-of-week effects with one-hot encoding.

### **3. Thompson Sampling**
- Balances exploration and exploitation for price optimization.
- Iteratively adjusts prices using posterior predictive sampling.

### **4. Visualization**
- Historical prices and demand curves.
- Profit trends across the price range.
- Optimal prices over iterations.
- Daily transaction volumes over time.

---

## **Results**

### **Optimal Price Identification**
- Iterative refinement using Bayesian inference and Thompson Sampling.
- Strong alignment between predicted demand curves and transaction data.

### **Profit Maximization**
- Generated synthetic data tests showed significant profit optimization.

---

## **Usage**

### **Prerequisites**
- Python 3.8 or later.
- Required libraries: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `NumPyro`, `JAX`.

### **Running the Scripts**
1. **Dataset Generation**:
   - Run `synthetic_data_generation.py` to generate the synthetic dataset.
   - Output: `generated_single_product_dataset_with_seasonal_variation.csv`.

2. **Optimal Price Estimation**:
   - Update the `file_path` variable in `optimal_price_estimation.py` to point to the dataset file.
   - Run the script:
     ```bash
     python optimal_price_estimation.py
     ```

3. **Visualization**:
   - Use `visualization.py` to plot demand and profit trends:
     ```bash
     python visualization.py
     ```

---

## **Future Work**

- Implementation of the **Volume Discount Learning** phase.
- Validation on publicly available datasets (e.g., Kaggle, UCI).
- Adaptation for inventory constraints and multi-product scenarios.

---

## **License**
This project is licensed under the **Apache License 2.0**.

---

## **References**
1. *Dynamic Pricing with Volume Discounts in Online Settings* - Mussi, M., et al. (2023) [Link to Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26845).
