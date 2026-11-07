# Regression Project: Car Price Prediction Engine 🏎️📈

This repository contains a complete, production-ready machine learning pipeline built completely from scratch without high-level libraries like Scikit-Learn. The goal is to predict a vehicle's market price (MSRP) using structural characteristics. 

Developed during the **ML by Zoomcamp** cohort.

##  Core Features & Engineering Procedures
* **Data Standardization:** Normalized messy feature schemas, missing records, and casing string inconsistencies via Pandas.
* **Log Transformation (`np.log1p`):** Eliminated severe right-skewed "long tail" price distributions to construct stable, normal bell-curve targets.
* **Custom Feature Engineering:** Programmed dynamic One-Hot Encoding functions targeting the top $N$ most frequent text categories per column to avoid the curse of dimensionality.
* **Matrix Stacking (`np.column_stack`):** Manually injected bias vector intercepts ($w_0$) straight into NumPy dimensional arrays.
* **Analytical Normal Equation Solver:** Solved weights using matrix transposition, dot products, and inversions.
* **L2 Regularization (Ridge Safety Net):** Prevented matrix division blowups caused by multi-collinear overlaps using dynamic regularized tuning ($r=1.0$).

## 🏁 Final Evaluation Metrics
* **Baseline Train RMSE:** 0.7552
* **Engineered Regularized Validation RMSE:** 0.6486
* **Final Test RMSE Score:** 0.6284 *(Highly stable generalization with zero overfitting!)*
