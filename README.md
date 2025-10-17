# DA5401 – Assignment 6: Imputation via Regression

**Name:** Bavge Deepak Rajkumar  
**Roll Number:** NA22B031  

---

## Overview

This assignment focuses on **data imputation techniques** — specifically **linear** and **nonlinear regression-based methods** — to handle missing data in the **Default of Credit Card Clients** dataset.

Missing values are introduced synthetically (MCAR pattern) into selected numeric variables, and different strategies are compared to observe how imputation quality impacts a downstream **classification task** (predicting credit default).

The assignment also contrasts imputation-based approaches with **listwise deletion**, highlighting the trade-offs between data completeness and model reliability.

---

## Dataset Used

- **Source:** [Kaggle – Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- **Shape:** 30,000 rows × 25 columns  
- **Target Variable:** `default.payment.next.month` (0 = No Default, 1 = Default)  
- **Features:** Demographic, credit limit, payment history, bill amounts, and repayment records.

> All features were originally complete. Missing values were introduced manually in 3 numeric columns to simulate real-world data gaps.

---

## Assignment Breakdown

### **Part A: Imputation via Regression**

#### Q1. Dataset Audit and Variable Selection
- Loaded and examined the dataset.
- Verified no natural missing values existed.
- Chose three numeric variables (`BILL_AMT3`, `PAY_AMT2`, `LIMIT_BAL`) for simulated missingness.

#### Q2. Construction of Dataset A (Simulated Missingness)
- Introduced ~10% random missingness (MCAR) in the three selected variables.
- Dataset A serves as the **base dataset** for all subsequent imputation experiments.

#### Q3. Dataset B – Linear Regression Imputation
- Restored non-target columns (`PAY_AMT2`, `LIMIT_BAL`) to their original form.
- Imputed missing `BILL_AMT3` values using a **Linear Regression model** trained on complete rows.
- Formed Dataset B with only the target column imputed, all other columns intact.

#### Q4. Dataset C – Nonlinear Regression Imputation
- Followed the same procedure as Dataset B but used a **Random Forest Regressor**.
- Captured nonlinear dependencies between predictors and the target column to improve imputation realism.

---

### **Part B: Classifier Setup and Evaluation**

#### Q1. Dataset D – Listwise Deletion
- Removed every row from Dataset A containing any missing value.
- Resulting dataset was smaller but completely free of NaNs.

#### Q2. Feature Standardization
- Used **StandardScaler** to standardize predictors in all datasets (A, B, C, D).  
- Scaling was applied using the training split only to prevent data leakage.

#### Q3. Model Evaluation (Logistic Regression)
- Trained separate **Logistic Regression classifiers** on each dataset.
- Evaluated each model on its corresponding test set using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**

| Dataset | Imputation Strategy | Accuracy | Macro F1 | Observation |
|----------|--------------------|-----------|-----------|--------------|
| **A** | Median (Baseline) | 0.8100 | 0.62 | Median robustly handles outliers. |
| **B** | Linear Regression | 0.8098 | 0.62 | Near-identical performance — confirms linear relationships. |
| **C** | Random Forest | 0.8098 | 0.62 | Nonlinear model adds complexity with no gain. |
| **D** | Listwise Deletion | 0.8037 | 0.61 | Slightly poorer due to data loss and reduced diversity. |

---

## Part C: Discussion and Inference

### Q1. Comparison of Methods
- Imputation-based models (A–C) achieved nearly identical accuracy (~0.81).
- Median and Linear Regression imputations were most efficient and robust.
- Nonlinear imputation (Random Forest) offered no improvement because the relationships between features (bill and payment amounts) are predominantly linear.

### Q2. Efficacy Discussion
- **Model D (Listwise Deletion)** performed slightly worse because removing incomplete rows reduced training data volume and potentially biased the sample distribution.
- **Linear vs. Nonlinear:** Linear regression matched nonlinear performance, reflecting that the imputed variable `BILL_AMT3` has linear dependence on neighboring bill amounts.
- **Best Strategy:** Use **simple median or linear regression imputation** for datasets with linear dependencies and random missingness.  
  These preserve structure without adding unnecessary model complexity.

> **Summary:** The imputation method had minimal influence on model accuracy because missingness was random (MCAR).  
> Simpler methods (median, linear) are sufficient, while listwise deletion should be avoided due to data loss.

---

## How to Run

1. Download the dataset from the [Kaggle source](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).  
2. Open `imputation.ipynb` in Jupyter Notebook.  
3. Run all cells sequentially.  
4. All datasets (A–D), splits, and classification results will be generated within the notebook.

> All code, outputs, and markdown analyses are contained in a single notebook file. No external dependencies beyond `pandas`, `numpy`, `scikit-learn`, and `seaborn` are required.

---

## Key Learnings

- Imputation is essential to maintain dataset integrity and prevent sample bias.  
- Median and linear regression imputations are effective for **MCAR** data with linear structure.  
- Nonlinear methods should be reserved for complex dependencies, not used by default.  
- Listwise deletion, while simple, sacrifices both data quantity and representativeness.  
- Standardization and consistent train–test splits are critical for fair model comparison.

---

## Final Recommendation

**Preferred Method:**  
→ **Linear Regression Imputation (Dataset B)**  

**Why:**  
- Retains all samples (no data loss).  
- Accurately reconstructs missing values with minimal error.  
- Delivers same predictive performance as nonlinear methods at a fraction of computational cost.

> In practice, always start with simple imputation.  
> Move toward model-based or nonlinear approaches only when exploratory analysis indicates nonlinearity or pattern-dependent missingness.
