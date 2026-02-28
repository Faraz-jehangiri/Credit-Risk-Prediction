# 🧮 Credit Risk Prediction – Loan Default/Approval

Predicting whether a loan applicant is a **good or bad credit risk** using real‑world style data from the Kaggle Loan Prediction dataset.

---

## 🎯 Objective

Build a **binary classification model** that predicts whether a loan application should be **approved** (low risk) or **rejected** (high risk), based on:

- Income  
- Loan amount  
- CIBIL score  
- Education and other applicant details  

The goal is to understand **which factors matter** and **how well a simple model like Logistic Regression can separate good vs bad risk**.

---

## 📂 Dataset

- Source: Kaggle – Loan Prediction dataset  
- Size: 4,269 rows (applicants)  
- Target column: `loan_status` (0/1 after preprocessing)  
- Example features:
  - `income_annum`
  - `loan_amount`
  - `cibil_score`
  - `loan_term`
  - Categorical fields like gender, education, self employment, property area, etc.

Basic preprocessing steps:

- Removed ID‑like columns (no predictive value, only identifiers).
- Handled missing values appropriately.
- Stripped extra spaces from string columns.
- Encoded categorical variables into numeric form.

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook includes EDA to understand how different features relate to loan approval:

- **Distributions**
  - Histograms of income and loan amount  
  - Distribution of CIBIL scores

- **Category vs Target**
  - Education vs loan_status
  - Other categorical features vs loan_status

- **Relationships**
  - Scatter plots (e.g. `cibil_score` vs `income_annum`, colored by approval)  
  - Clear trend: **higher CIBIL scores correspond to a much higher approval rate**, especially near 900.

These visuals help build intuition about what a “safe” applicant looks like before any machine learning.

---

## 🧠 Model – Logistic Regression

For the classification task, I used **Logistic Regression** as a simple and interpretable baseline model:

- Train/test split: 80% train, 20% test (with stratification to preserve class balance).
- Features: numeric + encoded categorical columns (excluding IDs).
- Target: binary `loan_status`.

### Training

- Model: `LogisticRegression(max_iter=1000)`
- The model learns a weight for each feature that increases or decreases the log‑odds of approval.

---

## 📊 Evaluation

### Accuracy

On the held‑out test set:

- **Accuracy:** ~**84.19%**

This means the model correctly predicts roughly **84 out of 100** applicants in the test set.

### Confusion Matrix

Test‑set confusion matrix (Actual rows × Predicted columns):

|            | Predicted 0 | Predicted 1 |
|------------|-------------|-------------|
| **Actual 0** | 227         | 96          |
| **Actual 1** | 39          | 492         |

Interpretation:

- **True Negatives (227):** correctly identified high‑risk (class 0) applicants.
- **True Positives (492):** correctly identified low‑risk (class 1) applicants.
- **False Positives (96):** model predicted low risk but they were actually high risk (risky approvals).
- **False Negatives (39):** model predicted high risk but they were actually low risk (missed good customers).

So the model:

- Performs well overall,
- But makes **more false positives than false negatives**, which is important from a credit risk perspective (approving bad risks can be more costly than rejecting good ones).

---

## 🏗️ Tech Stack

- **Language:** Python  
- **Libraries:**
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualizations
  - `scikit-learn` – model building and evaluation

---

## 🚀 How to Use This Notebook

1. Clone the repository.
2. Install dependencies (virtual environment recommended).
3. Open `Task 2.ipynb` in Jupyter / VS Code.
4. Run the notebook cells step by step:
   - Data loading & cleaning  
   - EDA visualizations  
   - Model training  
   - Accuracy + confusion matrix

The notebook is written in a way that walks through the process gradually so it’s easier to follow and modify.

---

## 🔄 Possible Next Steps

Some ideas to extend this project:

- Try **Decision Tree** or **Random Forest** and compare with Logistic Regression.
- Tune class weights or thresholds to reduce **false positives** (risky approvals).
- Add more feature engineering (ratios like income/loan_amount, etc.).
- Use cross‑validation instead of a single train/test split.

---

## ✍️ Notes

This project is mainly focused on:

- Understanding **credit risk intuition** (CIBIL, income, loan amount),
- Practicing **end‑to‑end ML workflow**: cleaning → EDA → model → evaluation,
- Keeping the model **interpretable**, not just accurate.

Feel free to open issues or suggestions if you have ideas to improve the model or analysis.
