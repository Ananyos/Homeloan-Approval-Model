# Homeloan-Approval-Model

## 📌 Overview

This project aims to predict whether a loan application should be approved or not using various machine learning models. The dataset contains historical loan application records with features such as income, credit history, loan amount, education, and more. The main goal is to develop a model that can assist financial institutions in making more accurate, fair, and data-driven loan decisions.

The project explores multiple classification algorithms and evaluates their performance, with a focus on interpretability and real-world application. Additional work includes feature selection, hyperparameter tuning, and the use of a custom probability threshold to better simulate strict lending policies.

---

## 📂 Project Contents

- **📁 Dataset**  
  - `loan-train-extended.csv`: Dataset containing historical loan application data.
  
- **📒 Jupyter Notebook**  
  - `Loan_Approval_Prediction.ipynb`: Full pipeline including EDA, preprocessing, feature selection, model training, tuning, evaluation, and prediction.

- **🔍 Key Features of the Project**
  - Exploratory Data Analysis (EDA)
  - Feature Engineering (e.g., Total Income)
  - Feature Selection using `SelectKBest`
  - Model training: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM
  - Hyperparameter Tuning with `GridSearchCV`
  - Custom threshold-based prediction
  - Feature Importance Analysis
  - Real-world test case simulation

---

## ✅ Final Result
- Best performing model: **Gradient Boosting**
- Accuracy: ~85%
- Custom threshold: 0.7 for stricter approval logic
- Key predictors: **Credit History**, **Total Income**, **Loan Amount**, **Property Area**

---

## ⚠️ Limitations & Future Work
- Trained on a U.S.-based dataset — may not generalize well globally (e.g., Thailand)
- Credit history feature is binary and oversimplified
- Future versions should incorporate local data, advanced credit features, and time-based updates

---

Feel free to run the notebook and try predicting new applications using the `predict_loan_approval()` function!
