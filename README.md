# Breast Cancer Classification - ML Assignment 2

## Problem Statement
Compare five classification algorithms for identifying
benign and malignant tumors using diagnostic measurements.

This project is for educational use only.

## Dataset Description
- Source: UCI Breast Cancer Wisconsin (Diagnostic)
- Dataset URL: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- Total instances: 569
- Input features: 30
- Target: diagnosis (B = 0, M = 1)
- Training records: 455
- Testing records: 114
- Train/test split: 80/20, stratified, random_state=42

## Machine Learning Models
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

## Evaluation Metrics
Accuracy, AUC, Precision, Recall, F1 and MCC.

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9649 | 0.996 | 0.975 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0 | 0.9286 | 0.963 | 0.9442 |

## Model Observations

### Logistic Regression
Accuracy: 0.9649. Highest AUC: 0.9960.
Three malignant cases were incorrectly classified as benign.

### Decision Tree
Accuracy: 0.9298. Lowest MCC: 0.8492.
Four false positives and four false negatives.

### KNN
Accuracy: 0.9561. F1: 0.9383.
Four malignant cases were missed.

### Naive Bayes
Precision: 1.0000, Recall: 0.8333.
Seven malignant cases were missed.

### Support Vector Machine
Accuracy: 0.9737, F1: 0.9630, MCC: 0.9442.
Zero false positives and three false negatives.

## Overall Winning Model
Support Vector Machine achieved the highest Accuracy,
F1 and MCC in this experiment.

Logistic Regression achieved the highest AUC.
SVM is selected using MCC as the primary ranking metric,
followed by F1 and AUC.

## GitHub Repository
Add your actual repository URL here.

## Streamlit Application
Add your live Streamlit URL after deployment.

## How to Run
1. Install dependencies: pip install -r requirements.txt
2. Start application: streamlit run app.py
3. Upload test_data.csv
4. Select one of the five trained models
5. View metrics, confusion matrix and classification report

## Limitations
This is a classroom experiment on a small held-out test set.
The results do not establish clinical suitability.
