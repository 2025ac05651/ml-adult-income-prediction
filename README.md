# Adult Income Prediction using Machine Learning

## Project Overview

This project was developed as part of the **BITS Pilani M.Tech in Artificial Intelligence and Machine Learning** Machine Learning Assignment.

The objective is to build and compare multiple machine learning classification models to predict whether an individual's annual income is:

- **<=50K**
- **>50K**

The project includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Streamlit web application
- Model deployment

---

# Dataset

**Dataset Name:** Adult Census Income Dataset

**Source:** UCI Machine Learning Repository / Kaggle

### Dataset Summary

- Total Instances: **32,537** (after removing duplicate records)
- Predictor Features: **14**
- Target Variable: **income**
- Problem Type: **Binary Classification**

Target Classes:

- <=50K
- >50K

---

# Machine Learning Models Used

The following classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest

---

# Model Performance

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-------|---------:|----:|----------:|---------:|---------:|---------:|
| Random Forest | 0.8560 | 0.8977 | 0.7425 | 0.6161 | 0.6734 | 0.5863 |
| k-Nearest Neighbors | 0.8239 | 0.8440 | 0.6518 | 0.5778 | 0.6126 | 0.5007 |
| Logistic Regression | 0.8239 | 0.8497 | 0.7140 | 0.4490 | 0.5513 | 0.4674 |
| Decision Tree | 0.8026 | 0.7313 | 0.5896 | 0.5938 | 0.5917 | 0.4615 |
| Naive Bayes | 0.7915 | 0.8274 | 0.6501 | 0.2915 | 0.4025 | 0.3329 |

### Best Performing Model

**Random Forest** achieved the best overall performance based on Accuracy, AUC Score, F1 Score, and Matthews Correlation Coefficient (MCC).

---

# Project Structure

```
ML_Assignment2/
│
├── app.py
├── README.md
├── requirements.txt
│
├── dataset/
│   ├── adult.csv
│   ├── adult_encoded.csv
│   └── test_data.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
│
└── notebooks/
    └── ML_Assignment2.ipynb
```

---

# Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_LINK>
```

Move into the project directory:

```bash
cd ML_Assignment2
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

# Running the Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# Sample Screenshots

### Streamlit Home Page

*(Insert screenshot of the main Streamlit interface here.)*

---

### Prediction Result

*(Insert screenshot showing a prediction result here.)*

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

---

# Deployment

**Streamlit Application**

> Add your deployed Streamlit URL here after deployment.

Example:

```
https://your-app-name.streamlit.app
```

---

# GitHub Repository

Add your GitHub repository URL here.

Example:

```
https://github.com/yourusername/ML_Assignment2
```

---


# Author

**Name:** SOWMIYA 

**BITS-ID:** 2025AC05651

**Programme:** M.Tech Artificial Intelligence and Machine Learning

**University:** BITS Pilani