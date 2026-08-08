# Adult Income Prediction using Machine Learning

## 1. Problem Statement

The objective of this project is to build and compare multiple supervised machine learning classification models to predict whether an individual's annual income is **greater than 50K** or **less than or equal to 50K** using the Adult Census Income Dataset.

The project demonstrates the complete machine learning workflow, including:

- Data preprocessing
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature encoding
- Feature scaling
- Model training
- Model evaluation
- Model comparison
- Streamlit web application deployment

---

# 2. Dataset Description

**Dataset Name:** Adult Census Income Dataset

**Source:**  Kaggle

### Dataset Summary

| Attribute | Description |
|------------|-------------|
| Problem Type | Binary Classification |
| Total Instances | 32,537 (after removing duplicate records) |
| Predictor Features | 14 |
| Target Variable | Income |

### Target Classes

- <=50K
- >50K

---
# 3. GitHub Repository Link

Repository URL:

**https://github.com/2025ac05651/ml-adult-income-prediction**

The repository contains:

- Complete source code
- Jupyter Notebook
- Trained Machine Learning models
- requirements.txt
- README.md
- Dataset
- Test dataset (CSV)
- Streamlit application

---
# 4. Machine Learning Models Used

The following supervised machine learning algorithms were implemented and compared.

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

---

# 5. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|---------:|----:|----------:|---------:|---------:|---------:|
| Logistic Regression | 0.8239 | 0.8497 | 0.7140 | 0.4490 | 0.5513 | 0.4674 |
| Decision Tree | 0.8026 | 0.7313 | 0.5896 | 0.5938 | 0.5917 | 0.4615 |
| k-Nearest Neighbors (kNN) | 0.8239 | 0.8440 | 0.6518 | 0.5778 | 0.6126 | 0.5007 |
| Gaussian Naive Bayes | 0.7915 | 0.8274 | 0.6501 | 0.2915 | 0.4025 | 0.3329 |
| **Random Forest (Ensemble)** | **0.8560** | **0.8977** | **0.7425** | **0.6161** | **0.6734** | **0.5863** |

---

# 6. Observations on Model Performance

| ML Model | Observation about Model Performance |
|-----------|-------------------------------------|
| Logistic Regression | Achieved good overall accuracy and high precision but comparatively lower recall, indicating that some positive income instances were missed. |
| Decision Tree | Produced balanced precision and recall but lower accuracy and AUC than most other models. It is more prone to overfitting than ensemble methods. |
| k-Nearest Neighbors (kNN) | Delivered balanced performance with better recall and F1-score than Logistic Regression, making it suitable for this dataset after feature scaling. |
| Gaussian Naive Bayes | Recorded the lowest overall performance because of its low recall and F1-score. The feature independence assumption limited its effectiveness. |
| Random Forest (Ensemble) | Achieved the highest Accuracy, AUC, F1-score and MCC, demonstrating strong generalization and robustness on the Adult Income dataset. |
| **Overall Winner** | **Random Forest (Ensemble)** was the best-performing model based on the evaluation results. It is highlighted as the overall winner, while the Streamlit application allows users to select and evaluate all five implemented models.. |

---

# 7. Project Structure

```
ml-adult-income-prediction/
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
---

# 8. Installation

Clone the repository.

```bash
git clone https://github.com/2025ac05651/ml-adult-income-prediction.git
```

Navigate to the project directory.

```bash
cd ml-adult-income-prediction
```

Install the required Python packages.

```bash
pip install -r requirements.txt
```

---

# 9. Running the Streamlit Application

Run the application using the following command.

```bash
streamlit run app.py
```

The application will be available locally at:

```text
http://localhost:8501
```

---

# 10. Live Streamlit Application

The deployed application can be accessed at:
https://2025ac05651-ml-adult-income-prediction.streamlit.app/


---

# 11. Screenshots

### Streamlit Home Page

<img width="1440" height="900" alt="2026-08-08 (17)" src="https://github.com/user-attachments/assets/ae181251-31e3-4d40-92ad-25c70ca18837" />

### Prediction Result

<img width="1440" height="900" alt="2026-08-08 (16)" src="https://github.com/user-attachments/assets/db8756ce-95d6-496c-9c03-b1161fdcf585" />

<img width="1440" height="900" alt="2026-08-08 (18)" src="https://github.com/user-attachments/assets/93feae28-ed58-492d-b51d-69aefae23a6d" />

<img width="1440" height="900" alt="2026-08-08 (24)" src="https://github.com/user-attachments/assets/66506720-fecd-4af9-bcfc-c08b5ba9c957" />

<img width="1440" height="900" alt="2026-08-08 (25)" src="https://github.com/user-attachments/assets/7b36c044-0f56-4187-be66-9d40e80655c0" />

<img width="1440" height="900" alt="2026-08-08 (26)" src="https://github.com/user-attachments/assets/0b3a339f-387e-485b-a37b-6d60f7d30c87" />

<img width="1440" height="900" alt="2026-08-08 (27)" src="https://github.com/user-attachments/assets/bc0e3ec8-2663-4b13-98c2-81874f1e274f" />

<img width="1440" height="900" alt="2026-08-08 (28)" src="https://github.com/user-attachments/assets/571f8587-6671-48ff-8dce-df9338acec3e" />

<img width="1440" height="900" alt="2026-08-08 (29)" src="https://github.com/user-attachments/assets/89425b06-69dc-432a-a96e-2f71458900db" />

# 12. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

---

# 13. Deployment

The application has been successfully deployed using **Streamlit Community Cloud**.

**Deployment URL**

https://2025ac05651-ml-adult-income-prediction.streamlit.app/


---

# 14. References

1. Dua, D. & Graff, C. UCI Machine Learning Repository – Adult Census Income Dataset. https://archive.ics.uci.edu/

2. Kaggle – Adult Census Income Dataset. https://www.kaggle.com/

3. Scikit-learn Documentation. https://scikit-learn.org/

4. Streamlit Documentation. https://docs.streamlit.io/

5. Pandas Documentation. https://pandas.pydata.org/

6. NumPy Documentation. https://numpy.org/

7. Matplotlib Documentation. https://matplotlib.org/

---

# 15. Author

**Name:** Sowmiya S

**BITS ID:** 2025AC05651

**Programme:** M.Tech in Artificial Intelligence and Machine Learning

**University:** BITS Pilani

---

## Acknowledgement

This project was developed as part of the **Machine Learning** course for the **M.Tech in Artificial Intelligence and Machine Learning** programme at **BITS Pilani**.
