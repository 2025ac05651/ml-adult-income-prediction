import streamlit as st
import pandas as pd
import joblib

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Adult Income Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# Load Models
# ==========================================

try:
    log_model = joblib.load("models/logistic_regression.pkl")
    dt_model = joblib.load("models/decision_tree.pkl")
    knn_model = joblib.load("models/knn.pkl")
    nb_model = joblib.load("models/naive_bayes.pkl")
    rf_model = joblib.load("models/random_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# ==========================================
# Title
# ==========================================

st.title("Adult Income Prediction using Machine Learning")

st.markdown("""
This application predicts whether a person's annual income is:

- **<=50K**
- **>50K**

using one of the following machine learning models:

- Logistic Regression
- Decision Tree
- k-Nearest Neighbors
- Naive Bayes
- Random Forest
""")

# ==========================================
# Sidebar
# ==========================================

st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "k-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest"
    ]
)

# ==========================================
# Input Section
# ==========================================

st.header("Enter Feature Values")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 17, 90, 35)
    workclass = st.number_input("Workclass (Encoded)", 0, 8, 4)
    fnlwgt = st.number_input("Final Weight", 10000, 1500000, 200000)
    education = st.number_input("Education (Encoded)", 0, 15, 9)
    education_num = st.number_input("Education Number", 1, 16, 10)
    marital_status = st.number_input("Marital Status (Encoded)", 0, 6, 2)
    occupation = st.number_input("Occupation (Encoded)", 0, 14, 7)

with col2:
    relationship = st.number_input("Relationship (Encoded)", 0, 5, 1)
    race = st.number_input("Race (Encoded)", 0, 4, 4)
    sex = st.number_input("Sex (Encoded)", 0, 1, 1)
    capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
    capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
    hours_per_week = st.number_input("Hours per Week", 1, 99, 40)
    native_country = st.number_input("Native Country (Encoded)", 0, 41, 39)

# ==========================================
# Create Input DataFrame
# ==========================================

input_data = pd.DataFrame([[
    age,
    workclass,
    fnlwgt,
    education,
    education_num,
    marital_status,
    occupation,
    relationship,
    race,
    sex,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country
]], columns=[
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education.num",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
    "native.country"
])
if st.button("Predict Income"):

    # Scale only for Logistic Regression and kNN
    if model_name == "Logistic Regression":
        prediction = log_model.predict(scaler.transform(input_data))[0]

    elif model_name == "k-Nearest Neighbors":
        prediction = knn_model.predict(scaler.transform(input_data))[0]

    elif model_name == "Decision Tree":
        prediction = dt_model.predict(input_data)[0]

    elif model_name == "Naive Bayes":
        prediction = nb_model.predict(input_data)[0]

    else:
        prediction = rf_model.predict(input_data)[0]

    st.subheader("Prediction")

    if prediction == 1:
        st.success("Predicted Income: >50K")
    else:
        st.info("Predicted Income: <=50K")

# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.markdown(
    "Developed as part of the **BITS Pilani M.Tech AI & ML - Machine Learning Assignment**."
)