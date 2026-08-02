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

st.info("""
### 📊 About this Project

This application predicts whether an individual's annual income is:

- **Greater than 50K**
- **Less than or equal to 50K**

The prediction is performed using five machine learning classification algorithms trained on the **Adult Census Income Dataset**.

**Available Models**
- Logistic Regression
- Decision Tree
- k-Nearest Neighbors (kNN)
- Gaussian Naive Bayes
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
# Model Metrics
# ==========================================
metrics={
"Random Forest":{"Accuracy":"85.60%","AUC":"89.77%","F1":"67.34%","MCC":"58.63%"},
"Logistic Regression":{"Accuracy":"82.39%","AUC":"84.97%","F1":"55.13%","MCC":"46.74%"},
"Decision Tree":{"Accuracy":"80.26%","AUC":"73.13%","F1":"59.17%","MCC":"46.15%"},
"k-Nearest Neighbors":{"Accuracy":"82.39%","AUC":"84.40%","F1":"61.26%","MCC":"50.07%"},
"Naive Bayes":{"Accuracy":"79.15%","AUC":"82.74%","F1":"40.25%","MCC":"33.29%"}
}

m=metrics[model_name]
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")

st.sidebar.metric("Accuracy", m["Accuracy"])
st.sidebar.metric("AUC Score", m["AUC"])
st.sidebar.metric("F1 Score", m["F1"])
st.sidebar.metric("MCC", m["MCC"])
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Information")

st.sidebar.write("**Dataset:** Adult Census Income")

st.sidebar.write("**Instances:** 32,537")

st.sidebar.write("**Features:** 14")

st.sidebar.write("**Target:** Income")

st.sidebar.write("**Classes:**")
st.sidebar.write("• <=50K")
st.sidebar.write("• >50K")

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
st.markdown("""
<style>

/* Predict Button */
.stButton > button {
    background-color: #4da3ff;
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 10px;
    border: none;

    width: 400px;
    height: 55px;

    display: block;
    margin: auto;

    transition: 0.3s;
}

.stButton > button:hover {
    background-color: #2d8cff;
    color: white;
}

</style>
""", unsafe_allow_html=True)
# ==========================================
# Predict Button
# ==========================================

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict = st.button(
        " Predict Income",
        type="primary",
        use_container_width=True
    )

# ==========================================
# Prediction
# ==========================================

if predict:

    # Select model
    if model_name == "Logistic Regression":
        prediction = log_model.predict(scaler.transform(input_data))[0]
        probs = log_model.predict_proba(scaler.transform(input_data))[0]

    elif model_name == "k-Nearest Neighbors":
        prediction = knn_model.predict(scaler.transform(input_data))[0]
        probs = knn_model.predict_proba(scaler.transform(input_data))[0]

    elif model_name == "Decision Tree":
        prediction = dt_model.predict(input_data)[0]
        probs = dt_model.predict_proba(input_data)[0]

    elif model_name == "Naive Bayes":
        prediction = nb_model.predict(input_data)[0]
        probs = nb_model.predict_proba(input_data)[0]

    else:
        prediction = rf_model.predict(input_data)[0]
        probs = rf_model.predict_proba(input_data)[0]

    confidence = max(probs) * 100

    st.markdown("---")
    st.markdown("## Prediction Result")

    st.progress(confidence / 100)

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2f}%"
    )

    if prediction == 1:

        st.markdown("""
        <div style="
            background-color:#d4edda;
            border:3px solid green;
            padding:30px;
            border-radius:15px;
            text-align:center;
            margin-top:15px;">
            <h1 style="color:green;font-size:42px;">
             Income Greater than 50K
            </h1>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
            background-color:#d1ecf1;
            border:3px solid #0c5460;
            padding:30px;
            border-radius:15px;
            text-align:center;
            margin-top:15px;">
            <h1 style="color:#0c5460;font-size:42px;">
             Income Less than or Equal to 50K
            </h1>
        </div>
        """, unsafe_allow_html=True)
# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.caption("Developed by Sowmiya S - 2025AC05651 | BITS Pilani M.Tech AI & ML")