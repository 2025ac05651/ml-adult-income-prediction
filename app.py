import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score
)

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
# Model Metrics - Baseline / Experiment Results
# ==========================================

metrics = {
    "Logistic Regression": {
        "Accuracy": "82.39%",
        "AUC": "84.97%",
        "Precision": "71.40%",
        "Recall": "44.90%",
        "F1": "55.13%",
        "MCC": "46.74%"
    },

    "Decision Tree": {
        "Accuracy": "80.26%",
        "AUC": "73.13%",
        "Precision": "58.96%",
        "Recall": "59.38%",
        "F1": "59.17%",
        "MCC": "46.15%"
    },

    "k-Nearest Neighbors": {
        "Accuracy": "82.39%",
        "AUC": "84.40%",
        "Precision": "65.18%",
        "Recall": "57.78%",
        "F1": "61.26%",
        "MCC": "50.07%"
    },

    "Naive Bayes": {
        "Accuracy": "79.15%",
        "AUC": "82.74%",
        "Precision": "65.01%",
        "Recall": "29.15%",
        "F1": "40.25%",
        "MCC": "33.29%"
    },

    "Random Forest": {
        "Accuracy": "85.60%",
        "AUC": "89.77%",
        "Precision": "74.25%",
        "Recall": "61.61%",
        "F1": "67.34%",
        "MCC": "58.63%"
    }
}

m = metrics[model_name]

st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance")

st.sidebar.caption("Baseline results from the model experiment")

st.sidebar.metric("Accuracy", m["Accuracy"])
st.sidebar.metric("AUC", m["AUC"])
st.sidebar.metric("Precision", m["Precision"])
st.sidebar.metric("Recall", m["Recall"])
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
# Model Selection Helper
# ==========================================

def get_selected_model(name):
    """Return the trained model corresponding to the selected model name."""
    if name == "Logistic Regression":
        return log_model
    elif name == "Decision Tree":
        return dt_model
    elif name == "k-Nearest Neighbors":
        return knn_model
    elif name == "Naive Bayes":
        return nb_model
    else:
        return rf_model


def prepare_features_for_model(X, selected_model_name):
    """Apply the same preprocessing used during model training."""

    if selected_model_name in ["Logistic Regression", "k-Nearest Neighbors"]:
        return scaler.transform(X)

    return X

# ==========================================
# Upload Test Dataset
# ==========================================

st.markdown("---")
st.header("📊 Model Evaluation on Uploaded Test Dataset")
st.write(
    "Upload a test dataset (CSV) containing the target column "
    "**income** to evaluate the selected model. "
    "The evaluation is performed dynamically on the uploaded data "
    "and displays all six required metrics, a classification report, "
    "and a confusion matrix."
)

uploaded_file = st.file_uploader(
    "Upload Test Data (.csv)",
    type=["csv"]
)

test_df = None

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)

        st.success("Test dataset uploaded successfully!")

        st.write(f"**Test records uploaded:** {len(test_df):,}")
        st.dataframe(test_df.head(), use_container_width=True)

    except Exception as e:
        st.error(f"Unable to read the uploaded CSV file: {e}")
        test_df = None
# ==========================================
# Dynamic Test Dataset Evaluation
# ==========================================

if test_df is not None:

    st.markdown("---")

    if "income" not in test_df.columns:

        st.error(
            "The uploaded CSV must contain the target column "
            "'income'. Please upload the encoded test dataset "
            "with the target column included."
        )

    else:

        # Expected feature order used during training
        expected_features = [
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
        ]

        missing_features = [
            column
            for column in expected_features
            if column not in test_df.columns
        ]

        if missing_features:

            st.error(
                "The uploaded dataset is missing the following "
                f"required feature(s): {', '.join(missing_features)}"
            )

        else:

            try:

                # Keep exactly the 14 features in the same order
                X_test = test_df[expected_features].copy()
                y_test = test_df["income"]

                # ==================================================
                # ALL FIVE MODEL COMPARISON
                # ==================================================

                st.subheader("🏆 Model Comparison on Test Dataset")

                st.write(
                    "The following table shows the performance of all "
                    "five machine learning models on the uploaded test dataset."
                )

                models = {
                    "Logistic Regression": log_model,
                    "Decision Tree": dt_model,
                    "k-Nearest Neighbors": knn_model,
                    "Naive Bayes": nb_model,
                    "Random Forest": rf_model
                }

                comparison_results = []

                for name, model in models.items():

                    # Apply scaling only to models that require it
                    X_model = prepare_features_for_model(
                        X_test,
                        name
                    )

                    # Predictions
                    model_pred = model.predict(X_model)

                    # AUC
                    if hasattr(model, "predict_proba"):

                        model_prob = model.predict_proba(X_model)[:, 1]

                        try:
                            model_auc = roc_auc_score(
                                y_test,
                                model_prob
                            )
                        except ValueError:
                            model_auc = None

                    else:
                        model_auc = None

                    # Metrics
                    model_accuracy = accuracy_score(
                        y_test,
                        model_pred
                    )

                    model_precision = precision_score(
                        y_test,
                        model_pred,
                        zero_division=0
                    )

                    model_recall = recall_score(
                        y_test,
                        model_pred,
                        zero_division=0
                    )

                    model_f1 = f1_score(
                        y_test,
                        model_pred,
                        zero_division=0
                    )

                    model_mcc = matthews_corrcoef(
                        y_test,
                        model_pred
                    )

                    comparison_results.append({
                        "Model": name,
                        "Accuracy": model_accuracy,
                        "AUC": model_auc,
                        "Precision": model_precision,
                        "Recall": model_recall,
                        "F1 Score": model_f1,
                        "MCC": model_mcc
                    })

                # ==================================================
                # FIVE MODEL COMPARISON TABLE
                # ==================================================

                comparison_df = pd.DataFrame(comparison_results)

                # Find best model based on F1 Score
                best_model_row = comparison_df.loc[
                    comparison_df["F1 Score"].idxmax()
                ]

                best_model_name = best_model_row["Model"]

                # Create a separate dataframe for display
                display_df = comparison_df.copy()

                # Convert metrics to percentage format
                metric_columns = [
                    "Accuracy",
                    "AUC",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "MCC"
                ]

                for column in metric_columns:
                    display_df[column] = display_df[column].apply(
                        lambda x: f"{x * 100:.2f}%"
                        if pd.notna(x)
                        else "N/A"
                    )


                # Highlight the best-performing model
                def highlight_best(row):

                    if row["Model"] == best_model_name:
                        return [
                            "background-color: #d4edda; "
                            "font-weight: bold"
                        ] * len(row)

                    return [""] * len(row)


                st.dataframe(
                    display_df.style.apply(
                        highlight_best,
                        axis=1
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.success(
                    f"🏆 Best Performing Model on Uploaded Test Data: "
                    f"**{best_model_name}**"
                )

                st.caption(
                    "The best-performing model is identified based on the "
                    "highest F1 Score on the uploaded test dataset."
                )
                # ==================================================
                # SELECTED MODEL DETAILED EVALUATION
                # ==================================================

                st.markdown("---")

                st.subheader(
                    f"📈 Detailed Evaluation — {model_name}"
                )

                selected_model = get_selected_model(model_name)

                X_model = prepare_features_for_model(
                    X_test,
                    model_name
                )

                # Generate predictions
                y_pred = selected_model.predict(X_model)

                # Generate probabilities for AUC
                if hasattr(selected_model, "predict_proba"):

                    y_prob = selected_model.predict_proba(
                        X_model
                    )[:, 1]

                    try:
                        auc = roc_auc_score(
                            y_test,
                            y_prob
                        )
                    except ValueError:
                        auc = None

                else:
                    auc = None

                # Calculate metrics
                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

                precision = precision_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    y_pred,
                    zero_division=0
                )

                mcc = matthews_corrcoef(
                    y_test,
                    y_pred
                )

                # ==================================================
                # SELECTED MODEL METRICS
                # ==================================================

                st.subheader("Evaluation Metrics")

                if auc is not None:
                    auc_display = f"{auc:.4f}"
                else:
                    auc_display = "N/A"

                dynamic_metrics_df = pd.DataFrame({
                    "Metric": [
                        "Accuracy",
                        "AUC",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "MCC"
                    ],
                    "Value": [
                        f"{accuracy:.4f}",
                        auc_display,
                        f"{precision:.4f}",
                        f"{recall:.4f}",
                        f"{f1:.4f}",
                        f"{mcc:.4f}"
                    ]
                })

                st.table(dynamic_metrics_df)

                # ==================================================
                # CLASSIFICATION REPORT
                # ==================================================

                st.subheader("Classification Report")

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=[0, 1],
                    target_names=["<=50K", ">50K"],
                    output_dict=True,
                    zero_division=0
                )

                report_df = pd.DataFrame(report).transpose()

                # Rename summary rows for better readability
                report_df.rename(
                    index={
                        "macro avg": "Macro Average",
                        "weighted avg": "Weighted Average"
                    },
                    inplace=True
                )

                # Round numerical values
                report_df = report_df.round(4)

                st.dataframe(
                    report_df,
                    use_container_width=True
                )
                # ==================================================
                # CONFUSION MATRIX
                # ==================================================

                st.subheader("Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred
                )

                # Smaller figure
                fig, ax = plt.subplots(
                    figsize=(4.5, 3.2)
                )

                ax.imshow(
                    cm,
                    cmap="Blues"
                )

                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):

                        ax.text(
                            j,
                            i,
                            cm[i, j],
                            ha="center",
                            va="center",
                            fontsize=11
                        )

                ax.set_xlabel(
                    "Predicted",
                    fontsize=10
                )

                ax.set_ylabel(
                    "Actual",
                    fontsize=10
                )

                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])

                ax.set_xticklabels([
                    "<=50K",
                    ">50K"
                ])

                ax.set_yticklabels([
                    "<=50K",
                    ">50K"
                ])

                ax.set_title(
                    f"{model_name} - Test Dataset Confusion Matrix",
                    fontsize=12
                )

                # IMPORTANT:
                # False prevents Streamlit from stretching
                # the figure across the entire page.
                st.pyplot(
                    fig,
                    use_container_width=False
                )

                plt.close(fig)

                st.success(
                    f"Dynamic evaluation completed successfully for "
                    f"{len(test_df):,} test records using {model_name}."
                )

            except Exception as e:

                st.error(
                    "An error occurred while evaluating the uploaded "
                    f"test dataset: {e}"
                )
# ==========================================
# Single Income Prediction
# ==========================================

st.markdown("---")
st.header("🔍 Single Income Prediction")
st.write(
    "Enter the feature values below and click **Predict Income** "
    "to predict whether the annual income is **>50K** or **<=50K**."
)

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
# Single Prediction
# ==========================================

if predict:

    try:
        selected_model = get_selected_model(model_name)

        # Apply the same preprocessing used during training.
        input_model = prepare_features_for_model(
            input_data,
            model_name
        )
        prediction = selected_model.predict(input_model)[0]
        probs = selected_model.predict_proba(input_model)[0]

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

    except Exception as e:
        st.error(
            f"Unable to generate the individual prediction: {e}"
        )

# ==========================================
# Footer
# ==========================================

st.markdown("---")
st.caption(
    "Developed by Sowmiya S - 2025AC05651 | BITS Pilani M.Tech AI & ML"
)