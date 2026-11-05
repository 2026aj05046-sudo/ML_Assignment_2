import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Classification_Assignment2")
st.write("Machine Learning Assignment 2")
st.warning("This is my assignment | BITS ID 2026aj05046")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

@st.cache_resource
def load_model(filename):
    return joblib.load(BASE_DIR / "model" / filename)

@st.cache_data
def load_features():
    df = pd.read_csv(BASE_DIR / "breast_cancer_dataset.csv", nrows=1)
    return df.drop(columns=["diagnosis"]).columns.tolist()

feature_names = load_features()

st.subheader("Step 1: Upload Test CSV")
uploaded_file = st.file_uploader(
    "Upload labeled test_data.csv",
    type=["csv"]
)

st.subheader("Step 2: Select Model")
selected_model = st.selectbox(
    "Choose a trained model",
    list(MODEL_FILES.keys())
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required = feature_names + ["diagnosis"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        if df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

        X = df[feature_names].apply(
            pd.to_numeric, errors="coerce"
        )

        if X.isna().any().any():
            st.error("Feature columns must contain valid numbers.")
            st.stop()

        mapping = {"B": 0, "M": 1, "0": 0, "1": 1}
        y = (
            df["diagnosis"]
            .astype(str)
            .str.strip()
            .map(mapping)
        )

        if y.isna().any():
            st.error("Diagnosis must be B/M or 0/1.")
            st.stop()

        y = y.astype(int)

        model = load_model(MODEL_FILES[selected_model])
        predictions = model.predict(X)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X)[:, positive_index]

        scores = {
            "Accuracy": accuracy_score(y, predictions),
            "AUC": (
                roc_auc_score(y, probabilities)
                if y.nunique() == 2 else None
            ),
            "Precision": precision_score(
                y, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y, predictions, zero_division=0
            ),
            "F1": f1_score(
                y, predictions, zero_division=0
            ),
            "MCC": matthews_corrcoef(y, predictions)
        }

        st.success(
            f"Successfully evaluated {len(df)} records."
        )

        st.subheader("Step 3: Evaluation Metrics")
        cols = st.columns(3)

        for i, (name, value) in enumerate(scores.items()):
            formatted = "N/A" if value is None else f"{value:.4f}"
            cols[i % 3].metric(name, formatted)

        st.subheader("Step 4: Confusion Matrix")
        cm = confusion_matrix(y, predictions, labels=[0, 1])

        st.dataframe(pd.DataFrame(
            cm,
            index=["Actual Benign", "Actual Malignant"],
            columns=["Predicted Benign", "Predicted Malignant"]
        ))

        st.subheader("Step 5: Classification Report")
        report = classification_report(
            y, predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose())

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload test_data.csv to begin evaluation.")
