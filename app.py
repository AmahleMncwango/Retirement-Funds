import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Retirement Fund Analysis Dashboard",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Machine Learning for Retirement Fund Analysis")
st.markdown("""
This dashboard explores South Africa’s **Two-Pot Retirement System** using demographic and employment data.  
Upload your dataset to view insights and run predictive models.
---
""")

# ----------------------------
# DATA UPLOAD
# ----------------------------
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

    # ----------------------------
    # DATA OVERVIEW
    # ----------------------------
    st.subheader("📊 Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    with st.expander("View Summary Statistics"):
        st.write(df.describe(include="all"))

    # ----------------------------
    # DATA VISUALISATION
    # ----------------------------
    st.subheader("📈 Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        selected_num_col = st.selectbox("Select numerical column for distribution plot", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[selected_num_col], kde=True, ax=ax)
        st.pyplot(fig)

    if categorical_cols:
        selected_cat_col = st.selectbox("Select categorical column for count plot", categorical_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=df[selected_cat_col], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ----------------------------
    # CORRELATION MATRIX
    # ----------------------------
    if len(numeric_cols) > 1:
        st.subheader("🔗 Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ----------------------------
    # MACHINE LEARNING (OPTIONAL)
    # ----------------------------
    st.subheader("🤖 Predictive Modelling")

    target_col = st.selectbox("Select target column (for prediction)", options=df.columns)
    feature_cols = st.multiselect("Select feature columns", [c for c in df.columns if c != target_col])

    if feature_cols and target_col:
        X = df[feature_cols]
        y = df[target_col]

        # Encode categorical targets
        if y.dtype == "object":
            y = pd.factorize(y)[0]

        # Handle categorical predictors
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric(label="Model Accuracy", value=f"{acc:.2%}")

        with st.expander("View Detailed Report"):
            st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
