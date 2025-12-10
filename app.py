import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

st.title("Decision Tree Classifier Lab — Weather Dataset (Categorical)")

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
theory_tab, lab_tab = st.tabs(["📘 Theory", "🧪 Lab Activity"])

# -------------------------------------------------------------------------
# THEORY TAB
# -------------------------------------------------------------------------
with theory_tab:
    st.header("📘 Theory of Decision Trees (Categorical Data)")

    st.markdown("""
### 🌳 What is a Decision Tree?

A **Decision Tree** is a flowchart-like structure used for classification.  
It works by repeatedly asking questions and splitting the dataset.

Example:
If Outlook = Sunny AND Humidity = High → Play = No
If Outlook = Overcast → Always Play
If Windy = True → May Not Play


---

## 🧮 Splitting Criteria

### 1️⃣ Gini Index  
\[
Gini = 1 - \sum_{i=1}^{n} (p_i)^2
\]

### 2️⃣ Entropy  
\[
Entropy = - \sum_{i=1}^{n} p_i \log_2(p_i)
\]

### 3️⃣ Information Gain  
\[
IG = Entropy_{parent} - \sum \frac{n_{child}}{n_{parent}} Entropy_{child}
\]

---

## 🎯 Why Decision Trees?

- Easy to visualize  
- Works with categorical data  
- Requires little preprocessing  
- Students can interpret rules easily  

---

## 🌦️ Real-Life Examples

- Predicting rain  
- Whether a person will buy a product  
- Approving a bank loan  
- Diagnosing a disease  
""")

# -------------------------------------------------------------------------
# LAB TAB (CSV Upload Required)
# -------------------------------------------------------------------------
with lab_tab:

    st.header("🧪 Upload Weather Dataset (Categorical)")

    uploaded_file = st.file_uploader("Upload CSV containing Outlook, Temperature, Humidity, Windy, Play", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Clean column names
        df.columns = df.columns.str.strip().str.lower()

        required_cols = ["outlook", "temperature", "humidity", "windy", "play"]

        if not all(col in df.columns for col in required_cols):
            st.error("❌ CSV must contain these columns: Outlook, Temperature, Humidity, Windy, Play")
            st.stop()

        st.subheader("📄 Uploaded Dataset")
        st.dataframe(df)

        # Encode categorical columns
        encoder = LabelEncoder()
        encoded_df = df.copy()
        for col in required_cols:
            encoded_df[col] = encoder.fit_transform(encoded_df[col])

        X = encoded_df[["outlook", "temperature", "humidity", "windy"]]
        y = encoded_df["play"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = DecisionTreeClassifier(criterion="entropy", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        st.subheader("📊 Model Accuracy")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")

        # Confusion Matrix
        st.subheader("🔢 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Greens", fmt="g")
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # Classification Report
        st.subheader("📑 Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Text tree
        st.subheader("🌳 Text-based Decision Tree")
        rules = export_text(model, feature_names=list(X.columns))
        st.text(rules)

        # Graphical tree
        st.subheader("🖼️ Graphical Decision Tree")
        fig_dt, ax_dt = plt.subplots(figsize=(18, 10))
        plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
        st.pyplot(fig_dt)

        # Prediction
        st.subheader("🔮 Predict Play (Yes/No)")

        outlook = st.selectbox("Outlook", df["outlook"].unique())
        temperature = st.selectbox("Temperature", df["temperature"].unique())
        humidity = st.selectbox("Humidity", df["humidity"].unique())
        windy = st.selectbox("Windy", df["windy"].unique())

        if st.button("Predict"):
            # Convert user-friendly values back to encoded
            user_df = pd.DataFrame({
                "outlook": [outlook],
                "temperature": [temperature],
                "humidity": [humidity],
                "windy": [windy]
            })

            # Encode user input using same encoder
            for col in user_df.columns:
                user_df[col] = LabelEncoder().fit(df[col]).transform(user_df[col])

            prediction = model.predict(user_df)[0]
            result = "Yes" if prediction == 1 else "No"

            st.success(f"🌤️ **Play? → {result}**")

    else:
        st.info("Please upload a CSV file to continue.")
