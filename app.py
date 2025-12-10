import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

st.title("Decision Tree Classifier Lab — Iris Dataset")

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
theory_tab, lab_tab = st.tabs(["📘 Theory", "🧪 Lab Activity"])

# -------------------------------------------------------------------------
# THEORY TAB
# -------------------------------------------------------------------------
with theory_tab:
    st.header("📘 Theory of Decision Tree Classifier")

    st.markdown("""
### 🌳 What is a Decision Tree?

A **Decision Tree** is a supervised machine learning algorithm used for **classification** and **regression**.  
It works by repeatedly splitting the dataset into subsets using rules/questions.

---

### 📌 Example of a Simple Decision Tree

petal length ≤ 2.45 → Setosa
petal length > 2.45
petal width ≤ 1.75 → Versicolor
petal width > 1.75 → Virginica


---

## 🧮 Splitting Criteria

Decision Trees choose the best feature to split based on:

---

### 1️⃣ **Gini Index**

\[
Gini = 1 - \sum_{i=1}^{n} (p_i)^2
\]

Where:  
- \(p_i\) = probability of class i at that node  
- Lower Gini = purer node  

---

### 2️⃣ **Entropy**

\[
Entropy = - \sum_{i=1}^{n} p_i \log_2(p_i)
\]

---

### 3️⃣ **Information Gain**

\[
IG = Entropy_{parent} - \sum \left( \frac{n_{child}}{n_{parent}} \times Entropy_{child} \right)
\]

Higher Information Gain = better feature to split on.

---

## 🎯 How does a Decision Tree make decisions?

1. Check all features  
2. Calculate impurity (Gini/Entropy)  
3. Pick the feature that improves purity the most  
4. Split the dataset  
5. Repeat until:
   - max_depth reached  
   - node is pure  
   - no further gain  

---

## ⚠️ Overfitting

Deep trees memorize training data.

Solutions:
- Limit depth  
- Minimum samples per leaf  
- Pruning  

---

## 🌍 Real-Life Uses
- Medical diagnosis  
- Fraud detection  
- Customer segmentation  
- Weather prediction  
""")

# -------------------------------------------------------------------------
# LAB TAB
# -------------------------------------------------------------------------
with lab_tab:

    st.header("🧪 Decision Tree Classifier — Iris Dataset")

    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    # Show dataset
    st.subheader("📄 Iris Dataset")
    st.dataframe(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create model
    model = DecisionTreeClassifier(criterion="gini", random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    st.subheader("📊 Model Accuracy")
    st.write(f"**Accuracy:** {acc:.3f}")

    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g",
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # Classification Report
    st.subheader("📑 Classification Report")
    st.text(classification_report(y_test, y_pred, target_names=iris.target_names))

    # ------------------------------
    # Text Tree
    # ------------------------------
    st.subheader("🌳 Text-based Decision Tree")
    rules = export_text(model, feature_names=list(X.columns))
    st.text(rules)

    # ------------------------------
    # Graphical Tree
    # ------------------------------
    st.subheader("🖼️ Graphical Decision Tree")

    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    plot_tree(model, feature_names=iris.feature_names,
              class_names=iris.target_names, filled=True)
    st.pyplot(fig_tree)

    # ------------------------------
    # Prediction Section
    # ------------------------------
    st.subheader("🔮 Predict Flower Type")

    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

    if st.button("Predict"):
        user_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = model.predict(user_data)[0]
        st.success(f"🌸 Predicted Class: **{iris.target_names[prediction]}**")
