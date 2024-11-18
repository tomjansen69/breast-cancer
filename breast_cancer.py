import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import pickle
from sklearn.inspection import PartialDependenceDisplay

# Load your data
data = pd.read_csv('breast_cancer.csv')

# Split data
X = data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean']]
y = data['diagnosis'].map({'M': 1, 'B': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Streamlit form for user input
st.title("Breast Cancer Prediction")
st.write("Enter the following features:")

# User input fields
radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=100.0, value=10.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=100.0, value=10.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=500.0, value=100.0)
area_mean = st.number_input("Area Mean", min_value=0.0, max_value=5000.0, value=1000.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.1)
compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.1)

# Prepare user input for prediction
user_input = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean]])

# Make the prediction when the user presses a button
if st.button("Predict"):
    prediction = pipeline.predict(user_input)
    if prediction == 1:
        st.write("The prediction is: Malignant (Cancerous)")
    else:
        st.write("The prediction is: Benign (Non-cancerous)")

    # Model evaluation - Confusion Matrix and ROC Curve
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, thresholds = roc_curve(y_test, pipeline.predict_proba(X_test)[:, 1])
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
