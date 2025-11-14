import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# Load Model, Scaler, Features
# =============================

with open("models/model.pkl", "rb") as f:
    model, scaler, feature_list = pickle.load(f)

# =============================
# App Title
# =============================

st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")
st.title("Bankruptcy Prediction App")
st.write("Enter financial ratios to predict bankruptcy risk.")


# =============================
# Input Form
# =============================

st.subheader("Input Financial Ratios")

input_data = {}

for feature in feature_list:
    input_data[feature] = st.number_input(
        feature,
        value=0.0,
        format="%.6f"
    )

# Convert to DataFrame
input_df = pd.DataFrame([input_data])


# =============================
# Prediction
# =============================

if st.button("Predict Bankruptcy"):
    # Scale input
    scaled_input = scaler.transform(input_df)

    # Predict probability
    probability = model.predict_proba(scaled_input)[0][1]
    prediction = model.predict(scaled_input)[0]

    st.subheader("Results")
    st.write(f"**Bankruptcy Probability:** `{probability:.4f}`")

    if prediction == 1:
        st.error("⚠ High Risk of Bankruptcy (Class = 1)")
    else:
