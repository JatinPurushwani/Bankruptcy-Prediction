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
# Streamlit UI
# =============================
st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")
st.title("Bankruptcy Prediction App")
st.write("Enter the financial ratios below to estimate bankruptcy risk.")

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

input_df = pd.DataFrame([input_data])

# =============================
# Prediction Logic (Threshold = 0.30)
# =============================

THRESHOLD = 0.30

if st.button("Predict Bankruptcy"):
    scaled_input = scaler.transform(input_df)

    probability = model.predict_proba(scaled_input)[0][1]
    prediction = int(probability > THRESHOLD)

    st.subheader("Results")
    st.write(f"**Bankruptcy Probability:** `{probability:.4f}`")
    st.write(f"**Decision Threshold:** `{THRESHOLD}`")

    if prediction == 1:
        st.error("⚠ High Risk of Bankruptcy (Predicted Class = 1)")
    else:
        st.success("✔ Low Risk of Bankruptcy (Predicted Class = 0)")

# =============================
# Show Processed Input
# =============================
st.subheader("Processed Input Data")
st.dataframe(input_df)

