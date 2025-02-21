import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.title("Credit Card Fraud Detection")

# Load trained model
@st.cache_data
def load_model():
    with open("xgb_model.sav", "rb") as file:
        model = pickle.load(file)
        return model

mod = load_model()

# Input fields for user
st.sidebar.header("Enter Transaction Details")
features = []
for i in range(1, 29):
    features.append(st.sidebar.number_input(f"Feature {i}", value=0.0))
amount = st.sidebar.number_input("Transaction Amount", value=0.0)
features.append(amount)

data = np.array([features])

# Make prediction
if st.sidebar.button("Predict"):
    prediction = mod.predict(data)[0]
    if prediction == 1:
        st.error("Fraudulent Transaction Detected!")
    else:
        st.success("Legitimate Transaction")
