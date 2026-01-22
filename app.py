import streamlit as st
import joblib
import numpy as np
import os

# Page Config
st.set_page_config(page_title="Titanic Predictor", page_icon="🚢")

# Title and Info
st.title("🚢 Titanic Survival Prediction")
st.write("Name: ISHOLA OLUFEMI | Matric: 22H032024")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'titanic_survival_model.pkl')
    return joblib.load(model_path)

try:
    model = load_model()
    st.success("System Status: Model Loaded Successfully")
except:
    st.error("Error: Model not found. Please upload 'titanic_survival_model.pkl' to the 'model' folder.")

# User Inputs
st.subheader("Passenger Details")
pclass = st.selectbox("Ticket Class", [1, 2, 3], format_func=lambda x: f"{x}st Class" if x==1 else(f"{x}nd Class" if x==2 else f"{x}rd Class"))
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
fare = st.number_input("Fare Price ($)", min_value=0.0, value=30.0)

# Convert Inputs for Model
sex_encoded = 0 if sex == "Male" else 1

# Predict Button
if st.button("Predict Survival Chance"):
    features = np.array([[pclass, sex_encoded, age, sibsp, fare]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][prediction]
    
    if prediction == 1:
        st.balloons()
        st.success(f"Prediction: SURVIVED (Confidence: {round(probability*100, 2)}%)")
    else:
        st.error(f"Prediction: DID NOT SURVIVE (Confidence: {round(probability*100, 2)}%)")
