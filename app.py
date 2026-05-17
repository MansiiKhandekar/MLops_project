import streamlit as st
import joblib
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")

# --- UI ELEMENTS ---
st.title("🩺 Diabetes Risk Predictor")
st.markdown("""
This simple application uses a Machine Learning model to predict whether a patient is at risk of diabetes based on their health metrics.
Adjust the sliders below to see the prediction change!
""")

# --- INPUT SLIDERS ---
st.header("Patient Health Metrics")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.slider("Glucose Level", min_value=0, max_value=200, value=117, step=1)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=130, value=72, step=1)
    skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=23, step=1)

with col2:
    insulin = st.slider("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=30, step=1)
    bmi = st.slider("BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1)
    dpf = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.37, step=0.01)
    age = st.slider("Age", min_value=21, max_value=100, value=29, step=1)

# --- PREDICTION LOGIC ---
if st.button("Predict Risk", type="primary"):
    # Load the model
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found! Please ensure you have trained the model and it's saved in the 'models' directory.")
    else:
        # Load the saved Random Forest model
        model = joblib.load(model_path)
        
        # Prepare the input data as a 2D array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # Target classes mapping
        target_names = ['Negative (No Diabetes)', 'Positive (Diabetes Risk)']
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_probs = model.predict_proba(input_data)[0]
        
        # --- DISPLAY RESULTS ---
        if prediction == 1:
            st.error(f"### Predicted Outcome: **{target_names[prediction]}**")
        else:
            st.success(f"### Predicted Outcome: **{target_names[prediction]}**")
        
        # Display probabilities using columns for a nice layout
        st.subheader("Prediction Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        prob_col1.metric(label="Negative", value=f"{prediction_probs[0]*100:.1f}%")
        prob_col2.metric(label="Positive", value=f"{prediction_probs[1]*100:.1f}%")
        
        st.info("💡 **Tip:** Change the source code or data, push to GitHub, let the CI/CD pipeline retrain the model, and watch these probabilities change for the same input!")
