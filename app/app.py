import os
import streamlit as st
import pickle
import numpy as np

def preprocess_input(input_list, scaler):
    array = np.array(input_list).reshape(1, -1)
    scaled = scaler.transform(array)
    return scaled
#from app.utils import preprocess_input

# Load model and scaler

model_path = os.path.join(os.path.dirname(__file__), '../models/logistic_model.pkl')

with open(model_path, 'rb') as f:
    #model = pickle.load(f)
    model = pickle.load(open('/Diabetes_Predictor_MVP/models/logistic_model.pkl', 'rb'))
    scaler = pickle.load(open('/Diabetes_Predictor_MVP/models/scaler.pkl', 'rb'))

st.title("🩺 Diabetes Prediction App")

st.markdown("""
Enter the following medical attributes to assess the likelihood of diabetes:
""")

# Input fields
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.slider('Glucose Level', 0, 200, 120)
blood_pressure = st.slider('Blood Pressure', 0, 122, 70)
skin_thickness = st.slider('Skin Thickness', 0, 100, 20)
insulin = st.slider('Insulin Level', 0, 846, 79)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
diabetes_pedigree = st.number_input('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
age = st.slider('Age', 10, 100, 33)

if st.button('Predict'):
    input_data = preprocess_input([pregnancies, glucose, blood_pressure,
                                   skin_thickness, insulin, bmi,
                                   diabetes_pedigree, age], scaler)
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"🚨 High risk of diabetes (probability: {prob:.2f})")
    else:
        st.success(f"✅ Low risk of diabetes (probability: {prob:.2f})")