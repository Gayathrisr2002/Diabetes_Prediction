
import streamlit as st
import pickle
import numpy as np

# Load the trained model
load_model = pickle.load(open('D:/SYSTEM DATA/Documents/GAYATHRI/ML_project/model.sav', 'rb'))

def diabetes_pred(input_data):
    input_data = np.asarray(input_data, dtype=float)  
    input_reshape = input_data.reshape(1, -1)
    prediction = load_model.predict(input_reshape)
    return prediction

def main():
    st.title("Diabetes Prediction Web App")
    
    pregnancies = st.number_input("NO. of pregnancies", min_value=0, max_value=20, step=1, format="%d")
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1, format="%d")
    diastolic = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1, format="%d")
    triceps = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1, format="%d")
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1, format="%d")
    bmi = st.number_input("BMI value", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
    dpf = st.number_input("Diabetes pedigree function value", min_value=0.0, max_value=3.0, step=0.001, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=150, step=1, format="%d")

    if st.button("Predict"):
        diagnosis = diabetes_pred([pregnancies, glucose, diastolic, triceps, insulin, bmi, dpf, age])
        if diagnosis[0] == 1:
            st.success("The person is diabetic")
        else:
            st.success("The person is not diabetic")

if __name__ == "__main__":
    main()
