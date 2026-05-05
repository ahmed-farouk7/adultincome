import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved objects
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("💰 Income Prediction App (SVM)")
st.write("Enter individual details to predict if income exceeds $50K/year.")

# 2. UI for User Input
age = st.number_input("Age", min_value=17, max_value=90, value=30)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education_num = st.slider("Years of Education", 1, 16, 10)
marital = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
hours = st.number_input("Hours per week", min_value=1, max_value=99, value=40)

# 3. Process Input to match training format
if st.button("Predict"):
    # Create a raw dataframe with the input
    input_data = pd.DataFrame([{
        'age': age, 'education.num': education_num, 'hours.per.week': hours,
        'workclass': workclass, 'marital.status': marital, 'occupation': occupation,
        # Default values for other columns not in UI for simplicity
        'fnlwgt': 180000, 'capital.gain': 0, 'capital.loss': 0, 
        'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'native.country': 'United-States'
    }])

    # Apply Feature Engineering (Same as we did in training)
    input_data['long_hours'] = (input_data['hours.per.week'] > 40).astype(int)
    input_data['has_gain'] = (input_data['capital.gain'] > 0).astype(int)
    input_data['has_loss'] = (input_data['capital.loss'] > 0).astype(int)
    input_data['age_group'] = pd.cut(input_data['age'], bins=[0,25,40,60,100], labels=[0,1,2,3]).astype(int)

    # Encoding & Alignment
    input_encoded = pd.get_dummies(input_data)
    input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scaling & PCA
    input_scaled = scaler.transform(input_final)
    input_pca = pca.transform(input_scaled)

    # Prediction
    prediction = model.predict(input_pca)
    
    if prediction[0] == '>50K':
        st.success("Result: High Income (>50K)")
    else:
        st.warning("Result: Low Income (<=50K)")