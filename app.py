# import libary that use
import streamlit as st
import pandas as pd
import pickle

# load model and feature list
with open("loan_approval_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("model_features.pkl", "rb") as feature_file:
    feature_names = pickle.load(feature_file)

# define function for prediction
def predict_loan_approval(model, new_data, features=feature_names, threshold=0.7):
    required_features = list(features)
    if 'TotalIncome' in required_features and 'TotalIncome' not in new_data.columns:
        new_data['TotalIncome'] = new_data['ApplicantIncome'] + new_data['CoapplicantIncome']
    new_data_selected = new_data[required_features]
    probability = model.predict_proba(new_data_selected)[:, 1]
    prediction = (probability >= threshold).astype(int)

    return prediction, probability

st.title("Loan Approval Prediction 💰🔮")
st.write("---")

st.write("Fill in the values below:")

# setting option for mapping option and value
gender_option = {"Female":0, "Male":1}
married_option = {"Not Married":0, "Married":1}
dependents_option = {"0":0, "1":1, "2":2, "3+":3}
education_option = {"Not Graduate":0, "Graduated":1}
self_employed_option = {"No":0, "Yes":1}
credit_history_option = {"Bad":0, "Good":1}
property_area_option = {"Urban":0, "Rural":1, "Semiurban":2}

# input value area
cols = st.columns(5)
gender = cols[0].selectbox("Gender", list(gender_option))
married = cols[1].selectbox("Married Status", list(married_option))
dependents= cols[2].selectbox("Dependents", list(dependents_option))
education = cols[3].selectbox("Education", list(education_option))
self_employed = cols[4].selectbox("Self Employed", list(self_employed_option))

applicant_income = st.number_input("Applicant Income", min_value=0, step=1000)
co_applicant_income = st.number_input("Co-Applicant Income", min_value=0, step=1000)
loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0, step=120, value=360)
cols = st.columns(2)
credit_history = cols[0].selectbox("Credit history", list(credit_history_option))
property_area = cols[1].selectbox("Property Area", list(property_area_option))
total_income = st.number_input("Total Income", value=applicant_income+co_applicant_income, disabled=True)

threshold = st.number_input("Threshold", value=0.7, min_value=0.0, max_value=1.0, step=0.01)

# make the value that input to df
data = pd.DataFrame({
    'Gender': gender_option.get(gender),
    'Married': married_option.get(married),
    'Dependents': dependents_option.get(dependents),
    'Education': education_option.get(education), 
    'Self_Employed': self_employed_option.get(self_employed),
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [co_applicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': credit_history_option.get(credit_history),
    'Property_Area': property_area_option.get(property_area),
    'TotalIncome': [total_income]
})

st.write("---")

# predict
if st.button("Predict"):
    try:
        missing_features = set(feature_names) - set(data.columns)
        if missing_features:
            print(f"Warning: Sample application is missing features: {missing_features}")

        prediction, probability = predict_loan_approval(model, data, feature_names, threshold)
        if prediction == 1:
            st.markdown("### ✅ **Loan Approved**", unsafe_allow_html=True)
        else:
            st.markdown("### ❌ **Loan Not Approved**", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")