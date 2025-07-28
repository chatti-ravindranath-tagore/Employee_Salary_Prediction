import streamlit as st
import joblib
import pandas as pd
model=joblib.load("prediction_model.pkl")
st.set_page_config(page_title="Employee Salary Prediction",page_icon="💰",layout="centered")
st.title("Employee Salary Prediction")
st.markdown("Predict weatheer an employee will earn more than 50k or not")
st.sidebar.header("Employee Details")
age=st.sidebar.slider("AGE",18,65,30)
education=st.sidebar.selectbox("Education-Level",["Bachelor","Masters","P.hD","HS-Graduate","Associative-P","Some-College"])
occupation=st.sidebar.selectbox("Job-Role",[
    "Tech-support","Craft-repair","Other-service","Sales",
    "Exec-managerial","Prof-specialty","Handlers-cleaners","Machine-op-inspct",
    "Adm-clerical","Farming-fishing","Transport-moving","Priv-house-serv",
    "Protective-serv","Armed-Forces"
])
hours_per_week=st.sidebar.slider("Hours Per Week",0,100,40)
experience=st.sidebar.slider("Years Of Experience",0,50,5)

# DataFrame
input_df=pd.DataFrame({
    "age":[age],
    "education":[education],
    "occupation":[occupation],
    "hours-per-week":[hours_per_week],
    "experience":[experience]
})

st.write("### INPUT DETAILS")
st.table(input_df)

#predict button
if st.button("PREDICT"):
  prediction=model.predict(input_df)
  st.success(f"Prediction: {prediction[0]}")

#Batch Prediction
st.markdown("-----")
st.markdown("### BATCH PREDICTION")
uploaded_file=st.file_uploader("Upload a CSV file",type=["csv"])

if uploaded_file is not None:
 batch_data=pd.read_csv(uploaded_file)
 st.write("Data preview:",batch_data.head())
 batch_preds=model.predict(batch_data)
 batch_data["Predicted Income"]=batch_preds
 st.write("Batch Predictions:")
 st.write(batch_data.head())
 csv=batch_data.to_csv(index=False).encode("utf-8")
 st.download_button("Download Predictions",data=csv,file_name="batch_predictions.csv",mime="text/csv")
