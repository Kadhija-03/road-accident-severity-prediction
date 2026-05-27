import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Road Accident Severity Predictor", layout="centered")

st.title("Road Accident Severity Predictor")
st.write("Fill in accident details and get predicted severity (Fatal / Serious / Slight).")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("rta_pipeline.joblib")

pipeline = load_model()

# Feature Inputs
st.sidebar.header("Input Accident Details")

Day_of_week = st.sidebar.selectbox(
    "Day of week",
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

Number_of_vehicles_involved = st.sidebar.number_input("Number of vehicles involved", min_value=1, step=1)
Number_of_casualties = st.sidebar.number_input("Number of casualties", min_value=0, step=1)

Area_accident_occured = st.sidebar.text_input("Area accident occurred", "Urban")
Types_of_Junction = st.sidebar.text_input("Type of Junction", "No junction")

Age_band_of_driver = st.sidebar.selectbox("Age band of driver",
                                          ["<18","18-30","31-50","51-70",">70"])
Sex_of_driver = st.sidebar.selectbox("Sex of driver", ["Male","Female","Unknown"])
Educational_level = st.sidebar.text_input("Educational level", "High school")
Vehicle_driver_relation = st.sidebar.text_input("Vehicle driver relation", "Employee")
Type_of_vehicle = st.sidebar.text_input("Type of vehicle", "Car")
Driving_experience = st.sidebar.text_input("Driving experience (e.g. 1-2yr)", "1-2yr")
Service_year_of_vehicle = st.sidebar.text_input("Service year of vehicle", "Above 10yr")
Type_of_collision = st.sidebar.text_input("Type of collision", "Rear-end")

Sex_of_casualty = st.sidebar.selectbox("Sex of casualty", ["Male","Female","Unknown"])
Age_band_of_casualty = st.sidebar.selectbox("Age band of casualty",
                                            ["<18","18-30","31-50","51-70",">70"])
Cause_of_accident = st.sidebar.text_input("Cause of accident", "Overspeeding")
Hour_of_Day = st.sidebar.slider("Hour of day", 0, 23, value=12)

# Prepare input
input_data = pd.DataFrame([{
    "Day_of_week": Day_of_week,
    "Number_of_vehicles_involved": Number_of_vehicles_involved,
    "Number_of_casualties": Number_of_casualties,
    "Area_accident_occured": Area_accident_occured,
    "Types_of_Junction": Types_of_Junction,
    "Age_band_of_driver": Age_band_of_driver,
    "Sex_of_driver": Sex_of_driver,
    "Educational_level": Educational_level,
    "Vehicle_driver_relation": Vehicle_driver_relation,
    "Type_of_vehicle": Type_of_vehicle,
    "Driving_experience": Driving_experience,
    "Service_year_of_vehicle": Service_year_of_vehicle,
    "Type_of_collision": Type_of_collision,
    "Sex_of_casualty": Sex_of_casualty,
    "Age_band_of_casualty": Age_band_of_casualty,
    "Cause_of_accident": Cause_of_accident,
    "Hour_of_Day": Hour_of_Day,
}])

st.subheader("Preview Input:")
st.write(input_data)

# Predict
if st.button("Predict Severity"):
    prediction = pipeline.predict(input_data)[0]
    st.success(f"Predicted Severity: **{prediction}**")

    probs = pipeline.predict_proba(input_data)[0]
    st.write("Probability Breakdown:")
    for cls, p in zip(pipeline.classes_, probs):
        st.write(f"{cls}: {p:.3f}")