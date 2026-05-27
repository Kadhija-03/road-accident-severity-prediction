# Road Accident Severity Prediction using Machine Learning

## Overview
This project predicts road accident severity using Machine Learning techniques based on accident-related factors such as driver details, vehicle information, collision type, and environmental conditions.

The system classifies accident severity into:
- Fatal Injury
- Serious Injury
- Slight Injury

An interactive Streamlit web application is developed to provide real-time accident severity prediction.

---

## Features
- Predicts accident severity using historical accident data
- Interactive Streamlit web interface
- Real-time prediction with probability scores
- User-friendly accident detail input system
- Machine Learning pipeline integration

---

## Technologies Used
- Python
- Machine Learning
- Streamlit
- Pandas
- Scikit-learn
- Joblib

---

## Dataset
The model is trained using a Road Traffic Accident dataset containing:
- Driver information
- Vehicle details
- Casualty information
- Collision type
- Environmental conditions

---

## Input Features
- Day of week
- Number of vehicles involved
- Number of casualties
- Area of accident
- Driver age and gender
- Educational level
- Type of vehicle
- Driving experience
- Type of collision
- Cause of accident
- Hour of day

---

## Project Structure

```text
road-accident-severity-prediction/
│
├── app_streamlit.py
├── rta_pipeline.joblib
├── RTA Dataset.csv
├── requirements.txt
├── README.md
└── Road Accident Severity Predictor.pdf
