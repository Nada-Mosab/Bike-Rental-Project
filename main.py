import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(page_title="Seoul Bike Sharing Demand Predictor", page_icon="🚲", layout="wide")

# Load Saved Models and Preprocessing Objects
@st.cache_resource
def load_assets():
    models_dir = "Models"
    lr_model = joblib.load(os.path.join(models_dir, "linear_regression_model.pkl"))
    rf_model = joblib.load(os.path.join(models_dir, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    features = joblib.load(os.path.join(models_dir, "model_features.pkl"))
    return lr_model, rf_model, scaler, features

lr_model, rf_model, scaler, model_features = load_assets()

# App Header
st.title("🚲 Seoul Bike Sharing Demand Prediction App")
st.markdown("This application predicts the number of bikes rented in Seoul based on environmental and temporal conditions. Built as part of the Regression Project.")

# Sidebar for User Inputs
st.sidebar.header("Input Features")

def user_input_features():
    temperature = st.sidebar.slider("Temperature (°C)", -20.0, 40.0, 20.0)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
    wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 10.0, 1.5)
    visibility = st.sidebar.slider("Visibility (10m)", 0, 2000, 1000)
    dew_point = st.sidebar.slider("Dew Point Temperature (°C)", -30.0, 30.0, 10.0)
    solar_radiation = st.sidebar.slider("Solar Radiation (MJ/m²)", 0.0, 3.5, 0.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 35.0, 0.0)
    snowfall = st.sidebar.slider("Snowfall (cm)", 0.0, 9.0, 0.0)
    hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
    
    season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    holiday = st.sidebar.selectbox("Holiday Status", ["No Holiday", "Holiday"])
    functioning = st.sidebar.selectbox("Functioning Day", ["Yes", "No"])
    
    # Create a dictionary for inputs
    data = {
        'Hour': hour,
        'Temperature(°C)': temperature,
        'Humidity(%)': humidity,
        'Wind speed (m/s)': wind_speed,
        'Visibility (10m)': visibility,
        'Dew point temperature(°C)': dew_point,
        'Solar Radiation (MJ/m²)': solar_radiation,
        'Rainfall(mm)': rainfall,
        'Snowfall (cm)': snowfall,
        'Seasons_' + season: 1 if season != 'Spring' else 0, # accounting for drop_first
        'Holiday_' + holiday: 1 if holiday == 'Holiday' else 0,
        'Functioning Day_' + functioning: 1 if functioning == 'Yes' else 0
    }
    
    features_df = pd.DataFrame(data, index=[0])
    return features_df

input_df = user_input_features()

# Align input columns with trained model features (fill missing one-hot columns with 0)
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

# Main Panel layout
st.subheader("🔍 User Selected Inputs")
st.dataframe(input_df)

# Model Selection for Prediction
model_choice = st.selectbox("Choose Model for Prediction", ["Random Forest Regressor", "Linear Regression"])

if st.button("Predict Bike Demand"):
    if model_choice == "Linear Regression":
        input_scaled = scaler.transform(input_df)
        prediction = lr_model.predict(input_scaled)[0]
    else:
        prediction = rf_model.predict(input_df)[0]
        
    st.success(f"Predicted Rented Bike Count: **{max(0, int(prediction))}** bikes")

# Side-by-side comparison section
st.markdown("---")
st.subheader("📊 Side-by-Side Model Comparison")
if st.button("Compare All Models on Current Input"):
    lr_pred = max(0, int(lr_model.predict(scaler.transform(input_df))[0]))
    rf_pred = max(0, int(rf_model.predict(input_df)[0]))
    
    comp_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest Regressor"],
        "Predicted Bike Count": [lr_pred, rf_pred]
    })
    st.table(comp_df)