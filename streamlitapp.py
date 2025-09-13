import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------
# Load trained ML model
# -------------------------
model = pickle.load(open("crop_yield.csv", "rb"))

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

st.title("🌾 Crop Yield Prediction App")
st.markdown("Predict the **expected crop yield** based on inputs like crop type, season, state, rainfall, fertilizers, and pesticides.")

# -------------------------
# Input options
# -------------------------

crops = [
    "Arecanut","Arhar/Tur","Castor seed","Coconut","Cotton(lint)","Dry chillies","Gram","Jute","Linseed","Maize","Mesta",
    "Niger seed","Onion","Other Rabi pulses","Potato","Rapeseed &Mustard","Rice","Sesamum","Small millets","Sugarcane",
    "Sweet potato","Tapioca","Tobacco","Turmeric","Wheat","Bajra","Black pepper","Cardamom","Coriander","Garlic","Ginger",
    "Groundnut","Horse-gram","Jowar","Ragi","Cashewnut","Banana","Soyabean","Barley","Khesari","Masoor","Moong(Green Gram)",
    "Other Kharif pulses","Safflower","Sannhamp","Sunflower","Urad","Peas & beans (Pulses)","other oilseeds","Other Cereals",
    "Cowpea(Lobia)","Oilseeds total","Guar seed","Other Summer Pulses","Moth"
]

seasons = ["Whole Year", "Kharif", "Rabi", "Autumn", "Summer", "Winter"]

states = [
    "Assam","Karnataka","Kerala","Meghalaya","West Bengal","Puducherry","Goa","Andhra Pradesh","Tamil Nadu","Odisha",
    "Bihar","Gujarat","Madhya Pradesh","Maharashtra","Mizoram","Punjab","Uttar Pradesh","Haryana","Himachal Pradesh",
    "Tripura","Nagaland","Chhattisgarh","Uttarakhand","Jharkhand","Delhi","Manipur","Jammu and Kashmir","Telangana",
    "Arunachal Pradesh","Sikkim"
]

# -------------------------
# User inputs
# -------------------------
col1, col2 = st.columns(2)

with col1:
    crop = st.selectbox("🌱 Select Crop", crops)
    season = st.selectbox("📅 Select Season", seasons)
    state = st.selectbox("🗺️ Select State", states)

with col2:
    area = st.number_input("🌍 Land Area (Hectares)", min_value=0.1, step=0.1)
    rainfall = st.number_input("🌧️ Annual Rainfall (mm)", min_value=0.0, step=1.0)
    fertilizer = st.number_input("💊 Fertilizer Used (kg)", min_value=0.0, step=1.0)
    pesticide = st.number_input("🧪 Pesticide Used (kg)", min_value=0.0, step=1.0)

# -------------------------
# Create DataFrame for model
# -------------------------
if st.button("🚀 Predict Yield"):
    # Create input dataframe
    input_data = pd.DataFrame({
        "Crop": [crop],
        "Season": [season],
        "State": [state],
        "Area": [area],
        "Annual_Rainfall": [rainfall],
        "Fertilizer": [fertilizer],
        "Pesticide": [pesticide]
    })

    # One-hot encode to match training
    input_encoded = pd.get_dummies(input_data)

    # Align with model features
    model_features = model.feature_names_in_  # available in sklearn >=1.0
    input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]

    # -------------------------
    # Show result
    # -------------------------
    st.success(f"🌾 Predicted Crop Yield: **{prediction:.2f} tons/hectare**")

    st.markdown("✅ Prediction complete! Adjust inputs to test different scenarios.")

