#streamlit
#pandas
##numpy
#scikit-learn
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Crop Yield Predictor", layout="centered")
st.title("🌱 Crop Yield Predictor")

st.markdown("Enter crop details below to estimate expected yield.")

# Dropdowns for user input
crops = [
    "Arecanut", "Arhar/Tur", "Castor seed", "Coconut", "Cotton(lint)", "Dry chillies", "Gram", "Jute", "Linseed",
    "Maize", "Mesta", "Niger seed", "Onion", "Other  Rabi pulses", "Potato", "Rapeseed &Mustard", "Rice", "Sesamum",
    "Small millets", "Sugarcane", "Sweet potato", "Tapioca", "Tobacco", "Turmeric", "Wheat", "Bajra", "Black pepper",
    "Cardamom", "Coriander", "Garlic", "Ginger", "Groundnut", "Horse-gram", "Jowar", "Ragi", "Cashewnut", "Banana",
    "Soyabean", "Barley", "Khesari", "Masoor", "Moong(Green Gram)", "Other Kharif pulses", "Safflower", "Sannhamp",
    "Sunflower", "Urad", "Peas & beans (Pulses)", "other oilseeds", "Other Cereals", "Cowpea(Lobia)", "Oilseeds total",
    "Guar seed", "Other Summer Pulses", "Moth"
]

seasons = ["Whole Year", "Kharif", "Rabi", "Autumn", "Summer", "Winter"]

states = [
    "Assam", "Karnataka", "Kerala", "Meghalaya", "West Bengal", "Puducherry", "Goa", "Andhra Pradesh", "Tamil Nadu",
    "Odisha", "Bihar", "Gujarat", "Madhya Pradesh", "Maharashtra", "Mizoram", "Punjab", "Uttar Pradesh", "Haryana",
    "Himachal Pradesh", "Tripura", "Nagaland", "Chhattisgarh", "Uttarakhand", "Jharkhand", "Delhi", "Manipur",
    "Jammu and Kashmir", "Telangana", "Arunachal Pradesh", "Sikkim"
]

# User inputs
crop = st.selectbox("🌾 Crop", crops)
season = st.selectbox("🗓️ Season", seasons)
state = st.selectbox("📍 State", states)
area = st.number_input("📐 Land Area (in hectares)", min_value=0.0, format="%.2f")
rainfall = st.number_input("🌧️ Annual Rainfall (in mm)", min_value=0.0, format="%.2f")
fertilizer = st.number_input("🧪 Fertilizer Used (kg)", min_value=0.0, format="%.2f")
pesticide = st.number_input("🧴 Pesticide Used (kg)", min_value=0.0, format="%.2f")

# Dummy model setup (for demo purposes)
def predict_yield(crop, season, state, area, rainfall, fertilizer, pesticide):
    # Simulate one-hot encoding
    features = {
        'Area': area,
        'Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide
    }

    for c in crops:
        features[f'Crop_{c}'] = 1 if crop == c else 0
    for s in seasons:
        features[f'Season_{s}'] = 1 if season == s else 0
    for st_name in states:
        features[f'State_{st_name}'] = 1 if state == st_name else 0

    input_df = pd.DataFrame([features])

    # Train a quick model (in-memory)
    # Replace this with your real dataset and model for production
    np.random.seed(42)
    dummy_X = pd.DataFrame(np.random.rand(100, len(input_df.columns)), columns=input_df.columns)
    dummy_y = np.random.rand(100) * 100
    model = RandomForestRegressor()
    model.fit(dummy_X, dummy_y)

    return model.predict(input_df)[0]

# Predict button
if st.button("🔍 Predict Yield"):
    result = predict_yield(crop, season, state, area, rainfall, fertilizer, pesticide)
    st.success(f"🌾 Estimated Crop Yield: **{result:.2f} units**")
