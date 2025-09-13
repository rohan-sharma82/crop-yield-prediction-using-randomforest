import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Input options
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

# Streamlit UI
st.title("🌱 Crop Yield Prediction App")

crop = st.selectbox("Select Crop", crops)
season = st.selectbox("Select Season", seasons)
state = st.selectbox("Select State", states)

area = st.number_input("Land Area (in hectares)", min_value=0.0, format="%.2f")
rainfall = st.number_input("Annual Rainfall (in mm)", min_value=0.0, format="%.2f")
fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, format="%.2f")
pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, format="%.2f")

# Prepare input for model
def prepare_input(crop, season, state, area, rainfall, fertilizer, pesticide):
    input_dict = {
        'Area': area,
        'Rainfall': rainfall,
        'Fertilizer': fertilizer,
        'Pesticide': pesticide
    }

    # One-hot encode categorical features
    for col in crops:
        input_dict[f'Crop_{col}'] = 1 if crop == col else 0
    for col in seasons:
        input_dict[f'Season_{col}'] = 1 if season == col else 0
    for col in states:
        input_dict[f'State_{col}'] = 1 if state == col else 0

    return pd.DataFrame([input_dict])

# Predict
if st.button("Predict Yield"):
    input_df = prepare_input(crop, season, state, area, rainfall, fertilizer, pesticide)
    prediction = model.predict(input_df)[0]
    st.success(f"🌾 Estimated Crop Yield: **{prediction:.2f}** units")
