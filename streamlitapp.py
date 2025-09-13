import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import gdown

# ---------------------------------------
# Google Drive Model Download
# ---------------------------------------
FILE_ID = "1Hn72yc8mOl4zrjy9rZpNfzeZyVIWQ1Ya"  # your model file ID
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_FILE = "crop_yield_model.pkl"

def download_model():
    """Download model file from Google Drive if not already present"""
    if not os.path.exists(MODEL_FILE):
        gdown.download(MODEL_URL, MODEL_FILE, quiet=False)
    return MODEL_FILE

def load_model():
    """Load trained ML model from pickle"""
    model_path = download_model()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
def main():
    st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
    st.title("🌱 Crop Yield Prediction App")
    st.markdown("Predict the **expected crop yield** based on crop, season, state, rainfall, fertilizer, and pesticide.")

    # Dropdown options
    crop_options = [
        "Arecanut","Arhar/Tur","Castor seed","Coconut","Cotton(lint)","Dry chillies","Gram","Jute","Linseed","Maize","Mesta",
        "Niger seed","Onion","Other Rabi pulses","Potato","Rapeseed &Mustard","Rice","Sesamum","Small millets","Sugarcane",
        "Sweet potato","Tapioca","Tobacco","Turmeric","Wheat","Bajra","Black pepper","Cardamom","Coriander","Garlic","Ginger",
        "Groundnut","Horse-gram","Jowar","Ragi","Cashewnut","Banana","Soyabean","Barley","Khesari","Masoor","Moong(Green Gram)",
        "Other Kharif pulses","Safflower","Sannhamp","Sunflower","Urad","Peas & beans (Pulses)","other oilseeds","Other Cereals",
        "Cowpea(Lobia)","Oilseeds total","Guar seed","Other Summer Pulses","Moth"
    ]

    season_options = ["Whole Year", "Kharif", "Rabi", "Autumn", "Summer", "Winter"]

    state_options = [
        "Assam","Karnataka","Kerala","Meghalaya","West Bengal","Puducherry","Goa","Andhra Pradesh","Tamil Nadu","Odisha",
        "Bihar","Gujarat","Madhya Pradesh","Maharashtra","Mizoram","Punjab","Uttar Pradesh","Haryana","Himachal Pradesh",
        "Tripura","Nagaland","Chhattisgarh","Uttarakhand","Jharkhand","Delhi","Manipur","Jammu and Kashmir","Telangana",
        "Arunachal Pradesh","Sikkim"
    ]

    # Layout with two columns
    col1, col2 = st.columns(2)

    with col1:
        crop = st.selectbox("🌾 Select Crop", crop_options)
        season = st.selectbox("📅 Select Season", season_options)
        state = st.selectbox("🗺️ Select State", state_options)

    with col2:
        area = st.number_input("🌍 Land Area (Hectares)", min_value=0.1, step=0.1, format="%.2f")
        rainfall = st.number_input("🌧️ Annual Rainfall (mm)", min_value=0.0, step=1.0, format="%.2f")
        fertilizer = st.number_input("💊 Fertilizer Used (kg)", min_value=0.0, step=1.0, format="%.2f")
        pesticide = st.number_input("🧪 Pesticide Used (kg)", min_value=0.0, step=1.0, format="%.2f")

    if st.button("🚀 Predict Yield"):
        # Load model
        model = load_model()

        # Prepare input
        input_df = pd.DataFrame({
            "Crop": [crop],
            "Season": [season],
            "State": [state],
            "Area": [area],
            "Annual_Rainfall": [rainfall],
            "Fertilizer": [fertilizer],
            "Pesticide": [pesticide]
        })

        # One-hot encode input
        input_encoded = pd.get_dummies(input_df)

        # Align with model features
        try:
            model_features = model.feature_names_in_  # sklearn >= 1.0
        except AttributeError:
            st.error("Model does not have feature_names_in_. Save model with sklearn >=1.0 or store feature list.")
            return

        input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

        # Predict
        prediction = model.predict(input_encoded)[0]

        # Show result
        st.success(f"🌾 Predicted Crop Yield: **{prediction:.2f} tons/hectare**")

# ---------------------------------------
# Run app
# ---------------------------------------
if __name__ == "__main__":
    main()
