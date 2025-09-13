import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests

# ---------------------------------------
# Google Drive Model Download (via requests)
# ---------------------------------------
FILE_ID = "1Hn72yc8mOl4zrjy9rZpNfzeZyVIWQ1Ya"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_FILE = "crop_yield_model.pkl"


def download_model():
    """Download model file from Google Drive if not already present"""
    if not os.path.exists(MODEL_FILE):
        st.info("📥 Downloading ML model from Google Drive...")
        response = requests.get(MODEL_URL)
        response.raise_for_status()  # throw error if download fails
        with open(MODEL_FILE, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded successfully!")
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
    st.markdown(
        "Predict the **expected crop yield** based on crop, season, state, "
        "land area, rainfall, fertilizer, and pesticide usage."
    )

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
        try:
            model = load_model()
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            return

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
            st.error("⚠️ Model is missing feature_names_in_. Retrain with sklearn >=1.0 to include feature names.")
            return

        input_encoded = input_encoded.reindex(columns=model_features, fill_value=0)

        # Predict
        try:
            prediction = model.predict(input_encoded)[0]
            st.success(f"🌾 Predicted Crop Yield: **{prediction:.2f} tons/hectare**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")


# ---------------------------------------
# Run app
# ---------------------------------------
if __name__ == "__main__":
    main()
