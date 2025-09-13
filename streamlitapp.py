import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests

# -----------------------------
# Google Drive model link setup
# -----------------------------
FILE_ID = "1Hn72yc8mOl4zrjy9rZpNfzeZyVIWQ1Ya"
# use export=download to force binary download
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
MODEL_FILE = "crop_yield_model.pkl"

def download_model():
    """Download the .pkl model from Google Drive if not already present."""
    if not os.path.exists(MODEL_FILE):
        st.info("📥 Downloading the ML model from Google Drive...")
        try:
            response = requests.get(MODEL_URL)
            response.raise_for_status()
        except Exception as e:
            st.error(f"Error during downloading model: {e}")
            return None

        # Write as binary
        with open(MODEL_FILE, "wb") as f:
            f.write(response.content)

        st.success("✅ Model downloaded successfully!")
    return MODEL_FILE

def load_model():
    """Load the model using pickle, ensuring it's binary."""
    path = download_model()
    if path is None:
        return None

    # Optional check: file size
    size = os.path.getsize(MODEL_FILE)
    if size < 1000:
        # If file is too small (e.g., HTML or error page), likely wrong
        st.error(f"Downloaded file is suspiciously small ({size} bytes). Might not be the correct model.")
        return None

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Failed to load model via pickle: {e}")
        return None

# -------------------------------------------------
# Streamlit app UI and prediction
# -------------------------------------------------
def main():
    st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
    st.title("🌾 Crop Yield Prediction")

    st.markdown("Enter details of crop, season, state, area, rainfall, fertilizer and pesticide to predict yield.")

    # Dropdown / input options
    crop_list = [
        "Arecanut","Arhar/Tur","Castor seed","Coconut","Cotton(lint)","Dry chillies","Gram","Jute","Linseed","Maize","Mesta",
        "Niger seed","Onion","Other Rabi pulses","Potato","Rapeseed &Mustard","Rice","Sesamum","Small millets","Sugarcane",
        "Sweet potato","Tapioca","Tobacco","Turmeric","Wheat","Bajra","Black pepper","Cardamom","Coriander","Garlic","Ginger",
        "Groundnut","Horse-gram","Jowar","Ragi","Cashewnut","Banana","Soyabean","Barley","Khesari","Masoor","Moong(Green Gram)",
        "Other Kharif pulses","Safflower","Sannhamp","Sunflower","Urad","Peas & beans (Pulses)","other oilseeds","Other Cereals",
        "Cowpea(Lobia)","Oilseeds total","Guar seed","Other Summer Pulses","Moth"
    ]

    season_list = ["Whole Year", "Kharif", "Rabi", "Autumn", "Summer", "Winter"]

    state_list = [
        "Assam","Karnataka","Kerala","Meghalaya","West Bengal","Puducherry","Goa","Andhra Pradesh","Tamil Nadu","Odisha",
        "Bihar","Gujarat","Madhya Pradesh","Maharashtra","Mizoram","Punjab","Uttar Pradesh","Haryana","Himachal Pradesh",
        "Tripura","Nagaland","Chhattisgarh","Uttarakhand","Jharkhand","Delhi","Manipur","Jammu and Kashmir","Telangana",
        "Arunachal Pradesh","Sikkim"
    ]

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        crop = st.selectbox("Crop", crop_list)
        season = st.selectbox("Season", season_list)
        state = st.selectbox("State", state_list)
    with col2:
        area = st.number_input("Land Area (hectares)", min_value=0.1, step=0.1, format="%.2f")
        rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, step=1.0, format="%.2f")
        fertilizer = st.number_input("Fertilizer Used (kg)", min_value=0.0, step=1.0, format="%.2f")
        pesticide = st.number_input("Pesticide Used (kg)", min_value=0.0, step=1.0, format="%.2f")

    if st.button("Predict Yield"):
        model = load_model()
        if model is None:
            st.error("Model could not be loaded. Please check the model file and permissions.")
            return

        # Build input
        input_df = pd.DataFrame({
            "Crop": [crop],
            "Season": [season],
            "State": [state],
            "Area": [area],
            "Annual_Rainfall": [rainfall],
            "Fertilizer": [fertilizer],
            "Pesticide": [pesticide]
        })

        # One hot encode
        input_encoded = pd.get_dummies(input_df)

        # Align with feature names saved in model
        try:
            features = model.feature_names_in_
        except AttributeError:
            st.error("Model doesn't have feature_names_in_. Please save model with sklearn >= 1.0 or store features list separately.")
            return

        input_encoded = input_encoded.reindex(columns=features, fill_value=0)

        # Predict
        try:
            prediction = model.predict(input_encoded)[0]
            st.success(f"✅ Predicted Crop Yield: {prediction:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
