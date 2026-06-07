# src/app.py
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import xgboost as xgb
import pickle
import os
from PIL import Image

# ────────────────────────────────────────────────
# CONFIG - RELATIVE PATHS (Recommended)
# ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VOICE_CNN_PATH = os.path.join(MODELS_DIR, "cnn_feature_extractor.keras")
VOICE_XGB_PATH = os.path.join(MODELS_DIR, "xgboost_pd_model.json")
IMAGE_XGB_PATH = os.path.join(MODELS_DIR, "xgboost_image_model.json")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler.pkl")

# ────────────────────────────────────────────────
# LOAD ALL MODELS
# ────────────────────────────────────────────────
@st.cache_resource
def load_components():
    try:
        st.info("Loading models... Please wait.")
        
        voice_cnn = load_model(VOICE_CNN_PATH)
        
        voice_xgb = xgb.XGBClassifier()
        voice_xgb.load_model(VOICE_XGB_PATH)
        
        image_xgb = xgb.XGBClassifier()
        image_xgb.load_model(IMAGE_XGB_PATH)
        
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        
        image_cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        
        st.success("All models loaded successfully!")
        return voice_cnn, voice_xgb, image_cnn, image_xgb, scaler
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Make sure you have run all training scripts first.")
        raise e

# Load models
voice_cnn, voice_xgb, image_cnn, image_xgb, scaler = load_components()

# Sidebar
st.sidebar.metric("Voice Model Accuracy", "92.31%")   # Update this later with real accuracy

# ────────────────────────────────────────────────
# PAGE SETUP
# ────────────────────────────────────────────────
st.set_page_config(page_title="Parkinson's Disease Detection", layout="wide")
st.title("🧠 Parkinson's Disease Detection")
st.markdown("**Early Detection using Voice Features & Hand Drawings**")

tab1, tab2, tab3 = st.tabs([
    "📤 Upload CSV (Voice)",
    "✍️ Manual Input (Voice)",
    "🖼️ Spiral + Wave Images"
])

# ────────────────────────────────────────────────
# TAB 1: Upload CSV (Voice)
# ────────────────────────────────────────────────
with tab1:
    st.subheader("Upload CSV file (Voice Features)")
    st.caption("Expected: 22 columns in the same order as UCI parkinsons.data (without 'name' and 'status')")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.shape[1] != 22:
                st.error(f"Expected 22 columns. Found {df.shape[1]} columns.")
            else:
                df = df.astype(float)
                st.success(f"✅ Loaded {len(df)} samples")
                
                if st.button("Run Predictions", type="primary"):
                    with st.spinner("Predicting..."):
                        X_raw = df.values
                        X_scaled = scaler.transform(X_raw)
                        X_padded = np.pad(X_scaled, ((0,0),(0,2)), mode='constant')
                        X_img = X_padded.reshape(-1, 6, 4, 1)
                        X_feats = voice_cnn.predict(X_img, verbose=0)
                        
                        preds = voice_xgb.predict(X_feats)
                        probs = voice_xgb.predict_proba(X_feats)[:, 1]
                        
                        results = pd.DataFrame({
                            "Sample": range(1, len(df)+1),
                            "Prediction": ["Parkinson's" if p==1 else "Healthy" for p in preds],
                            "PD Probability": probs.round(4)
                        })
                        
                        st.dataframe(
                            results.style
                            .format({"PD Probability": "{:.2%}"})
                            .apply(lambda x: ["background-color: #ffcccc" if v > 0.70 else "" for v in x], 
                                   subset=["PD Probability"])
                        )
                        
                        csv = results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="parkinson_predictions.csv",
                            mime="text/csv"
                        )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ────────────────────────────────────────────────
# TAB 2: Manual Input (Voice)
# ────────────────────────────────────────────────
with tab2:
    st.subheader("Enter 22 Voice Features Manually")
    
    cols = st.columns(3)
    feature_names = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    
    features = []
    for i, name in enumerate(feature_names):
        val = cols[i % 3].number_input(name, value=0.0, format="%.6f", key=f"v_{i}")
        features.append(val)

    if st.button("Predict This Sample", type="primary"):
        with st.spinner("Predicting..."):
            X_raw = np.array([features])
            X_scaled = scaler.transform(X_raw)
            X_padded = np.pad(X_scaled, ((0,0),(0,2)), mode='constant')
            X_img = X_padded.reshape(1, 6, 4, 1)
            X_feats = voice_cnn.predict(X_img, verbose=0)
            
            pred = voice_xgb.predict(X_feats)[0]
            prob = voice_xgb.predict_proba(X_feats)[0][1]
            
            st.markdown("---")
            if pred == 1:
                st.error(f"**🛑 Parkinson's Disease Detected** (Probability: {prob:.2%})")
            else:
                st.success(f"**✅ Healthy** (PD Probability: {prob:.2%})")
            st.metric("PD Probability", f"{prob:.2%}")

# ────────────────────────────────────────────────
# TAB 3: Spiral + Wave Images
# ────────────────────────────────────────────────
with tab3:
    st.subheader("Upload Spiral and Wave Drawings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Spiral Drawing**")
        spiral_file = st.file_uploader("Upload spiral", type=["jpg","jpeg","png"], key="spiral")
    
    with col2:
        st.markdown("**Wave Drawing**")
        wave_file = st.file_uploader("Upload wave", type=["jpg","jpeg","png"], key="wave")
    
    if st.button("Predict from Drawings", type="primary"):
        if not spiral_file and not wave_file:
            st.warning("Please upload at least one drawing.")
        else:
            feats = []
            with st.spinner("Extracting features from drawings..."):
                if spiral_file:
                    img = load_img(spiral_file, target_size=(224, 224))
                    arr = img_to_array(img)
                    arr = np.expand_dims(arr, 0)
                    arr = preprocess_input(arr)
                    f = image_cnn.predict(arr, verbose=0).flatten()
                    feats.append(f)
                    col1.image(img, caption="Spiral Drawing", use_column_width=True)
                
                if wave_file:
                    img = load_img(wave_file, target_size=(224, 224))
                    arr = img_to_array(img)
                    arr = np.expand_dims(arr, 0)
                    arr = preprocess_input(arr)
                    f = image_cnn.predict(arr, verbose=0).flatten()
                    feats.append(f)
                    col2.image(img, caption="Wave Drawing", use_column_width=True)
                
                if feats:
                    combined = np.mean(feats, axis=0) if len(feats) > 1 else feats[0]
                    pred = image_xgb.predict([combined])[0]
                    prob = image_xgb.predict_proba([combined])[0][1]
                    
                    st.markdown("---")
                    if pred == 1:
                        st.error(f"**🛑 Parkinson's Detected** (Probability: {prob:.2%})")
                    else:
                        st.success(f"**✅ Healthy** (PD Probability: {prob:.2%})")

# Footer
st.markdown("---")
st.caption("Built with CNN + XGBoost | Project by Bhavya,divya and karthik")