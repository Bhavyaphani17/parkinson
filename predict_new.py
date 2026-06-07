# src/predict_new.py
# pylint: disable=import-error
# pyright: reportMissingImports=false
# pyright: reportMissingModuleSource=false

import numpy as np
from tensorflow.keras.models import load_model
import xgboost as xgb
import pickle
import os
from datetime import datetime

# ────────────────────────────────────────────────
#                     CONFIG
# ────────────────────────────────────────────────
MODELS_DIR = '../models'
CNN_MODEL_PATH = os.path.join(MODELS_DIR, 'cnn_feature_extractor.keras')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_pd_model.json')
SCALER_PATH    = os.path.join(MODELS_DIR, 'scaler.pkl')

# ────────────────────────────────────────────────
#               SAMPLE DATA (from parkinsons.data)
# ────────────────────────────────────────────────
# You can add as many samples as you want here
SAMPLES = [
    # Example 1: Parkinson's case (phon_R01_S01_1, actual status=1)
    {
        "name": "phon_R01_S01_1 (known PD)",
        "features": [
            119.99200, 157.30200, 74.99700, 0.00784, 0.00007, 0.00370, 0.00554, 0.01109,
            0.04374, 0.42600, 0.02182, 0.03130, 0.02971, 0.06545, 0.02211, 21.03300,
            0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
        ]
    },
    # Example 2: Healthy case (phon_R01_S42_6, actual status=0)
    {
        "name": "phon_R01_S42_6 (known Healthy)",
        "features": [
            244.99000, 272.21000, 239.17000, 0.00451, 0.00002, 0.00279, 0.00237, 0.00837,
            0.01897, 0.18100, 0.01084, 0.01121, 0.01255, 0.03253, 0.01049, 21.52800,
            0.522812, 0.646818, -7.304500, 0.171088, 2.095237, 0.096220
        ]
    },
    # Add more samples here if you want...
]

# ────────────────────────────────────────────────
#                     MAIN CODE
# ────────────────────────────────────────────────

def load_components():
    print("Loading models and scaler...")
    cnn = load_model(CNN_MODEL_PATH)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    print("Models and scaler loaded successfully.")
    return cnn, xgb_model, scaler


def prepare_input(features_list):
    """Convert list of feature lists → numpy array (n_samples, 22)"""
    X = np.array(features_list)
    if X.shape[1] != 22:
        raise ValueError(f"Each sample must have exactly 22 features. Got {X.shape[1]}")
    return X


def predict_samples(cnn, xgb_model, scaler, X_raw):
    # Scale
    X_scaled = scaler.transform(X_raw)
    
    # Pad to 24 features (6×4) and reshape to (n, 6, 4, 1)
    pad_width = max(0, 24 - X_scaled.shape[1])
    X_padded = np.pad(X_scaled, ((0, 0), (0, pad_width)), mode='constant')
    X_img = X_padded.reshape(-1, 6, 4, 1)
    
    # Extract CNN features
    X_features = cnn.predict(X_img, verbose=0)
    
    # XGBoost prediction
    preds = xgb_model.predict(X_features)
    probs = xgb_model.predict_proba(X_features)[:, 1]  # prob of PD (class 1)
    
    return preds, probs


def save_results(predictions):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"../models/predictions_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Parkinson's Disease Prediction Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        
        for name, pred, prob in predictions:
            result = "Parkinson's Disease" if pred == 1 else "Healthy"
            f.write(f"Sample: {name}\n")
            f.write(f"Prediction: {result}\n")
            f.write(f"PD Probability: {prob:.4f}\n")
            f.write("-"*40 + "\n")
    
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    try:
        cnn, xgb_model, scaler = load_components()
        
        # Prepare input data
        sample_names = [s["name"] for s in SAMPLES]
        feature_lists = [s["features"] for s in SAMPLES]
        
        X_input = prepare_input(feature_lists)
        
        # Make predictions
        print(f"\nPredicting for {len(SAMPLES)} sample(s)...")
        predictions, probabilities = predict_samples(cnn, xgb_model, scaler, X_input)
        
        # Show results
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        
        results_list = []
        for name, pred, prob in zip(sample_names, predictions, probabilities):
            result_str = "Parkinson's Disease" if pred == 1 else "Healthy"
            print(f"• {name:30} → {result_str:18} (PD prob: {prob:.4f})")
            results_list.append((name, pred, prob))
        
        print("="*60)
        
        # Save to file
        save_results(results_list)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Make sure all model files exist in the 'models' folder.")