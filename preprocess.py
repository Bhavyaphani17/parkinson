# src/preprocess.py
import os
import numpy as np
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Use relative path (recommended)
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'parkinsons.data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_and_clean_data():
    """Load UCI Parkinson's voice dataset and prepare features."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    
    # Drop 'name' column if present
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
    
    X = df.drop('status', axis=1).values  # 22 features
    y = df['status'].values               # 0 = Healthy, 1 = Parkinson's
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    """Stratified split"""
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

if __name__ == "__main__":
    print("🚀 Step 1: Preprocessing voice data...")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    X_scaled, y, scaler = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Save files
    np.save(os.path.join(MODELS_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(MODELS_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(MODELS_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(MODELS_DIR, 'y_test.npy'), y_test)
    
    with open(os.path.join(MODELS_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Preprocessing completed!")
    print(f"   Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"   Healthy in train: {sum(y_train==0)} | PD in train: {sum(y_train==1)}")
    print(f"   Files saved in: {MODELS_DIR}")