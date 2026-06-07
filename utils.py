# src/utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data():
    # Path relative to where you run the script (from src/ folder)
    path = r"C:\Users\vadla\Documents\PARKINSON\data\parkinsons.data"
    
    df = pd.read_csv(path)
    
    # Remove 'name' column if it exists
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
    
    # Features (X) = all columns except 'status'
    X = df.drop('status', axis=1)
    y = df['status']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y.values, scaler

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test