# src/train_image_model.py
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.model_selection import train_test_split
import pickle

# Relative paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
IMAGES_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'drawings')  # Create this folder

# You need to create this structure:
# data/drawings/
#   ├── healthy/
#   │   ├── spiral_1.png
#   │   ├── wave_1.png
#   │   └── ...
#   └── parkinson/
#       ├── spiral_2.png
#       ├── wave_2.png
#       └── ...

def extract_features_from_images(image_paths, image_cnn):
    """Extract features using MobileNetV2"""
    features = []
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Warning: Image not found - {path}")
            continue
        img = load_img(path, target_size=(224, 224))
        arr = img_to_array(img)
        arr = np.expand_dims(arr, 0)
        arr = preprocess_input(arr)
        feat = image_cnn.predict(arr, verbose=0).flatten()
        features.append(feat)
    return np.array(features)

if __name__ == "__main__":
    print("🚀 Step 5: Training Image-based XGBoost Model (Spiral + Wave)...")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load pre-trained MobileNetV2 as feature extractor
    print("Loading MobileNetV2 feature extractor...")
    image_cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    
    # TODO: You need to prepare your image dataset first
    # For now, we'll show how to load if you have the data
    
    # Example: Collect image paths (you need to implement this part)
    # healthy_spirals = [...]   # list of paths
    # pd_spirals = [...]
    # healthy_waves = [...]
    # pd_waves = [...]
    
    print("\n⚠️  IMPORTANT: You need to prepare your drawing images first!")
    print("Create folder structure:")
    print(f"   {IMAGES_DIR}/healthy/")
    print(f"   {IMAGES_DIR}/parkinson/")
    
    # Placeholder - replace with your actual image loading logic
    # For demonstration, we'll assume you have combined features already
    # In real scenario, you would combine spiral + wave features for each subject
    
    print("\nSince image dataset is not provided yet, creating a placeholder model...")
    print("You can train this properly once you have labeled spiral/wave drawings.")
    
    # For now, we create a dummy model so your app doesn't crash
    # Later replace with real training when you have data
    
    # Dummy training data (replace this with real features)
    np.random.seed(42)
    X_dummy = np.random.rand(100, 1280)  # MobileNetV2 outputs 1280 features
    y_dummy = np.random.randint(0, 2, 100)
    
    X_train_img, X_test_img, y_train_img, y_test_img = train_test_split(
        X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy
    )
    
    # Train XGBoost for images
    image_xgb = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    image_xgb.fit(X_train_img, y_train_img)
    
    # Evaluate
    y_pred_img = image_xgb.predict(X_test_img)
    acc_img = accuracy_score(y_test_img, y_pred_img)
    
    print(f"\nImage Model Accuracy (on dummy data): {acc_img:.2%}")
    
    # Save the image model with exact name expected by app.py
    image_model_path = os.path.join(MODELS_DIR, "xgboost_image_model.json")
    image_xgb.save_model(image_model_path)
    
    print(f"✅ Image XGBoost model saved at: {image_model_path}")
    print("\nNote: This is currently using dummy data.")
    print("To make it real:")
    print("1. Collect spiral and wave drawings from healthy and PD patients")
    print("2. Label them properly")
    print("3. Update the image loading and feature extraction part")
    
    print("\n🎉 Image model training completed (placeholder)!")