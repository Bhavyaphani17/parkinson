# src/extract_features.py
import os
import numpy as np
from tensorflow.keras.models import load_model

# Relative paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def reshape_to_image(X, target_shape=(6, 4, 1)):
    """Pad 22 features to 24 and reshape into (6, 4, 1) image format"""
    pad_width = max(0, 24 - X.shape[1])
    X_padded = np.pad(X, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    X_img = X_padded.reshape(-1, *target_shape)
    return X_img

if __name__ == "__main__":
    print("🚀 Step 3: Extracting CNN features from trained model...")
    
    # Load data
    X_train = np.load(os.path.join(MODELS_DIR, 'X_train.npy'))
    X_test = np.load(os.path.join(MODELS_DIR, 'X_test.npy'))
    
    # Load the trained CNN
    cnn_path = os.path.join(MODELS_DIR, 'cnn_feature_extractor.keras')
    if not os.path.exists(cnn_path):
        raise FileNotFoundError(f"Trained CNN not found at: {cnn_path}\nPlease run cnn_feature_extractor.py first!")
    
    cnn_model = load_model(cnn_path)
    print("✅ Trained CNN model loaded successfully.")
    
    print("Reshaping data to image-like format...")
    X_train_img = reshape_to_image(X_train)
    X_test_img = reshape_to_image(X_test)
    
    print(f"Input shape for CNN: {X_train_img.shape}")
    
    print("Extracting features from training set...")
    train_features = cnn_model.predict(X_train_img, batch_size=32, verbose=1)
    
    print("Extracting features from test set...")
    test_features = cnn_model.predict(X_test_img, batch_size=32, verbose=1)
    
    # Save extracted features
    np.save(os.path.join(MODELS_DIR, 'train_cnn_features.npy'), train_features)
    np.save(os.path.join(MODELS_DIR, 'test_cnn_features.npy'), test_features)
    
    print("\n✅ Feature extraction completed!")
    print(f"Train features shape: {train_features.shape}")
    print(f"Test features shape : {test_features.shape}")
    print(f"Features saved in: {MODELS_DIR}")