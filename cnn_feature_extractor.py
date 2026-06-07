# src/cnn_feature_extractor.py
import os
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Use relative paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def build_and_train_cnn(input_shape=(6, 4, 1), epochs=50, batch_size=32):
    """Build and train a lightweight CNN as feature extractor for voice data."""
    
    inputs = Input(shape=input_shape, name='input')
    
    # Convolutional layers with BatchNorm and Dropout for better generalization
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    features = Dense(64, activation='relu', name='features')(x)  # Final feature vector
    
    model = Model(inputs=inputs, outputs=features, name='Voice_CNN_Feature_Extractor')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',                    # We use it as feature extractor (reconstruction-like)
        metrics=['mae']
    )
    
    model.summary()
    
    # Load training data
    X_train = np.load(os.path.join(MODELS_DIR, 'X_train.npy'))
    
    # Reshape to image format (6x4x1)
    pad_width = max(0, 24 - X_train.shape[1])
    X_padded = np.pad(X_train, ((0, 0), (0, pad_width)), mode='constant')
    X_img = X_padded.reshape(-1, 6, 4, 1)
    
    print(f"Training CNN on {X_img.shape} input shape...")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # Train (self-supervised style - learning to extract meaningful features)
    history = model.fit(
        X_img, X_img,   # Auto-encoder like (input → features → reconstruct via later layers if needed)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save trained model
    save_path = os.path.join(MODELS_DIR, "cnn_feature_extractor.keras")
    model.save(save_path)
    print(f"✅ Trained CNN saved to: {save_path}")
    
    return model


if __name__ == "__main__":
    print("🚀 Step 2: Building and training CNN feature extractor...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    build_and_train_cnn(epochs=30)   # You can increase epochs if needed
    print("CNN training completed!")