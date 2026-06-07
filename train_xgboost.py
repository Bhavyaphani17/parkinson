# src/train_xgboost.py
import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Relative paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

if __name__ == "__main__":
    print("🚀 Step 4: Training XGBoost on CNN-extracted features...")

    # Load extracted features and labels
    X_train = np.load(os.path.join(MODELS_DIR, 'train_cnn_features.npy'))
    X_test = np.load(os.path.join(MODELS_DIR, 'test_cnn_features.npy'))
    y_train = np.load(os.path.join(MODELS_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(MODELS_DIR, 'y_test.npy'))

    print(f"Train features: {X_train.shape} | Test features: {X_test.shape}")
    print(f"Class distribution - Healthy: {sum(y_train==0)} | PD: {sum(y_train==1)}")

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Healthy: {sum(y_train_res==0)} | PD: {sum(y_train_res==1)}")

    # XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=600,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=2,
        gamma=0.1,
        eval_metric='logloss',
        random_state=42,
        tree_method='hist',
        use_label_encoder=False
    )

    print("\nTraining XGBoost model...")
    xgb_clf.fit(X_train_res, y_train_res)

    # Predictions & Evaluation
    y_pred = xgb_clf.predict(X_test)
    y_prob = xgb_clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC Score     : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', "Parkinson's"]))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', "Parkinson's"], 
                yticklabels=['Healthy', "Parkinson's"])
    plt.title('Confusion Matrix - Voice Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(MODELS_DIR, 'confusion_matrix_voice.png'))
    plt.close()

    # Save the model with exact name expected by app.py
    model_path = os.path.join(MODELS_DIR, "xgboost_pd_model.json")
    xgb_clf.save_model(model_path)
    print(f"\n✅ XGBoost model saved successfully at: {model_path}")
    print(f"Final Voice Model Accuracy: {acc*100:.2f}%")

    # Optional: Save feature importance plot
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(xgb_clf, max_num_features=15)
    plt.title("Top 15 Important Features (CNN + XGBoost)")
    plt.savefig(os.path.join(MODELS_DIR, 'feature_importance.png'))
    plt.close()
    
    print("Feature importance plot saved.")
    print("\n🎉 Voice model training completed successfully!")