
# =============================================
#   Deep Learning Fraud Detection
#   Neural Network (LSTM + Dense)
#   deep_learning.py
# =============================================

import numpy as np
import pandas as pd
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── Load & Prepare Data ──
def load_data():
    print("="*55)
    print("  DEEP LEARNING FRAUD DETECTION")
    print("  Neural Network Model")
    print("="*55)

    from data_generator import generate_banking_data
    from features import engineer_features, FEATURES

    df = generate_banking_data()
    df = engineer_features(df)

    X = df[FEATURES].values
    y = df["is_fraud"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pickle.dump(scaler, open("models/dl_scaler.pkl","wb"))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2,
        stratify=y, random_state=42
    )

    # Balance
    train_df = pd.DataFrame(X_train)
    train_df["label"] = y_train
    maj  = train_df[train_df.label==0]
    mn   = train_df[train_df.label==1]
    mn_up = resample(mn, replace=True,
                     n_samples=len(maj), random_state=42)
    bal  = pd.concat([maj, mn_up]).sample(frac=1, random_state=42)
    X_bal = bal.drop("label",axis=1).values
    y_bal = bal["label"].values

    print(f"  Train (balanced): {len(X_bal):,}")
    print(f"  Test:             {len(X_test):,}")
    print(f"  Features:         {X_bal.shape[1]}")
    return X_bal, X_test, y_bal, y_test, X_scaled.shape[1]

# ── Build Neural Network ──
def build_model(input_dim):
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),

        # Hidden layer 1
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layer 2
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Hidden layer 3
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        # Hidden layer 4
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")]
    )
    return model

# ── Train Model ──
def train_model(X_train, y_train, input_dim):
    print("\n  Building Neural Network...")
    model = build_model(input_dim)
    model.summary()

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_auc", patience=10,
        restore_best_weights=True, mode="max"
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-6
    )
    checkpoint = callbacks.ModelCheckpoint(
        "models/deep_learning_model.keras",
        monitor="val_auc", save_best_only=True, mode="max"
    )

    print("\n  Training Neural Network...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    print("\n  ✅ Model saved: models/deep_learning_model.keras")
    return model, history

# ── Evaluate ──
def evaluate_model(model, X_test, y_test):
    print("\n" + "="*55)
    print("  DEEP LEARNING EVALUATION")
    print("="*55)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    roc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)

    print(classification_report(
        y_test, y_pred,
        target_names=["Legitimate","Fraud"]
    ))
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    return y_prob, y_pred, roc, f1

# ── Plot Results ──
def plot_results(history, y_test, y_prob, y_pred):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Deep Learning Fraud Detection Results",
                 fontsize=16, fontweight="bold")

    # Training History - Accuracy
    ax = axes[0,0]
    ax.plot(history.history["accuracy"],     label="Train", color="#457B9D", lw=2)
    ax.plot(history.history["val_accuracy"], label="Val",   color="#E63946", lw=2)
    ax.set_title("Model Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()

    # Training History - Loss
    ax = axes[0,1]
    ax.plot(history.history["loss"],     label="Train", color="#457B9D", lw=2)
    ax.plot(history.history["val_loss"], label="Val",   color="#E63946", lw=2)
    ax.set_title("Model Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # Training History - AUC
    ax = axes[1,0]
    ax.plot(history.history["auc"],     label="Train AUC", color="#2A9D8F", lw=2)
    ax.plot(history.history["val_auc"], label="Val AUC",   color="#E9C46A", lw=2)
    ax.set_title("ROC-AUC Score", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend()

    # Confusion Matrix
    ax = axes[1,1]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit","Fraud"],
                yticklabels=["Legit","Fraud"])
    ax.set_title("Confusion Matrix", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("plots/deep_learning_results.png",
                dpi=150, bbox_inches="tight")
    print("  ✅ Chart saved: plots/deep_learning_results.png")
    plt.close()

# ── Predict Single Transaction ──
def predict_dl(transaction: dict) -> dict:
    from features import engineer_features, FEATURES
    model  = keras.models.load_model("models/deep_learning_model.keras")
    scaler = pickle.load(open("models/dl_scaler.pkl","rb"))

    df     = pd.DataFrame([transaction])
    df     = engineer_features(df)
    X      = scaler.transform(df[FEATURES])
    prob   = float(model.predict(X, verbose=0)[0][0])

    return {
        "is_fraud":    prob > 0.5,
        "probability": round(prob, 4),
        "risk_level":  "CRITICAL" if prob>0.85 else
                       "HIGH"     if prob>0.65 else
                       "MEDIUM"   if prob>0.40 else "LOW",
        "model":       "Deep Learning Neural Network"
    }

# ── Main ──
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, input_dim = load_data()
    model, history = train_model(X_train, y_train, input_dim)
    y_prob, y_pred, roc, f1 = evaluate_model(model, X_test, y_test)
    plot_results(history, y_test, y_prob, y_pred)

    print("\n" + "="*55)
    print("  DEEP LEARNING COMPLETE!")
    print(f"  ROC-AUC  : {roc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("  Model    : models/deep_learning_model.keras")
    print("  Chart    : plots/deep_learning_results.png")
    print("="*55)

    # Test prediction
    print("\n  Testing prediction...")
    test_txn = {
        "amount_lkr":  250000, "hour": 2,
        "frequency":   20,     "distance_km": 800,
        "failed_logins": 6,    "new_device": 1,
        "account_age": 10,     "countries": 3,
        "velocity_24h": 18,    "email_risk": 0.90,
        "balance_lkr": 120000, "is_weekend": 0,
    }
    result = predict_dl(test_txn)
    print(f"  Fraud: {result['is_fraud']}")
    print(f"  Probability: {result['probability']:.1%}")
    print(f"  Risk: {result['risk_level']}")
    print(f"  Model: {result['model']}")
