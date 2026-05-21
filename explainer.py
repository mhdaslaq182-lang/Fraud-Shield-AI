import shap
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from features import engineer_features, FEATURES

def load_model():
    return pickle.load(open("models/random_forest.pkl", "rb"))

def get_shap_fraud(explainer, X):
    """
    Handle SHAP shape (n, features, 2) — index [:,:,1] = fraud class
    """
    sv = explainer.shap_values(X)
    if isinstance(sv, np.ndarray) and sv.ndim == 3:
        return sv[:, :, 1]   # shape (n, 18) — fraud class
    elif isinstance(sv, list):
        return sv[1]
    else:
        return sv

def explain_single(transaction: dict, label: str = "Transaction"):
    """Explain why ONE transaction was flagged."""
    model     = load_model()
    df        = pd.DataFrame([transaction])
    df        = engineer_features(df)
    X         = df[FEATURES]

    explainer  = shap.TreeExplainer(model)
    shap_fraud = get_shap_fraud(explainer, X)  # shape (1, 18)
    fraud_shap = shap_fraud[0]                 # shape (18,)

    contributions = sorted(
        zip(FEATURES, fraud_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    print("\n" + "="*60)
    print(f"  🔎 SHAP — {label}")
    print("="*60)
    print(f"  {'Feature':<25} {'Value':>10}  {'Impact':<10}  Bar")
    print("-"*60)
    for feat, shap_val in contributions[:10]:
        raw_val   = X[feat].values[0]
        direction = "↑ FRAUD" if shap_val > 0 else "↓ SAFE "
        bar       = "█" * min(int(abs(shap_val) * 60), 20)
        print(f"  {feat:<25} {raw_val:>10.3f}  {direction}   {bar}")
    print("="*60)
    print("  ↑ FRAUD = pushes toward fraud")
    print("  ↓ SAFE  = pushes toward legitimate")
    print("="*60)
    return contributions


def explain_batch(X_test: pd.DataFrame, n_samples: int = 200):
    """Generate SHAP summary plots."""
    model = load_model()
    os.makedirs("plots", exist_ok=True)

    sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)
    print(f"\n  Calculating SHAP for {len(sample)} transactions...")

    explainer  = shap.TreeExplainer(model)
    shap_fraud = get_shap_fraud(explainer, sample)

    # ── Plot 1: Bar ──
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_fraud, sample,
        feature_names=FEATURES,
        plot_type="bar",
        show=False
    )
    plt.title("Feature Importance — SHAP Values", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: plots/shap_importance.png")

    # ── Plot 2: Beeswarm ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_fraud, sample,
        feature_names=FEATURES,
        show=False
    )
    plt.title("SHAP Beeswarm — Feature Impact on Fraud", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✅ Saved: plots/shap_beeswarm.png")

    return shap_fraud


if __name__ == "__main__":
    from data_generator import generate_banking_data
    from sklearn.model_selection import train_test_split

    print("="*60)
    print("  SHAP EXPLAINABILITY TEST")
    print("="*60)

    fraud_txn = {
        "amount": 2500.00, "hour": 2, "frequency": 20,
        "distance_km": 800.0, "failed_logins": 6,
        "new_device": 1, "account_age": 10,
        "countries": 3, "velocity_24h": 18,
        "email_risk": 0.90, "balance_before": 1200.0,
        "is_weekend": 0,
    }
    legit_txn = {
        "amount": 45.00, "hour": 14, "frequency": 3,
        "distance_km": 2.0, "failed_logins": 0,
        "new_device": 0, "account_age": 730,
        "countries": 1, "velocity_24h": 2,
        "email_risk": 0.05, "balance_before": 8500.0,
        "is_weekend": 0,
    }

    explain_single(fraud_txn,  label="FRAUD TRANSACTION")
    explain_single(legit_txn,  label="LEGITIMATE TRANSACTION")

    print("\n📊 Generating batch SHAP plots...")
    df = generate_banking_data()
    df = engineer_features(df)
    X  = df[FEATURES]
    _, X_test, _, _ = train_test_split(
        X, df["is_fraud"], test_size=0.2,
        stratify=df["is_fraud"], random_state=42
    )
    explain_batch(X_test, n_samples=200)

    print("\n" + "="*60)
    print("  ✅ SHAP COMPLETE!")
    print("  📊 plots/shap_importance.png")
    print("  📊 plots/shap_beeswarm.png")
    print("="*60)
