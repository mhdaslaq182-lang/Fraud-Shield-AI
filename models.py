import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

def balance_classes(X, y, seed=42):
    """Upsample fraud cases to match legitimate count."""
    df = X.copy()
    df["label"] = y.values
    majority = df[df.label == 0]
    minority = df[df.label == 1]
    minority_up = resample(minority, replace=True,
                           n_samples=len(majority),
                           random_state=seed)
    balanced = pd.concat([majority, minority_up])
    balanced = balanced.sample(frac=1, random_state=seed)
    return balanced.drop("label", axis=1), balanced["label"]


def train_all(X_train, y_train):
    """Train 3 supervised models + 1 unsupervised."""
    os.makedirs("models", exist_ok=True)

    print("\n  Balancing classes...")
    X_bal, y_bal = balance_classes(X_train, y_train)
    print(f"  Balanced size: {len(X_bal):,} samples")

    # ── Logistic Regression ──
    print("\n  Training Logistic Regression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled, y_bal)
    pickle.dump(lr,     open("models/logistic.pkl", "wb"))
    pickle.dump(scaler, open("models/scaler.pkl",   "wb"))
    print("  Logistic Regression saved ✅")

    # ── Random Forest ──
    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_bal, y_bal)
    pickle.dump(rf, open("models/random_forest.pkl", "wb"))
    print("  Random Forest saved ✅")

    # ── Gradient Boosting ──
    print("\n  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_bal, y_bal)
    pickle.dump(gb, open("models/gradient_boost.pkl", "wb"))
    print("  Gradient Boosting saved ✅")

    # ── Isolation Forest (Unsupervised) ──
    print("\n  Training Isolation Forest (Anomaly Detection)...")
    iso = IsolationForest(
        contamination=0.08,
        n_estimators=150,
        random_state=42
    )
    iso.fit(X_train)
    pickle.dump(iso, open("models/isolation_forest.pkl", "wb"))
    print("  Isolation Forest saved ✅")

    return {
        "logistic":         lr,
        "random_forest":    rf,
        "gradient_boost":   gb,
        "isolation_forest": iso,
        "scaler":           scaler
    }


def load_model(name="random_forest"):
    """Load a saved model by name."""
    path = f"models/{name}.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Run main.py first.")
    return pickle.load(open(path, "rb"))


if __name__ == "__main__":
    from data_generator import generate_banking_data
    from features import engineer_features, FEATURES
    from sklearn.model_selection import train_test_split

    print("=" * 45)
    print("  MODEL TRAINING TEST")
    print("=" * 45)

    df = generate_banking_data()
    df = engineer_features(df)

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = train_all(X_train, y_train)

    print("\n" + "=" * 45)
    print("  ALL MODELS TRAINED & SAVED ✅")
    print("=" * 45)
    print(f"  Models saved in: models/")
    print(f"  Files: logistic.pkl, random_forest.pkl,")
    print(f"         gradient_boost.pkl, isolation_forest.pkl")
