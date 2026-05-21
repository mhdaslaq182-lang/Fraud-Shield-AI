
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    average_precision_score, precision_recall_curve
)

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

FEATURES = (
    [f"V{i}" for i in range(1, 29)] +
    ["Amount_log", "Amount_scaled", "Hour",
     "is_night", "is_high_amount", "is_round_amount"]
)

def load_kaggle_data():
    print("="*55)
    print("  LOADING REAL KAGGLE DATASET")
    print("="*55)
    df = pd.read_csv("data/creditcard.csv")
    print(f"  Total transactions : {len(df):,}")
    print(f"  Legitimate         : {(df.Class==0).sum():,}")
    print(f"  Fraudulent         : {(df.Class==1).sum():,}")
    print(f"  Fraud rate         : {df.Class.mean():.3%}")
    print(f"  Features           : {df.shape[1]-1}")
    print(f"  Amount range       : ${df.Amount.min():.2f} - ${df.Amount.max():.2f}")
    print("="*55)
    return df

def engineer_features(df):
    df = df.copy()
    df["Amount_log"]      = np.log1p(df["Amount"])
    df["Amount_scaled"]   = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["Hour"]            = (df["Time"] % 86400) // 3600
    df["is_night"]        = ((df["Hour"] >= 0) & (df["Hour"] <= 5)).astype(int)
    df["is_high_amount"]  = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    df["is_round_amount"] = (df["Amount"] % 100 == 0).astype(int)
    return df

def balance_data(X, y, seed=42):
    df = X.copy()
    df["label"] = y.values
    majority = df[df.label == 0]
    minority = df[df.label == 1]
    target   = min(len(minority) * 10, len(majority))
    maj_down = resample(majority, replace=False, n_samples=target, random_state=seed)
    balanced = pd.concat([maj_down, minority]).sample(frac=1, random_state=seed)
    print(f"  Balanced: {len(balanced):,} samples ({minority.shape[0]:,} fraud + {target:,} legit)")
    return balanced.drop("label", axis=1), balanced["label"]

def train_models(X_train, y_train):
    print("\n" + "="*55)
    print("  TRAINING ON REAL KAGGLE DATA")
    print("="*55)
    print("\n  Balancing classes...")
    X_bal, y_bal = balance_data(X_train, y_train)

    print("\n  Training Logistic Regression...")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_bal)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_scaled, y_bal)
    pickle.dump(scaler, open("models/kaggle_scaler.pkl","wb"))
    print("  Logistic Regression saved")

    print("\n  Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_bal, y_bal)
    print("  Random Forest saved")

    print("\n  Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, random_state=42)
    gb.fit(X_bal, y_bal)
    print("  Gradient Boosting saved")

    models = {"logistic": lr, "random_forest": rf, "gradient_boost": gb}
    for name, model in models.items():
        pickle.dump(model, open(f"models/kaggle_{name}.pkl","wb"))
        print(f"  Saved: models/kaggle_{name}.pkl")
    return models

def evaluate_models(models, X_test, y_test):
    print("\n" + "="*55)
    print("  EVALUATION ON REAL DATA")
    print("="*55)
    scaler  = pickle.load(open("models/kaggle_scaler.pkl","rb"))
    results = {}
    for name, model in models.items():
        X      = scaler.transform(X_test) if name == "logistic" else X_test
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        roc    = roc_auc_score(y_test, y_prob)
        f1     = f1_score(y_test, y_pred)
        ap     = average_precision_score(y_test, y_prob)
        print(f"\n-- {name.upper()} --")
        print(classification_report(y_test, y_pred, target_names=["Legitimate","Fraud"]))
        print(f"  ROC-AUC  : {roc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Avg Prec : {ap:.4f}")
        results[name] = {"model":model,"y_pred":y_pred,"y_prob":y_prob,"roc":roc,"f1":f1,"ap":ap}
    return results

def plot_results(results, y_test, df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Real Kaggle Credit Card Fraud Detection Results", fontsize=16, fontweight="bold")
    colors = ["#E63946","#457B9D","#2A9D8F"]

    ax = axes[0, 0]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc']:.3f})", color=color, lw=2)
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_title("ROC Curves", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, label=f"{name} (AP={res['ap']:.3f})", color=color, lw=2)
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    names = list(results.keys())
    x     = np.arange(len(names))
    w     = 0.25
    for i, (metric, vals) in enumerate({
        "ROC-AUC":  [results[m]["roc"] for m in names],
        "F1 Score": [results[m]["f1"]  for m in names],
        "Avg Prec": [results[m]["ap"]  for m in names],
    }.items()):
        ax.bar(x + i*w, vals, w, label=metric, color=colors[i])
    ax.set_xticks(x + w)
    ax.set_xticklabels([n.replace("_","\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    best = max(results, key=lambda k: results[k]["roc"])
    cm   = confusion_matrix(y_test, results[best]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
    ax.set_title(f"Confusion Matrix ({best})", fontweight="bold")

    ax = axes[1, 1]
    fraud = df[df.Class==1]["Amount"]
    legit = df[df.Class==0]["Amount"].sample(1000)
    ax.hist(legit, bins=50, alpha=0.6, color="#2A9D8F", label="Legitimate", density=True)
    ax.hist(fraud, bins=50, alpha=0.8, color="#E63946", label="Fraud", density=True)
    ax.set_title("Amount Distribution", fontweight="bold")
    ax.set_xlabel("Amount ($)")
    ax.legend()

    ax = axes[1, 2]
    df2 = df.copy()
    df2["Hour"] = (df2["Time"] % 86400) // 3600
    hourly = df2.groupby(["Hour","Class"]).size().reset_index(name="count")
    for cls, color, label in [(0,"#2A9D8F","Legitimate"),(1,"#E63946","Fraud")]:
        h = hourly[hourly.Class==cls]
        ax.plot(h["Hour"], h["count"], color=color, label=label, lw=2)
    ax.set_title("Transactions by Hour", fontweight="bold")
    ax.set_xlabel("Hour of Day")
    ax.legend()

    plt.tight_layout()
    plt.savefig("plots/kaggle_results.png", dpi=150, bbox_inches="tight")
    print("\n  Chart saved: plots/kaggle_results.png")
    plt.close()

if __name__ == "__main__":
    df      = load_kaggle_data()
    df      = engineer_features(df)
    X       = df[FEATURES]
    y       = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Fraud in test: {y_test.sum():,} ({y_test.mean():.3%})")
    models  = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    plot_results(results, y_test, df)
    print("\n" + "="*55)
    print("  REAL KAGGLE DATASET COMPLETE!")
    print("  284,807 real transactions trained!")
    print("  Charts: plots/kaggle_results.png")
    print("="*55)
