import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score
)

def evaluate_all(models, scaler, X_test, y_test):
    print("\n" + "="*50)
    print("  MODEL EVALUATION RESULTS")
    print("="*50)
    results = {}
    for name, model in models.items():
        if name in ["isolation_forest", "scaler"]:
            continue
        X = scaler.transform(X_test) if name == "logistic" else X_test
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        roc_auc  = roc_auc_score(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        f1       = f1_score(y_test, y_pred)
        print(f"\n── {name.upper().replace('_',' ')} ──")
        print(classification_report(y_test, y_pred, target_names=["Legitimate","Fraud"]))
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"  Avg Prec : {avg_prec:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        results[name] = {
            "model": model, "y_pred": y_pred,
            "y_prob": y_prob, "roc_auc": roc_auc,
            "avg_prec": avg_prec, "f1": f1,
        }
    iso      = models["isolation_forest"]
    iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)
    iso_f1   = f1_score(y_test, iso_pred)
    print(f"\n── ISOLATION FOREST (Unsupervised) ──")
    print(classification_report(y_test, iso_pred, target_names=["Legitimate","Fraud"]))
    print(f"  F1 Score : {iso_f1:.4f}")
    return results

def plot_results(results, y_test, feature_names):
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Banking Fraud Detection — Results", fontsize=16, fontweight="bold")
    colors = ["#E63946", "#457B9D", "#2A9D8F"]

    # ROC Curves
    ax = axes[0, 0]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", color=color, lw=2)
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_title("ROC Curves", fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

    # Precision-Recall
    ax = axes[0, 1]
    for (name, res), color in zip(results.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(rec, prec, label=f"{name} (AP={res['avg_prec']:.3f})", color=color, lw=2)
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=8)

    # Model Comparison
    ax = axes[0, 2]
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.25
    metrics = {
        "ROC-AUC":  [results[m]["roc_auc"]  for m in model_names],
        "Avg Prec": [results[m]["avg_prec"] for m in model_names],
        "F1 Score": [results[m]["f1"]       for m in model_names],
    }
    for i, (metric, vals) in enumerate(metrics.items()):
        ax.bar(x + i*width, vals, width, label=metric, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_","\n") for m in model_names], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Comparison", fontweight="bold")
    ax.legend(fontsize=8)

    # Confusion Matrix
    ax = axes[1, 0]
    best = max(results, key=lambda k: results[k]["roc_auc"])
    cm = confusion_matrix(y_test, results[best]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legitimate","Fraud"],
                yticklabels=["Legitimate","Fraud"])
    ax.set_title(f"Confusion Matrix ({best})", fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    # Feature Importance
    ax = axes[1, 1]
    rf = results.get("random_forest", {}).get("model")
    if rf is None:
        rf = pickle.load(open("models/random_forest.pkl","rb"))
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    ax.barh([feature_names[i] for i in indices][::-1],
            importances[indices][::-1], color="#457B9D")
    ax.set_title("Top 10 Features (Random Forest)", fontweight="bold")
    ax.set_xlabel("Importance")

    # Fraud Type Distribution
    ax = axes[1, 2]
    df = pd.read_csv("data/banking_dataset.csv")
    labels = {0:"Legitimate", 1:"Credit Card", 2:"Money Laundering", 3:"Account Takeover"}
    counts = df["fraud_type"].map(labels).value_counts()
    ax.bar(counts.index, counts.values,
           color=["#2A9D8F","#E63946","#E9C46A","#F4A261"], edgecolor="white")
    ax.set_title("Transaction Types", fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelsize=8)
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 50,
                str(int(bar.get_height())), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("plots/evaluation_results.png", dpi=150, bbox_inches="tight")
    print("\n✅ Chart saved: plots/evaluation_results.png")
    plt.close()

if __name__ == "__main__":
    from data_generator import generate_banking_data
    from features import engineer_features, FEATURES
    from models import train_all
    from sklearn.model_selection import train_test_split

    df = generate_banking_data()
    df = engineer_features(df)
    X  = df[FEATURES]
    y  = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    trained = train_all(X_train, y_train)
    scaler  = pickle.load(open("models/scaler.pkl","rb"))
    results = evaluate_all(trained, scaler, X_test, y_test)
    plot_results(results, y_test, FEATURES)

    print("\n" + "="*50)
    print("  EVALUATION COMPLETE ✅")
    print("="*50)
