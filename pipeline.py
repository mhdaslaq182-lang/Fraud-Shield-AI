
# ══════════════════════════════════════════
#   Real-World Transaction Pipeline
#   pipeline.py
# ══════════════════════════════════════════

import pandas as pd
import numpy as np
import pickle
import time
import json
import os
from datetime import datetime
from features import engineer_features, FEATURES

SL_BANKS    = ["BOC","Peoples","Commercial","Sampath","HNB","NSB","Seylan","NTB","DFCC","Pan Asia"]
SL_CITIES   = ["Colombo","Kandy","Galle","Jaffna","Negombo","Matara","Kurunegala","Trincomalee"]
SL_MERCHANTS= ["Daraz.lk","PickMe","Dialog","Keells","Cargills","Odel","Abans","Singer","SLT"]

def generate_transaction():
    """Generate a realistic Sri Lankan banking transaction"""
    is_fraud = np.random.random() < 0.08

    if is_fraud:
        fraud_type = np.random.choice([
            "Credit Card Fraud",
            "Account Takeover", 
            "Money Laundering"
        ])
        txn = {
            "transaction_id": f"TXN{int(time.time()*1000)}",
            "timestamp":      datetime.now().isoformat(),
            "bank":           np.random.choice(SL_BANKS),
            "city":           np.random.choice(SL_CITIES),
            "merchant":       np.random.choice(SL_MERCHANTS),
            "amount_lkr":     round(np.random.uniform(50000, 500000), 2),
            "balance_lkr":    round(np.random.uniform(10000, 200000), 2),
            "hour":           int(np.random.choice([1,2,3,4])),
            "frequency":      int(np.random.poisson(18)),
            "distance_km":    round(np.random.uniform(300, 2000), 1),
            "failed_logins":  int(np.random.randint(4, 10)),
            "new_device":     1,
            "account_age":    int(np.random.randint(1, 45)),
            "countries":      int(np.random.randint(2, 5)),
            "velocity_24h":   int(np.random.poisson(12)),
            "email_risk":     round(np.random.uniform(0.3, 0.9), 2),
            "is_weekend":     int(np.random.choice([0,1])),
            "fraud_type":     fraud_type,
            "is_fraud_sim":   True,
        }
    else:
        txn = {
            "transaction_id": f"TXN{int(time.time()*1000)}",
            "timestamp":      datetime.now().isoformat(),
            "bank":           np.random.choice(SL_BANKS),
            "city":           np.random.choice(SL_CITIES),
            "merchant":       np.random.choice(SL_MERCHANTS),
            "amount_lkr":     round(abs(np.random.exponential(8000)), 2),
            "balance_lkr":    round(np.random.uniform(50000, 5000000), 2),
            "hour":           int(np.random.randint(8, 20)),
            "frequency":      int(np.random.poisson(3)),
            "distance_km":    round(abs(np.random.exponential(10)), 1),
            "failed_logins":  int(np.random.choice([0,0,0,1], p=[0.7,0.15,0.1,0.05])),
            "new_device":     int(np.random.choice([0,1], p=[0.92,0.08])),
            "account_age":    int(np.random.randint(180, 3000)),
            "countries":      1,
            "velocity_24h":   int(np.random.poisson(2)),
            "email_risk":     round(np.random.uniform(0, 0.2), 2),
            "is_weekend":     int(np.random.choice([0,1])),
            "fraud_type":     "Legitimate",
            "is_fraud_sim":   False,
        }
    return txn

def process_transaction(txn: dict, model) -> dict:
    """Process transaction through AI pipeline"""
    start = time.time()

    df   = pd.DataFrame([txn])
    df   = engineer_features(df)
    prob = model.predict_proba(df[FEATURES])[0][1]

    process_time = round((time.time() - start) * 1000, 2)

    risk = "CRITICAL" if prob > 0.85 else            "HIGH"     if prob > 0.65 else            "MEDIUM"   if prob > 0.40 else "LOW"

    result = {
        "transaction_id":    str(txn.get("transaction_id","")),
        "timestamp":         str(txn.get("timestamp","")),
        "bank":              str(txn.get("bank","")),
        "city":              str(txn.get("city","")),
        "merchant":          str(txn.get("merchant","")),
        "amount_lkr":        float(txn.get("amount_lkr",0)),
        "balance_lkr":       float(txn.get("balance_lkr",0)),
        "hour":              int(txn.get("hour",0)),
        "fraud_type":        str(txn.get("fraud_type","")),
        "fraud_probability": float(round(prob, 4)),
        "risk_level":        str(risk),
        "is_fraud_detected": bool(prob > 0.5),
        "process_time_ms":   float(process_time),
        "model":             "Random Forest",
        "decision":          "BLOCKED" if prob > 0.5 else "APPROVED",
    }

    # Log to pipeline log
    os.makedirs("logs", exist_ok=True)
    with open("logs/pipeline_log.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    # Send alert if fraud
    if prob > 0.5:
        try:
            from alert_system import trigger_alert
            trigger_alert(txn, prob,
                         fraud_type=txn.get("fraud_type","Unknown"),
                         send_email=True)
        except:
            pass

    return result

def get_pipeline_stats():
    """Get real-time pipeline statistics"""
    log_file = "logs/pipeline_log.jsonl"
    if not os.path.exists(log_file):
        return None

    records = []
    with open(log_file, "r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    if not records:
        return None

    df = pd.DataFrame(records)
    return {
        "total":          len(df),
        "fraud_count":    int(df["is_fraud_detected"].sum()),
        "fraud_rate":     round(df["is_fraud_detected"].mean() * 100, 2),
        "avg_time_ms":    round(df["process_time_ms"].mean(), 2),
        "total_amount":   round(df["amount_lkr"].sum(), 2),
        "fraud_amount":   round(df[df["is_fraud_detected"]]["amount_lkr"].sum(), 2),
        "by_bank":        df[df["is_fraud_detected"]]["bank"].value_counts().to_dict(),
        "by_city":        df[df["is_fraud_detected"]]["city"].value_counts().to_dict(),
        "recent":         records[-20:][::-1],
    }
