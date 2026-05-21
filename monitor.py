import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from features import engineer_features, FEATURES

def generate_live_transaction():
    is_fraud = np.random.random() < 0.10
    if is_fraud:
        fraud_type = np.random.choice(["credit_card","aml","account_takeover"])
        if fraud_type == "credit_card":
            return {
                "amount_lkr":    round(np.random.uniform(50000, 500000), 2),
                "hour":          int(np.random.choice([0,1,2,3,4,23])),
                "frequency":     int(np.random.poisson(18)),
                "distance_km":   round(np.random.uniform(300, 2000), 1),
                "failed_logins": int(np.random.randint(0, 3)),
                "new_device":    1,
                "account_age":   int(np.random.randint(1, 45)),
                "countries":     int(np.random.randint(2, 5)),
                "velocity_24h":  int(np.random.poisson(12)),
                "email_risk":    round(np.random.uniform(0.3, 0.7), 2),
                "balance_lkr":   round(np.random.uniform(10000, 200000), 2),
                "is_weekend":    int(np.random.choice([0,1])),
                "_type":         "💳 Credit Card Fraud"
            }
        elif fraud_type == "aml":
            return {
                "amount_lkr":    float(np.random.choice([99900, 499900, 999900])),
                "hour":          int(np.random.randint(0, 24)),
                "frequency":     int(np.random.poisson(25)),
                "distance_km":   round(np.random.uniform(100, 500), 1),
                "failed_logins": 0,
                "new_device":    0,
                "account_age":   int(np.random.randint(30, 200)),
                "countries":     int(np.random.randint(3, 6)),
                "velocity_24h":  int(np.random.poisson(20)),
                "email_risk":    round(np.random.uniform(0.1, 0.4), 2),
                "balance_lkr":   round(np.random.uniform(1000000, 10000000), 2),
                "is_weekend":    int(np.random.choice([0,1])),
                "_type":         "💰 Money Laundering"
            }
        else:
            return {
                "amount_lkr":    round(np.random.uniform(20000, 300000), 2),
                "hour":          int(np.random.choice([1,2,3,4])),
                "frequency":     int(np.random.poisson(10)),
                "distance_km":   round(np.random.uniform(500, 3000), 1),
                "failed_logins": int(np.random.randint(5, 15)),
                "new_device":    1,
                "account_age":   int(np.random.randint(365, 2000)),
                "countries":     int(np.random.randint(2, 4)),
                "velocity_24h":  int(np.random.poisson(15)),
                "email_risk":    round(np.random.uniform(0.6, 1.0), 2),
                "balance_lkr":   round(np.random.uniform(100000, 2000000), 2),
                "is_weekend":    int(np.random.choice([0,1])),
                "_type":         "👤 Account Takeover"
            }
    else:
        return {
            "amount_lkr":    round(np.random.exponential(8000), 2),
            "hour":          int(np.random.randint(8, 20)),
            "frequency":     int(np.random.poisson(4)),
            "distance_km":   round(np.random.exponential(10), 1),
            "failed_logins": int(np.random.choice([0,1], p=[0.97,0.03])),
            "new_device":    int(np.random.choice([0,1], p=[0.92,0.08])),
            "account_age":   int(np.random.randint(180, 3000)),
            "countries":     1,
            "velocity_24h":  int(np.random.poisson(2)),
            "email_risk":    round(np.random.uniform(0.0, 0.2), 2),
            "balance_lkr":   round(np.random.uniform(50000, 5000000), 2),
            "is_weekend":    int(np.random.choice([0,1], p=[0.7,0.3])),
            "_type":         "✅ Legitimate"
        }


def monitor_live(n_transactions=20, delay=1.0):
    model = pickle.load(open("models/random_forest.pkl", "rb"))

    print("\n" + "="*70)
    print("  🔴 LIVE BANKING FRAUD MONITOR 🇱🇰 — STARTED")
    print("="*70)
    print(f"  {'TIME':<10} {'TXN':>4} {'AMOUNT (Rs.)':>15} {'STATUS':<20} {'RISK':>6}  TYPE")
    print("-"*70)

    log         = []
    fraud_count = 0

    for i in range(n_transactions):
        txn      = generate_live_transaction()
        txn_type = txn.pop("_type")

        df       = pd.DataFrame([txn])
        df       = engineer_features(df)
        prob     = model.predict_proba(df[FEATURES])[0][1]
        is_fraud = prob > 0.5

        if is_fraud:
            fraud_count += 1

        status    = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
        risk      = f"{prob:.0%}"
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(f"  {timestamp:<10} #{i+1:>3} "
              f"Rs.{txn['amount_lkr']:>12,.2f} "
              f"{status:<20} {risk:>6}  {txn_type}")

        if prob > 0.85:
            print(f"  {'':10} ⚠️  HIGH RISK ALERT — Rs.{txn['amount_lkr']:,.2f}")

        log.append({
            **txn,
            "fraud_prob": round(prob, 4),
            "is_fraud":   int(is_fraud),
            "txn_type":   txn_type,
            "timestamp":  timestamp
        })
        time.sleep(delay)

    print("-"*70)
    print(f"\n  📊 SUMMARY")
    print(f"  Total Transactions : {n_transactions}")
    print(f"  Fraud Detected     : {fraud_count}")
    print(f"  Legitimate         : {n_transactions - fraud_count}")
    print(f"  Fraud Rate         : {fraud_count/n_transactions:.1%}")

    import os
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(log).to_csv("logs/monitor_log.csv", index=False)
    print(f"  ✅ Log saved: logs/monitor_log.csv")
    print("="*70)
    return log


if __name__ == "__main__":
    monitor_live(n_transactions=20, delay=0.8)
