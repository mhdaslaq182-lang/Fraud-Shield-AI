import pandas as pd
import numpy as np
import pickle
from features import engineer_features, FEATURES

def predict_transaction(transaction: dict, model_name="random_forest") -> dict:
    model  = pickle.load(open(f"models/{model_name}.pkl", "rb"))
    scaler = pickle.load(open("models/scaler.pkl", "rb"))

    df   = pd.DataFrame([transaction])
    df   = engineer_features(df)
    X    = scaler.transform(df[FEATURES]) if model_name == "logistic" else df[FEATURES]

    prob       = model.predict_proba(X)[0][1]
    is_fraud   = prob > 0.5
    risk_level = "🔴 CRITICAL" if prob > 0.85 else \
                 "🟠 HIGH"     if prob > 0.65 else \
                 "🟡 MEDIUM"   if prob > 0.40 else \
                 "🟢 LOW"

    reasons = []
    if transaction.get("amount_lkr", 0) > 100000:
        reasons.append(f"Large amount: Rs. {transaction['amount_lkr']:,.2f}")
    if transaction.get("hour", 12) <= 5:
        reasons.append(f"Night transaction: {transaction['hour']}:00")
    if transaction.get("failed_logins", 0) >= 3:
        reasons.append(f"Failed logins: {transaction['failed_logins']}")
    if transaction.get("new_device", 0):
        reasons.append("New device used")
    if transaction.get("countries", 1) > 1:
        reasons.append(f"Multiple countries: {transaction['countries']}")
    if transaction.get("velocity_24h", 0) > 10:
        reasons.append(f"High velocity: {transaction['velocity_24h']} txns/24h")
    if transaction.get("email_risk", 0) > 0.6:
        reasons.append(f"Risky email domain: {transaction['email_risk']:.2f}")
    if transaction.get("distance_km", 0) > 200:
        reasons.append(f"Far from home: {transaction['distance_km']:.0f}km")

    return {
        "is_fraud":          is_fraud,
        "fraud_probability": round(float(prob), 4),
        "risk_level":        risk_level,
        "reasons":           reasons if is_fraud else ["No suspicious signals detected"],
        "model_used":        model_name,
    }


def print_result(result: dict, transaction: dict):
    print("\n" + "="*55)
    print("  🇱🇰 FRAUD DETECTION RESULT")
    print("="*55)
    print(f"  Amount       : Rs. {transaction.get('amount_lkr', 0):,.2f}")
    print(f"  Hour         : {transaction.get('hour', 0)}:00")
    print(f"  Distance     : {transaction.get('distance_km', 0):.0f} km")
    print(f"  Failed Logins: {transaction.get('failed_logins', 0)}")
    print(f"  New Device   : {'Yes' if transaction.get('new_device') else 'No'}")
    print("-"*55)
    status = "🚨 FRAUD DETECTED" if result["is_fraud"] else "✅ LEGITIMATE"
    print(f"  Status       : {status}")
    print(f"  Probability  : {result['fraud_probability']:.1%}")
    print(f"  Risk Level   : {result['risk_level']}")
    print(f"  Model Used   : {result['model_used']}")
    print("-"*55)
    print("  Reasons:")
    for r in result["reasons"]:
        print(f"    • {r}")
    print("="*55)


if __name__ == "__main__":
    print("\n🧪 TEST 1 — Suspicious Transaction (Rs. 250,000)")
    fraud_txn = {
        "amount_lkr": 250000.00, "hour": 2, "frequency": 20,
        "distance_km": 800.0, "failed_logins": 6, "new_device": 1,
        "account_age": 10, "countries": 3, "velocity_24h": 18,
        "email_risk": 0.90, "balance_lkr": 120000.0, "is_weekend": 0,
    }
    result = predict_transaction(fraud_txn)
    print_result(result, fraud_txn)

    print("\n🧪 TEST 2 — Normal Transaction (Rs. 4,500)")
    legit_txn = {
        "amount_lkr": 4500.00, "hour": 14, "frequency": 3,
        "distance_km": 2.0, "failed_logins": 0, "new_device": 0,
        "account_age": 730, "countries": 1, "velocity_24h": 2,
        "email_risk": 0.05, "balance_lkr": 850000.0, "is_weekend": 0,
    }
    result = predict_transaction(legit_txn)
    print_result(result, legit_txn)
