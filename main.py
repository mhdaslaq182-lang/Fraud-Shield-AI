# =============================================
#   AI BANKING FRAUD DETECTION SYSTEM
#   🇱🇰 Sri Lanka Edition
#   main.py — Master Pipeline
# =============================================

import os
import pickle
from sklearn.model_selection import train_test_split

print("""
╔══════════════════════════════════════════════════╗
   🛡️  AI BANKING FRAUD DETECTION SYSTEM
   🇱🇰  Sri Lanka Edition — CG02 Group 07
   Complete Pipeline Starting...
╚══════════════════════════════════════════════════╝
""")

# ── Step 1: Face Verification ──
print("─"*55)
print("  STEP 1: Face Recognition Login")
print("─"*55)
try:
    import face_recognition
    import pickle as pkl
    if os.path.exists("faces/encodings.pkl"):
        db = pkl.load(open("faces/encodings.pkl","rb"))
        print(f"  ✅ Face DB loaded: {list(db.keys())}")
        print(f"  ✅ Registered users: {len(db)}")
    else:
        print("  ⚠️  No faces registered yet")
        print("  Run: python3 -c \"from face_auth import register_face; register_face('admin')\"")
except Exception as e:
    print(f"  ⚠️  Face recognition skipped: {e}")

# ── Step 2: Generate Data ──
print("\n" + "─"*55)
print("  STEP 2: Generating Sri Lankan Banking Dataset 🇱🇰")
print("─"*55)
from data_generator import generate_banking_data
df = generate_banking_data(n_samples=10000, fraud_ratio=0.08)

# ── Step 3: Feature Engineering ──
print("\n" + "─"*55)
print("  STEP 3: Engineering Features")
print("─"*55)
from features import engineer_features, FEATURES
df = engineer_features(df)
print(f"  ✅ {len(FEATURES)} features ready")

# ── Step 4: Train/Test Split ──
print("\n" + "─"*55)
print("  STEP 4: Preparing Data Split")
print("─"*55)
X = df[FEATURES]
y = df["is_fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  ✅ Train : {len(X_train):,} samples")
print(f"  ✅ Test  : {len(X_test):,} samples")

# ── Step 5: Train Models ──
print("\n" + "─"*55)
print("  STEP 5: Training ML Models")
print("─"*55)
from models import train_all
trained = train_all(X_train, y_train)

# ── Step 6: Evaluate ──
print("\n" + "─"*55)
print("  STEP 6: Evaluating Models")
print("─"*55)
from evaluate import evaluate_all, plot_results
scaler  = pickle.load(open("models/scaler.pkl","rb"))
results = evaluate_all(trained, scaler, X_test, y_test)
plot_results(results, y_test, FEATURES)

# ── Step 7: SHAP Explainability ──
print("\n" + "─"*55)
print("  STEP 7: SHAP Explainability")
print("─"*55)
from explainer import explain_single
fraud_txn = {
    "amount_lkr": 250000, "hour": 2, "frequency": 20,
    "distance_km": 800.0, "failed_logins": 6,
    "new_device": 1, "account_age": 10,
    "countries": 3, "velocity_24h": 18,
    "email_risk": 0.90, "balance_lkr": 120000.0,
    "is_weekend": 0,
}
explain_single(fraud_txn, label="FRAUD TRANSACTION")

# ── Step 8: Predictions ──
print("\n" + "─"*55)
print("  STEP 8: Testing Predictions")
print("─"*55)
from predict import predict_transaction, print_result
legit_txn = {
    "amount_lkr": 4500, "hour": 14, "frequency": 3,
    "distance_km": 2.0, "failed_logins": 0,
    "new_device": 0, "account_age": 730,
    "countries": 1, "velocity_24h": 2,
    "email_risk": 0.05, "balance_lkr": 850000.0,
    "is_weekend": 0,
}
r1 = predict_transaction(fraud_txn)
r2 = predict_transaction(legit_txn)
print_result(r1, fraud_txn)
print_result(r2, legit_txn)

# ── Step 9: Live Monitor ──
print("\n" + "─"*55)
print("  STEP 9: Live Transaction Monitor")
print("─"*55)
from monitor import monitor_live
monitor_live(n_transactions=5, delay=0.5)

# ── Step 10: Alert System ──
print("\n" + "─"*55)
print("  STEP 10: Alert System")
print("─"*55)
from alert_system import trigger_alert, show_alert_log
trigger_alert(fraud_txn, prob=0.98,
              fraud_type="Credit Card Fraud",
              send_email=False)
show_alert_log()

# ── Step 11: AML Analysis ──
print("\n" + "─"*55)
print("  STEP 11: AML Graph Analysis")
print("─"*55)
from aml_graph import run_aml_analysis
run_aml_analysis()

# ── Done ──
print("""
╔══════════════════════════════════════════════════╗
   ✅ SYSTEM COMPLETE — ALL 11 STEPS DONE!
   🇱🇰 Sri Lanka Banking Fraud Detection System

   📁 Generated Files:
      data/banking_dataset.csv
      data/transaction_network.csv
      models/random_forest.pkl
      plots/evaluation_results.png
      plots/shap_importance.png
      plots/shap_beeswarm.png
      plots/aml_network.png
      alerts/fraud_alerts.log
      logs/monitor_log.csv

   🌐 Launch Dashboard:
      streamlit run app.py
      http://localhost:8501

   👥 CG02 — Group 07
╚══════════════════════════════════════════════════╝
""")
