
# ══════════════════════════════════════════
#   Bank API Integration
#   api_routes.py
# ══════════════════════════════════════════

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import engineer_features, FEATURES
from groq import Groq

api = Blueprint("api", __name__)

# ── API Keys for each bank ─────────────────────────────────────────
# Loaded from BANK_API_KEYS env var, formatted "KEY1:Bank Name 1,KEY2:Bank Name 2".
# Falls back to a hardcoded dev set if the env var is missing.
def _load_api_keys():
    raw = os.environ.get("BANK_API_KEYS", "").strip()
    if not raw:
        return {
            "BOC-2026-FRAUDSHIELD":       "Bank of Ceylon",
            "PEOPLES-2026-FRAUDSHIELD":   "Peoples Bank",
            "SAMPATH-2026-FRAUDSHIELD":   "Sampath Bank",
            "HNB-2026-FRAUDSHIELD":       "Hatton National Bank",
            "COMMERCIAL-2026-FRAUDSHIELD":"Commercial Bank",
            "NSB-2026-FRAUDSHIELD":       "National Savings Bank",
            "SEYLAN-2026-FRAUDSHIELD":    "Seylan Bank",
            "NTB-2026-FRAUDSHIELD":       "Nations Trust Bank",
        }
    out = {}
    for pair in raw.split(","):
        if ":" in pair:
            k, v = pair.split(":", 1)
            out[k.strip()] = v.strip()
    return out

API_KEYS = _load_api_keys()

# ── Load Model ──
def get_model():
    return pickle.load(open("../models/random_forest.pkl", "rb"))

# ── API Key Authentication ──
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not api_key or api_key not in API_KEYS:
            return jsonify({
                "error": "Invalid or missing API key",
                "code":  401
            }), 401
        request.bank_name = API_KEYS[api_key]
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════
# POST /api/v1/check-transaction
# ══════════════════════════════════════════
@api.route("/api/v1/check-transaction", methods=["POST"])
@require_api_key
def check_transaction():
    """
    Main fraud detection endpoint.
    Bank sends transaction → AI checks → Returns decision
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        required = ["amount_lkr", "hour", "distance_km",
                    "failed_logins", "new_device", "account_age",
                    "countries", "velocity_24h", "email_risk", "balance_lkr"]

        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}",
                "code":  400
            }), 400

        # Add defaults
        data.setdefault("frequency",  3)
        data.setdefault("is_weekend", 0)

        # AI Prediction
        start = datetime.now()
        model = get_model()
        df    = pd.DataFrame([data])
        df    = engineer_features(df)
        prob  = float(model.predict_proba(df[FEATURES])[0][1])
        ms    = round((datetime.now() - start).total_seconds() * 1000, 2)

        risk = "CRITICAL" if prob>0.85 else                "HIGH"     if prob>0.65 else                "MEDIUM"   if prob>0.40 else "LOW"

        decision = "BLOCKED" if prob > 0.5 else "APPROVED"

        # Send alert if fraud
        if prob > 0.5:
            try:
                sys.path.insert(0, "..")
                from alert_system import trigger_alert
                trigger_alert(data, prob,
                             fraud_type=f"API Fraud - {request.bank_name}",
                             send_email=True)
            except:
                pass

        # Log
        os.makedirs("../logs", exist_ok=True)
        with open("../logs/api_log.jsonl", "a") as f:
            import json
            f.write(json.dumps({
                "timestamp":    datetime.now().isoformat(),
                "bank":         request.bank_name,
                "amount":       float(data["amount_lkr"]),
                "decision":     decision,
                "probability":  round(prob, 4),
                "risk":         risk,
                "process_ms":   ms,
            }) + "\n")

        return jsonify({
            "status":            "success",
            "bank":              request.bank_name,
            "transaction_id":    data.get("transaction_id", f"TXN{int(datetime.now().timestamp())}"),
            "timestamp":         datetime.now().isoformat(),
            "decision":          decision,
            "fraud_probability": round(prob, 4),
            "fraud_percentage":  f"{prob*100:.1f}%",
            "risk_level":        risk,
            "action":            "Block this transaction immediately" if prob>0.5 else "Transaction is safe to proceed",
            "process_time_ms":   ms,
            "model":             "Random Forest + Deep Learning",
            "powered_by":        "Fraud Shield AI - Sri Lanka"
        })

    except Exception as e:
        return jsonify({"error": str(e), "code": 500}), 500


# ══════════════════════════════════════════
# POST /api/v1/batch-check
# ══════════════════════════════════════════
@api.route("/api/v1/batch-check", methods=["POST"])
@require_api_key
def batch_check():
    """
    Check multiple transactions at once.
    Bank sends list of transactions → AI checks all → Returns all decisions
    """
    try:
        data = request.get_json()
        transactions = data.get("transactions", [])

        if not transactions:
            return jsonify({"error": "No transactions provided"}), 400

        if len(transactions) > 1000:
            return jsonify({"error": "Maximum 1000 transactions per batch"}), 400

        model   = get_model()
        results = []
        fraud_count = 0

        for txn in transactions:
            txn.setdefault("frequency",  3)
            txn.setdefault("is_weekend", 0)

            try:
                df   = pd.DataFrame([txn])
                df   = engineer_features(df)
                prob = float(model.predict_proba(df[FEATURES])[0][1])
                risk = "CRITICAL" if prob>0.85 else                        "HIGH"     if prob>0.65 else                        "MEDIUM"   if prob>0.40 else "LOW"

                if prob > 0.5:
                    fraud_count += 1

                results.append({
                    "transaction_id":    txn.get("transaction_id", ""),
                    "decision":          "BLOCKED" if prob>0.5 else "APPROVED",
                    "fraud_probability": round(prob, 4),
                    "risk_level":        risk,
                })
            except Exception as e:
                results.append({
                    "transaction_id": txn.get("transaction_id",""),
                    "error": str(e)
                })

        return jsonify({
            "status":        "success",
            "bank":          request.bank_name,
            "total":         len(transactions),
            "fraud_detected":fraud_count,
            "legitimate":    len(transactions) - fraud_count,
            "fraud_rate":    f"{fraud_count/len(transactions)*100:.1f}%",
            "results":       results,
            "powered_by":    "Fraud Shield AI - Sri Lanka"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════
# GET /api/v1/status
# ══════════════════════════════════════════
@api.route("/api/v1/status", methods=["GET"])
def api_status():
    """Check if API is running"""
    return jsonify({
        "status":    "online",
        "service":   "Fraud Shield AI",
        "version":   "1.0.0",
        "country":   "Sri Lanka",
        "endpoints": [
            "POST /api/v1/check-transaction",
            "POST /api/v1/batch-check",
            "GET  /api/v1/status",
            "GET  /api/v1/api-keys",
        ],
        "timestamp": datetime.now().isoformat()
    })


# ══════════════════════════════════════════
# POST /api/chat  — Fraud Shield Chatbot
# ══════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are Fraud Shield AI Assistant, a knowledgeable and friendly expert chatbot for a banking fraud detection system built specifically for Sri Lanka.

You have deep expertise in:
- Fraud detection modules: Banking Fraud, E-Commerce Fraud, Mobile Payment Fraud, Insurance Fraud, Loan Fraud, Phishing Detection
- Risk levels: LOW (0-40%) safe, MEDIUM (40-65%) monitor, HIGH (65-85%) likely fraud, CRITICAL (85-100%) block immediately
- AI model: Random Forest + Deep Learning ensemble analysing 12+ features (amount, hour, distance, failed logins, new device, account age, countries, velocity, email risk, balance)
- Face biometric login using face_recognition library with live camera verification
- Automatic email alerts via Gmail SMTP for HIGH and CRITICAL detections (prob > 65%)
- Analytics dashboard: fraud patterns by city (Colombo, Kandy, Galle etc.), bank, hour of day, and fraud type
- Live Monitor: real-time transaction streaming with SSE showing AI decisions as they happen
- AI Pipeline: step-by-step processing view of each transaction
- Batch CSV Upload: upload hundreds of transactions, get fraud scores for all at once
- REST API: banks integrate using POST /api/v1/check-transaction with X-API-Key header
- SHAP Explainer: shows which features (amount, distance, hour etc.) caused the AI to flag a transaction
- Supported banks: BOC, Peoples Bank, Sampath, HNB, Commercial Bank, NSB, Seylan, NTB
- Sri Lanka context: covers fraud patterns in 60+ cities, typical LKR transaction amounts

Give thorough, detailed, and helpful responses. Use bullet points or numbered lists when explaining multiple things. Be conversational, warm, and professional. If the user asks a general question, give a complete answer with examples where useful. Only redirect to your area of expertise if the question is completely unrelated to fraud, banking, or technology."""

@api.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg  = str(data.get("message", "")).strip()
    history = data.get("history", [])
    if not msg:
        return jsonify({"reply": "Please type a message."}), 400
    try:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-6:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": msg})
        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, max_tokens=1024, temperature=0.7
        )
        reply = resp.choices[0].message.content.strip()
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"reply": f"Sorry, I'm having trouble connecting. Please try again. ({str(e)[:60]})"}), 500


# ══════════════════════════════════════════
# GET /api/v1/api-keys
# ══════════════════════════════════════════
@api.route("/api/v1/api-keys", methods=["GET"])
@require_api_key
def get_api_keys():
    """Get all available API keys"""
    return jsonify({
        "status": "success",
        "bank":   request.bank_name,
        "api_keys": {
            bank: key
            for key, bank in API_KEYS.items()
        }
    })
