import sys, os, pickle, json, re, io, time, threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, send_file, Response, stream_with_context
from flask_login import login_required

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import engineer_features, FEATURES
from alert_system import send_email_alert

fraud = Blueprint("fraud", __name__)
BASE  = os.path.join(os.path.dirname(__file__), "..")
_models = {}

# ── helpers ───────────────────────────────────────────────────────────────────

def _load(name):
    if name not in _models:
        path = os.path.join(BASE, "models", f"{name}.pkl")
        _models[name] = pickle.load(open(path, "rb")) if os.path.exists(path) else None
    return _models[name]

def _risk(p):
    if p > 0.85: return "CRITICAL"
    if p > 0.65: return "HIGH"
    if p > 0.40: return "MEDIUM"
    return "LOW"

def _log(data, prob, fraud_type):
    try:
        log_path = os.path.join(os.path.dirname(__file__), "alerts", "fraud_alerts.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        entry = {"timestamp": datetime.now().isoformat(), "fraud_type": fraud_type,
                 "probability": round(prob, 4), "risk_level": _risk(prob)}
        entry.update({k: v for k, v in data.items() if isinstance(v, (int, float, str, bool))})
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if prob > 0.65:
            threading.Thread(target=send_email_alert, args=(data, prob, fraud_type), daemon=True).start()
    except Exception:
        pass

def _ok(prob, extra=None):
    r = {"prob": prob, "pct": f"{prob:.1%}", "risk": _risk(prob),
         "fraud": prob > 0.5, "decision": "BLOCKED" if prob > 0.5 else "APPROVED"}
    if extra:
        r.update(extra)
    return jsonify(r)

# ── BANKING FRAUD ─────────────────────────────────────────────────────────────

@fraud.route("/fraud/banking")
@login_required
def banking():
    return render_template("fraud/banking.html", active="banking")

@fraud.route("/fraud/banking/predict", methods=["POST"])
@login_required
def banking_predict():
    d = request.get_json()
    try:
        txn = {
            "amount_lkr":    float(d.get("amount_lkr", 15000)),
            "hour":          int(d.get("hour", 14)),
            "frequency":     int(d.get("frequency", 3)),
            "distance_km":   float(d.get("distance_km", 5)),
            "failed_logins": int(d.get("failed_logins", 0)),
            "new_device":    int(d.get("new_device", 0)),
            "account_age":   int(d.get("account_age", 365)),
            "countries":     int(d.get("countries", 1)),
            "velocity_24h":  int(d.get("velocity_24h", 2)),
            "email_risk":    float(d.get("email_risk", 0.1)),
            "balance_lkr":   float(d.get("balance_lkr", 500000)),
            "is_weekend":    int(d.get("is_weekend", 0)),
        }
        m    = _load("random_forest")
        df   = engineer_features(pd.DataFrame([txn]))
        prob = float(m.predict_proba(df[FEATURES])[0][1])
        if prob > 0.5:
            _log(txn, prob, "Banking Fraud")
        return _ok(prob)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── E-COMMERCE ────────────────────────────────────────────────────────────────

@fraud.route("/fraud/ecommerce")
@login_required
def ecommerce():
    return render_template("fraud/ecommerce.html", active="ecommerce")

@fraud.route("/fraud/ecommerce/predict", methods=["POST"])
@login_required
def ecommerce_predict():
    d = request.get_json()
    try:
        X = pd.DataFrame([{
            "order_amount":      float(d.get("order_amount", 2500)),
            "items_count":       int(d.get("items_count", 2)),
            "hour":              int(d.get("hour", 14)),
            "is_new_customer":   int(d.get("is_new_customer", 0)),
            "failed_payments":   int(d.get("failed_payments", 0)),
            "different_address": int(d.get("different_address", 0)),
            "device_changes":    int(d.get("device_changes", 0)),
            "return_rate":       float(d.get("return_rate", 0.05)),
            "account_age_days":  int(d.get("account_age_days", 200)),
            "promo_abuse":       int(d.get("promo_abuse", 0)),
        }])
        m = _load("ecommerce_model")
        if not m:
            return jsonify({"error": "Model not found — run multi_domain.py first"}), 404
        prob = float(m.predict_proba(X)[0][1])
        if prob > 0.5:
            _log(d, prob, "E-Commerce Fraud")
        return _ok(prob)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── MOBILE PAYMENT ────────────────────────────────────────────────────────────

@fraud.route("/fraud/mobile")
@login_required
def mobile():
    return render_template("fraud/mobile.html", active="mobile")

@fraud.route("/fraud/mobile/predict", methods=["POST"])
@login_required
def mobile_predict():
    d = request.get_json()
    try:
        X = pd.DataFrame([{
            "amount_lkr":        float(d.get("amount_lkr", 500)),
            "hour":              int(d.get("hour", 12)),
            "top_up_frequency":  int(d.get("top_up_frequency", 3)),
            "sim_age_days":      int(d.get("sim_age_days", 365)),
            "is_roaming":        int(d.get("is_roaming", 0)),
            "pin_attempts":      int(d.get("pin_attempts", 0)),
            "receiver_known":    int(d.get("receiver_known", 1)),
            "transaction_speed": float(d.get("transaction_speed", 2.0)),
        }])
        m = _load("mobile_payment_model")
        if not m:
            return jsonify({"error": "Model not found"}), 404
        prob = float(m.predict_proba(X)[0][1])
        if prob > 0.5:
            _log(d, prob, "Mobile Payment Fraud")
        return _ok(prob)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── INSURANCE ─────────────────────────────────────────────────────────────────

@fraud.route("/fraud/insurance")
@login_required
def insurance():
    return render_template("fraud/insurance.html", active="insurance")

@fraud.route("/fraud/insurance/predict", methods=["POST"])
@login_required
def insurance_predict():
    d = request.get_json()
    try:
        X = pd.DataFrame([{
            "claim_amount":      float(d.get("claim_amount", 50000)),
            "policy_age_days":   int(d.get("policy_age_days", 500)),
            "previous_claims":   int(d.get("previous_claims", 0)),
            "documents_missing": int(d.get("documents_missing", 0)),
            "claim_speed_days":  int(d.get("claim_speed_days", 30)),
            "witness_count":     int(d.get("witness_count", 1)),
            "injury_severity":   float(d.get("injury_severity", 0.2)),
            "lawyer_involved":   int(d.get("lawyer_involved", 0)),
        }])
        m = _load("insurance_model")
        if not m:
            return jsonify({"error": "Model not found"}), 404
        prob = float(m.predict_proba(X)[0][1])
        if prob > 0.5:
            _log(d, prob, "Insurance Fraud")
        return _ok(prob)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── LOAN FRAUD ────────────────────────────────────────────────────────────────

@fraud.route("/fraud/loan")
@login_required
def loan():
    return render_template("fraud/loan.html", active="loan")

@fraud.route("/fraud/loan/predict", methods=["POST"])
@login_required
def loan_predict():
    d = request.get_json()
    try:
        X = pd.DataFrame([{
            "loan_amount":       float(d.get("loan_amount", 200000)),
            "monthly_income":    float(d.get("monthly_income", 80000)),
            "credit_score":      int(d.get("credit_score", 650)),
            "employment_years":  int(d.get("employment_years", 5)),
            "existing_loans":    int(d.get("existing_loans", 1)),
            "age":               int(d.get("age", 35)),
            "address_changes":   int(d.get("address_changes", 0)),
            "doc_inconsistency": int(d.get("doc_inconsistency", 0)),
            "multiple_apps":     int(d.get("multiple_apps", 0)),
        }])
        m = _load("online_loan_model")
        if not m:
            return jsonify({"error": "Model not found"}), 404
        prob = float(m.predict_proba(X)[0][1])
        if prob > 0.5:
            _log(d, prob, "Loan Fraud")
        return _ok(prob)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── PHISHING ──────────────────────────────────────────────────────────────────

@fraud.route("/fraud/phishing")
@login_required
def phishing():
    return render_template("fraud/phishing.html", active="phishing")

@fraud.route("/fraud/phishing/analyze", methods=["POST"])
@login_required
def phishing_analyze():
    d = request.get_json()
    text = d.get("text", "")
    try:
        KEYWORDS = ["urgent","verify your account","click here","password","suspended",
                    "unusual activity","confirm your identity","won a prize","bank details",
                    "wire transfer","act now","limited time","security alert",
                    "update your information","dear customer","free gift","congratulations"]
        matched  = [kw for kw in KEYWORDS if kw in text.lower()]
        urls     = re.findall(r"https?://\S+", text)
        bad_urls = [u for u in urls if any(x in u for x in ["bit.ly","tinyurl","secure-","login-","verify-"])]
        score    = min(len(matched) * 12 + len(bad_urls) * 25, 100)
        if score > 40:
            _log({"text_length": len(text)}, score / 100, "Phishing Attack")
        return jsonify({"score": score, "is_phishing": score > 40, "matched": matched,
                        "bad_urls": bad_urls, "risk": _risk(score / 100)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── LIVE MONITOR (streams served under /dashboard/) ──────────────────────────

@fraud.route("/dashboard/monitor/stream")
@login_required
def monitor_stream():
    delay = float(request.args.get("delay", 0.5))

    def generate():
        try:
            m = _load("random_forest")
            BANKS  = ["BOC", "Peoples", "Commercial", "Sampath", "HNB", "NSB", "Seylan", "NTB"]
            CITIES = [
                "Colombo","Kandy","Galle","Jaffna","Negombo","Kurunegala","Ratnapura","Matara",
                "Anuradhapura","Trincomalee","Batticaloa","Badulla","Nuwara Eliya","Kalutara",
                "Gampaha","Polonnaruwa","Kegalle","Hambantota","Puttalam","Vavuniya","Ampara",
                "Dambulla","Matale","Chilaw","Weligama","Ambalangoda","Hikkaduwa","Tangalle",
                "Monaragala","Bandarawela","Haputale","Ella","Hatton","Embilipitiya","Moratuwa",
                "Panadura","Horana","Avissawella","Kalmunai","Tissamaharama","Kataragama",
                "Nawalapitiya","Gampola","Welimada","Kilinochchi","Mannar","Mullaitivu",
                "Nikaweratiya","Kuliyapitiya","Maho","Wariyapola","Alawwa","Rambukkana",
                "Mawanella","Sigiriya","Dambulla","Galewela","Mahiyanganaya","Kekirawa",
            ]
            fraud_count = 0
            i = 0
            while True:
                sim = np.random.random() < 0.10
                txn = {
                    "amount_lkr":    round(np.random.uniform(50000, 500000) if sim else max(100, np.random.exponential(8000)), 2),
                    "hour":          int(np.random.choice([1, 2, 3]) if sim else np.random.randint(8, 20)),
                    "frequency":     int(np.random.poisson(18 if sim else 4)),
                    "distance_km":   round(np.random.uniform(300, 2000) if sim else max(0, np.random.exponential(10)), 1),
                    "failed_logins": int(np.random.randint(4, 10) if sim else np.random.choice([0, 1], p=[0.97, 0.03])),
                    "new_device":    int(1 if sim else np.random.choice([0, 1], p=[0.92, 0.08])),
                    "account_age":   int(np.random.randint(1, 45) if sim else np.random.randint(180, 3000)),
                    "countries":     int(np.random.randint(2, 5) if sim else 1),
                    "velocity_24h":  int(np.random.poisson(12 if sim else 2)),
                    "email_risk":    round(np.random.uniform(0.3, 0.7) if sim else np.random.uniform(0, 0.2), 2),
                    "balance_lkr":   round(np.random.uniform(10000, 200000) if sim else np.random.uniform(50000, 5000000), 2),
                    "is_weekend":    int(np.random.choice([0, 1])),
                }
                df       = engineer_features(pd.DataFrame([txn]))
                prob     = float(m.predict_proba(df[FEATURES])[0][1])
                is_fraud = prob > 0.5
                if is_fraud:
                    fraud_count += 1
                    _log(txn, prob, "Live Monitor Fraud")
                i += 1
                row = {
                    "num": i,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "amount": f"Rs. {txn['amount_lkr']:,.0f}",
                    "bank": str(np.random.choice(BANKS)),
                    "city": str(np.random.choice(CITIES)),
                    "status": "FRAUD" if is_fraud else "SAFE",
                    "risk": f"{prob:.0%}",
                    "prob_raw": round(prob * 100, 1),
                    "is_fraud": is_fraud,
                    "fraud_count": fraud_count,
                }
                yield f"data: {json.dumps(row)}\n\n"
                time.sleep(max(0.1, delay))
        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── PIPELINE STREAM ───────────────────────────────────────────────────────────

@fraud.route("/dashboard/pipeline/stream")
@login_required
def pipeline_stream():
    delay = float(request.args.get("delay", 0.4))

    def generate():
        try:
            sys.path.insert(0, BASE)
            from pipeline import generate_transaction, process_transaction
            m = _load("random_forest")
            fraud_count, total_ms = 0, 0
            i = 0
            while True:
                txn = generate_transaction()
                r   = process_transaction(txn, m)
                if r["is_fraud_detected"]:
                    fraud_count += 1
                    _log(txn, r["fraud_probability"], "Pipeline")
                total_ms += r["process_time_ms"]
                i += 1
                row = {
                    "num": i,
                    "time": r["timestamp"][11:19],
                    "bank": r["bank"], "city": r["city"],
                    "amount": f"Rs. {r['amount_lkr']:,.0f}",
                    "decision": r["decision"],
                    "risk": r["risk_level"],
                    "prob": f"{r['fraud_probability']:.1%}",
                    "prob_raw": round(r["fraud_probability"] * 100, 1),
                    "ms": r["process_time_ms"],
                    "is_fraud": r["is_fraud_detected"],
                    "fraud_count": fraud_count,
                    "avg_ms": round(total_ms / i, 1),
                }
                yield f"data: {json.dumps(row)}\n\n"
                time.sleep(max(0.1, delay))
        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── ANALYTICS ─────────────────────────────────────────────────────────────────

@fraud.route("/fraud/analytics")
@login_required
def analytics():
    return render_template("fraud/analytics.html", active="analytics")

@fraud.route("/fraud/analytics/data")
@login_required
def analytics_data():
    try:
        df = pd.read_csv(os.path.join(BASE, "data", "banking_dataset.csv"))
        df = engineer_features(df)
        city_fraud = df[df.is_fraud == 1]["city"].value_counts().head(10)
        bank_fraud = df[df.is_fraud == 1]["bank"].value_counts()
        hl = df[df.is_fraud == 0].groupby("hour").size().to_dict()
        hf = df[df.is_fraud == 1].groupby("hour").size().to_dict()
        fraud_types = df[df.is_fraud == 1]["fraud_type"].value_counts().to_dict() if "fraud_type" in df.columns else {}
        return jsonify({
            "city_labels": city_fraud.index.tolist(),
            "city_values": city_fraud.values.tolist(),
            "bank_labels": bank_fraud.index.tolist(),
            "bank_values": bank_fraud.values.tolist(),
            "hours": list(range(24)),
            "hourly_legit": [hl.get(h, 0) for h in range(24)],
            "hourly_fraud": [hf.get(h, 0) for h in range(24)],
            "fraud_types": fraud_types,
            "total": len(df), "fraud_count": int(df.is_fraud.sum()),
            "fraud_rate": f"{df.is_fraud.mean():.1%}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── SHAP EXPLAINER ────────────────────────────────────────────────────────────

@fraud.route("/fraud/shap")
@login_required
def shap_page():
    return render_template("fraud/shap.html", active="shap")

@fraud.route("/fraud/shap/explain", methods=["POST"])
@login_required
def shap_explain():
    d = request.get_json()
    try:
        import shap as shap_lib
        txn = {
            "amount_lkr":    float(d.get("amount_lkr", 250000)),
            "hour":          int(d.get("hour", 2)),
            "frequency":     5,
            "distance_km":   float(d.get("distance_km", 800)),
            "failed_logins": int(d.get("failed_logins", 6)),
            "new_device":    int(d.get("new_device", 1)),
            "account_age":   30,
            "countries":     int(d.get("countries", 3)),
            "velocity_24h":  int(d.get("velocity_24h", 18)),
            "email_risk":    float(d.get("email_risk", 0.9)),
            "balance_lkr":   500000,
            "is_weekend":    0,
        }
        m    = _load("random_forest")
        df   = engineer_features(pd.DataFrame([txn]))
        X    = df[FEATURES]
        prob = float(m.predict_proba(X)[0][1])
        exp  = shap_lib.TreeExplainer(m)
        sv   = exp.shap_values(X)
        fshap = sv[:, :, 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else (sv[1] if isinstance(sv, list) else sv)
        contribs = sorted(zip(FEATURES, fshap[0].tolist()), key=lambda x: abs(x[1]), reverse=True)
        raw = {f: float(X[f].values[0]) for f in FEATURES}
        return jsonify({"prob": prob, "pct": f"{prob:.1%}", "fraud": prob > 0.5,
                        "contributions": [{"feature": f, "value": round(v, 4), "raw": round(raw[f], 2)}
                                          for f, v in contribs[:10]]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── ALERT LOG ─────────────────────────────────────────────────────────────────

@fraud.route("/fraud/alerts")
@login_required
def alerts_page():
    return render_template("fraud/alerts.html", active="alerts")

@fraud.route("/fraud/alerts/data")
@login_required
def alerts_data():
    log_path = os.path.join(os.path.dirname(__file__), "alerts", "fraud_alerts.log")
    alerts = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    alerts.append(json.loads(line.strip()))
                except Exception:
                    pass
    return jsonify({"alerts": list(reversed(alerts)), "total": len(alerts)})

@fraud.route("/fraud/alerts/clear", methods=["POST"])
@login_required
def alerts_clear():
    log_path = os.path.join(os.path.dirname(__file__), "alerts", "fraud_alerts.log")
    if os.path.exists(log_path):
        open(log_path, "w").close()
    return jsonify({"success": True})

# ── BANK DATA UPLOAD ──────────────────────────────────────────────────────────

@fraud.route("/fraud/upload")
@login_required
def upload():
    return render_template("fraud/upload.html", active="upload")

@fraud.route("/fraud/upload/analyze", methods=["POST"])
@login_required
def upload_analyze():
    try:
        f = request.files.get("file")
        if not f:
            return jsonify({"error": "No file uploaded"}), 400
        df = pd.read_csv(f)
        required = ["amount_lkr", "hour", "distance_km", "failed_logins",
                    "new_device", "account_age", "countries", "velocity_24h",
                    "email_risk", "balance_lkr"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400
        df.setdefault("frequency", 3)
        if "frequency" not in df.columns:
            df["frequency"] = 3
        if "is_weekend" not in df.columns:
            df["is_weekend"] = 0
        m = _load("random_forest")
        results = []
        for _, row in df.iterrows():
            txn  = row.to_dict()
            df_t = engineer_features(pd.DataFrame([txn]))
            for feat in FEATURES:
                if feat not in df_t.columns:
                    df_t[feat] = 0
            prob = float(m.predict_proba(df_t[FEATURES])[0][1])
            results.append({"amount_lkr": round(float(txn.get("amount_lkr", 0)), 2),
                             "bank": str(txn.get("bank", "")), "city": str(txn.get("city", "")),
                             "fraud_probability": round(prob, 4), "risk_level": _risk(prob),
                             "is_fraud": prob > 0.5,
                             "decision": "BLOCKED" if prob > 0.5 else "APPROVED"})
        fraud_r = [r for r in results if r["is_fraud"]]
        return jsonify({"total": len(results), "fraud_count": len(fraud_r),
                        "legit_count": len(results) - len(fraud_r),
                        "fraud_rate": f"{len(fraud_r) / len(results) * 100:.1f}%",
                        "results": results, "fraud_results": fraud_r})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@fraud.route("/fraud/upload/template")
@login_required
def upload_template():
    sample = pd.DataFrame({
        "transaction_id": ["TXN001", "TXN002", "TXN003"],
        "amount_lkr":     [15000, 250000, 5000],
        "hour":           [14, 2, 10],
        "distance_km":    [5, 800, 2],
        "failed_logins":  [0, 6, 0],
        "new_device":     [0, 1, 0],
        "account_age":    [365, 10, 1000],
        "countries":      [1, 3, 1],
        "velocity_24h":   [2, 18, 1],
        "email_risk":     [0.1, 0.9, 0.05],
        "balance_lkr":    [500000, 120000, 250000],
        "is_weekend":     [0, 0, 1],
        "bank":           ["BOC", "Sampath", "HNB"],
        "city":           ["Colombo", "Kandy", "Galle"],
    })
    buf = io.BytesIO()
    sample.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(buf, mimetype="text/csv", as_attachment=True, download_name="bank_template.csv")

# ── API TESTER ────────────────────────────────────────────────────────────────

@fraud.route("/fraud/api-tester")
@login_required
def api_tester():
    return render_template("fraud/api_tester.html", active="api_tester")
