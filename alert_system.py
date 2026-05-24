import os
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

EMAIL_CONFIG = {
    "sender":   "mhdaslaq182@gmail.com",
    "password": "jlsgcpijznluftza",
    "receiver": "mhdaslaq182@gmail.com",
    "smtp":     "smtp.gmail.com",
    "port":     587,
}

def log_alert(transaction: dict, prob: float, fraud_type: str = "Unknown"):
    os.makedirs("alerts", exist_ok=True)
    amount = transaction.get("amount_lkr",
             transaction.get("amount", 0)) or 0
    entry = {
        "timestamp":   datetime.now().isoformat(),
        "fraud_prob":  round(prob, 4),
        "fraud_type":  fraud_type,
        "amount_lkr":  amount,
        "hour":        transaction.get("hour", 0),
        "distance_km": transaction.get("distance_km", 0),
        "countries":   transaction.get("countries", 1),
        "new_device":  transaction.get("new_device", 0),
        "email_risk":  transaction.get("email_risk", 0),
    }
    with open("alerts/fraud_alerts.log", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  📝 Alert logged → alerts/fraud_alerts.log")


def send_email_alert(transaction: dict, prob: float, fraud_type: str = "Unknown"):
    amount = transaction.get("amount_lkr",
             transaction.get("amount", 0)) or 0
    subject = f"🚨 FRAUD ALERT — {fraud_type} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    body = f"""
╔══════════════════════════════════════════╗
   🚨 SRI LANKA BANKING FRAUD ALERT 🇱🇰
╚══════════════════════════════════════════╝

Time           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Fraud Type     : {fraud_type}
Probability    : {prob:.1%}
Risk Level     : {"CRITICAL" if prob>0.85 else "HIGH" if prob>0.65 else "MEDIUM"}

─── Transaction Details ───────────────────
Amount         : Rs. {amount:,.2f}
Hour           : {transaction.get('hour', 0)}:00
Distance       : {transaction.get('distance_km', 0):.0f} km
Countries      : {transaction.get('countries', 1)}
New Device     : {'YES ⚠️' if transaction.get('new_device') else 'No'}
Failed Logins  : {transaction.get('failed_logins', 0)}
Email Risk     : {transaction.get('email_risk', 0):.2f}

⚠️  Please review this transaction immediately.
    """
    msg = MIMEMultipart()
    msg["From"]    = EMAIL_CONFIG["sender"]
    msg["To"]      = EMAIL_CONFIG["receiver"]
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP(EMAIL_CONFIG["smtp"], EMAIL_CONFIG["port"])
        server.starttls()
        server.login(EMAIL_CONFIG["sender"], EMAIL_CONFIG["password"])
        server.sendmail(EMAIL_CONFIG["sender"], EMAIL_CONFIG["receiver"], msg.as_string())
        server.quit()
        print(f"  📧 Email sent to {EMAIL_CONFIG['receiver']}")
        return True
    except Exception as e:
        print(f"  ⚠️  Email not sent (configure EMAIL_CONFIG): {e}")
        return False


def _send_push_alert(transaction: dict, prob: float, fraud_type: str):
    """Best-effort Web Push broadcast. Only runs inside an active Flask app context."""
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flask_app"))
        from flask import current_app  # noqa: F401  (will raise if no app ctx)
        from push_notify import broadcast_fraud_alert
        amount = transaction.get("amount_lkr", transaction.get("amount", 0)) or 0
        result = broadcast_fraud_alert(
            prob       = prob,
            amount_lkr = amount,
            fraud_type = fraud_type,
            txn_id     = transaction.get("transaction_id"),
        )
        print(f"  📱 Push: {result['sent']} sent, {result['failed']} failed, "
              f"{result['expired_deleted']} expired")
    except Exception as e:
        # No Flask app context, no subscriptions, or push disabled — silently skip
        print(f"  ⓘ  Push skipped: {e}")


def trigger_alert(transaction: dict, prob: float,
                  fraud_type: str = "Unknown",
                  send_email: bool = False,
                  send_push:  bool = True):
    amount = transaction.get("amount_lkr",
             transaction.get("amount", 0)) or 0
    print(f"\n  🚨 FRAUD ALERT TRIGGERED")
    print(f"  Type       : {fraud_type}")
    print(f"  Probability: {prob:.1%}")
    print(f"  Amount     : Rs. {amount:,.2f}")
    log_alert(transaction, prob, fraud_type)
    if send_email:
        send_email_alert(transaction, prob, fraud_type)
    if send_push:
        _send_push_alert(transaction, prob, fraud_type)


def show_alert_log():
    log_file = "alerts/fraud_alerts.log"
    if not os.path.exists(log_file):
        print("  No alerts logged yet.")
        return
    print("\n" + "="*55)
    print("  📋 FRAUD ALERT LOG")
    print("="*55)
    with open(log_file, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        try:
            entry = json.loads(line)
            amount = entry.get("amount_lkr",
                     entry.get("amount", 0)) or 0
            print(f"\n  Alert #{i}")
            print(f"  Time   : {entry.get('timestamp','N/A')}")
            print(f"  Type   : {entry.get('fraud_type','N/A')}")
            print(f"  Amount : Rs. {amount:,.2f}")
            print(f"  Risk   : {entry.get('fraud_prob',0):.1%}")
        except:
            pass
    print(f"\n  Total alerts: {len(lines)}")
    print("="*55)


if __name__ == "__main__":
    test_txn = {
        "amount_lkr":  250000.00,
        "hour":        2,
        "distance_km": 800.0,
        "countries":   3,
        "new_device":  1,
        "failed_logins": 6,
        "email_risk":  0.90,
    }
    trigger_alert(test_txn, prob=0.98,
                  fraud_type="Account Takeover",
                  send_email=False)
    show_alert_log()
