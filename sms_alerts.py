
# =============================================
#   SMS Alert System — Twilio
#   sms_alerts.py
# =============================================

from twilio.rest import Client
import os
import json
from datetime import datetime

# ══════════════════════════════════════════
#   TWILIO CONFIG
#   Steps to get these:
#   1. Go to https://www.twilio.com
#   2. Sign up FREE
#   3. Get Account SID and Auth Token
#   4. Get a free Twilio phone number
# ══════════════════════════════════════════
TWILIO_CONFIG = {
    "account_sid":  "YOUR_ACCOUNT_SID",   # ← from twilio.com
    "auth_token":   "YOUR_AUTH_TOKEN",    # ← from twilio.com
    "from_number":  "+1XXXXXXXXXX",       # ← your Twilio number
    "to_number":    "+94XXXXXXXXX",       # ← your Sri Lanka number
}

def send_sms_alert(transaction: dict, prob: float,
                   fraud_type: str = "Unknown") -> bool:
    """Send SMS alert when fraud is detected."""

    amount = transaction.get("amount_lkr",
             transaction.get("amount", 0)) or 0

    risk = "CRITICAL" if prob>0.85 else            "HIGH"     if prob>0.65 else "MEDIUM"

    message = (
        f"FRAUD ALERT - {fraud_type}\n"
        f"Risk: {risk} ({prob:.0%})\n"
        f"Amount: Rs. {amount:,.0f}\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
        f"AI Fraud Detection System - Sri Lanka"
    )

    try:
        client = Client(
            TWILIO_CONFIG["account_sid"],
            TWILIO_CONFIG["auth_token"]
        )
        msg = client.messages.create(
            body=message,
            from_=TWILIO_CONFIG["from_number"],
            to=TWILIO_CONFIG["to_number"]
        )
        print(f"  📱 SMS sent! SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"  ⚠️  SMS failed: {e}")
        print(f"  ℹ️  Configure TWILIO_CONFIG to enable SMS")
        return False


def send_whatsapp_alert(transaction: dict, prob: float,
                        fraud_type: str = "Unknown") -> bool:
    """Send WhatsApp alert (Twilio WhatsApp sandbox)."""

    amount = transaction.get("amount_lkr",
             transaction.get("amount", 0)) or 0

    message = (
        f"🚨 *FRAUD ALERT*\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"*Type:* {fraud_type}\n"
        f"*Risk:* {'🔴 CRITICAL' if prob>0.85 else '🟠 HIGH' if prob>0.65 else '🟡 MEDIUM'}\n"
        f"*Probability:* {prob:.0%}\n"
        f"*Amount:* Rs. {amount:,.0f}\n"
        f"*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"_AI Fraud Detection System_\n"
        f"_Sri Lanka Edition_ 🇱🇰"
    )

    try:
        client = Client(
            TWILIO_CONFIG["account_sid"],
            TWILIO_CONFIG["auth_token"]
        )
        msg = client.messages.create(
            body=message,
            from_=f"whatsapp:{TWILIO_CONFIG['from_number']}",
            to=f"whatsapp:{TWILIO_CONFIG['to_number']}"
        )
        print(f"  💬 WhatsApp sent! SID: {msg.sid}")
        return True
    except Exception as e:
        print(f"  ⚠️  WhatsApp failed: {e}")
        return False


def trigger_all_alerts(transaction: dict, prob: float,
                       fraud_type: str = "Unknown",
                       send_sms: bool = False,
                       send_whatsapp: bool = False,
                       send_email: bool = False):
    """
    Trigger all alert channels.
    Always logs. Optionally sends SMS/WhatsApp/Email.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from alert_system import trigger_alert, log_alert

    amount = transaction.get("amount_lkr",
             transaction.get("amount", 0)) or 0

    print(f"\n  {'='*50}")
    print(f"  🚨 FRAUD DETECTED — SENDING ALERTS")
    print(f"  {'='*50}")
    print(f"  Type        : {fraud_type}")
    print(f"  Probability : {prob:.1%}")
    print(f"  Amount      : Rs. {amount:,.2f}")
    print(f"  Time        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {'-'*50}")

    # Always log
    log_alert(transaction, prob, fraud_type)
    print(f"  ✅ Logged to alerts/fraud_alerts.log")

    # SMS
    if send_sms:
        send_sms_alert(transaction, prob, fraud_type)

    # WhatsApp
    if send_whatsapp:
        send_whatsapp_alert(transaction, prob, fraud_type)

    # Email
    if send_email:
        from alert_system import send_email_alert
        send_email_alert(transaction, prob, fraud_type)

    print(f"  {'='*50}")


def show_twilio_setup():
    """Show how to set up Twilio."""
    print("""
╔══════════════════════════════════════════════╗
   📱 TWILIO SMS SETUP GUIDE
╚══════════════════════════════════════════════╝

Step 1: Go to https://www.twilio.com
Step 2: Click "Sign Up" — it's FREE
Step 3: Verify your phone number
Step 4: Go to Console Dashboard
Step 5: Copy these values:

   Account SID  → replace YOUR_ACCOUNT_SID
   Auth Token   → replace YOUR_AUTH_TOKEN

Step 6: Get a free phone number:
   Console → Phone Numbers → Get a Number
   Copy the number → replace +1XXXXXXXXXX

Step 7: Add YOUR Sri Lanka number:
   Replace +94XXXXXXXXX with your number
   Example: +94771234567

Step 8: In sms_alerts.py update TWILIO_CONFIG

═══════════════════════════════════════════════
For WhatsApp (Free Sandbox):
   Console → Messaging → Try it out → WhatsApp
   Follow the sandbox setup
═══════════════════════════════════════════════
""")


if __name__ == "__main__":
    print("="*55)
    print("  SMS ALERT SYSTEM TEST")
    print("  Sri Lanka Fraud Detection 🇱🇰")
    print("="*55)

    show_twilio_setup()

    # Test with dummy transaction
    test_txn = {
        "amount_lkr":    250000,
        "hour":          2,
        "distance_km":   800,
        "countries":     3,
        "new_device":    1,
        "failed_logins": 6,
        "email_risk":    0.90,
    }

    print("\n  Testing alert system (log only)...")
    trigger_all_alerts(
        transaction  = test_txn,
        prob         = 0.98,
        fraud_type   = "Credit Card Fraud",
        send_sms     = False,   # Set True after Twilio setup
        send_whatsapp= False,   # Set True after WhatsApp setup
        send_email   = False,   # Set True after Gmail setup
    )

    print("\n  ✅ SMS Alert System Ready!")
    print("  Configure TWILIO_CONFIG to enable real SMS")
