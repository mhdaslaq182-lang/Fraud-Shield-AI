"""
Web Push helper for Fraud Shield AI.
Sends a notification payload to a stored PushSubscription using pywebpush + VAPID.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from pywebpush import webpush, WebPushException

# ── Load VAPID config from env ────────────────────────────────────
VAPID_PUBLIC_KEY  = os.environ.get("VAPID_PUBLIC_KEY", "")
VAPID_CLAIM_EMAIL = os.environ.get("VAPID_CLAIM_EMAIL", "admin@example.com")
_priv_rel         = os.environ.get("VAPID_PRIVATE_KEY_PATH", "vapid_private.pem")
VAPID_PRIVATE_KEY_PATH = str(Path(__file__).parent / _priv_rel)


def _vapid_claims():
    return {"sub": f"mailto:{VAPID_CLAIM_EMAIL}"}


def send_push(subscription, payload):
    """
    Send a single push.  `subscription` is a dict {endpoint, keys:{p256dh,auth}}.
    Returns (success: bool, status_code_or_error: str).
    """
    if not os.path.exists(VAPID_PRIVATE_KEY_PATH):
        return False, "VAPID private key not configured"

    try:
        resp = webpush(
            subscription_info = subscription,
            data              = json.dumps(payload),
            vapid_private_key = VAPID_PRIVATE_KEY_PATH,
            vapid_claims      = _vapid_claims(),
            ttl               = 60,  # seconds — fraud alerts are time-sensitive
        )
        return True, str(resp.status_code)
    except WebPushException as e:
        # 404/410 = subscription expired/revoked — caller should delete it
        sc = e.response.status_code if e.response is not None else 0
        return False, f"WebPushException:{sc}:{e}"
    except Exception as e:
        return False, f"Error:{e}"


def broadcast_fraud_alert(prob, amount_lkr, fraud_type="Fraud detected", txn_id=None,
                          recipients=None):
    """
    Send a fraud-alert push to every stored subscription (or only `recipients`).
    Returns a summary dict: {sent, failed, expired_deleted}.
    Imported lazily by alert_system.trigger_alert.
    """
    from database import db, PushSubscription  # local import to avoid circulars

    risk = ("CRITICAL" if prob > 0.85 else
            "HIGH"     if prob > 0.65 else
            "MEDIUM"   if prob > 0.40 else "LOW")

    payload = {
        "title": f"🚨 {risk} Risk — {fraud_type}",
        "body":  f"LKR {float(amount_lkr):,.2f}  •  {prob*100:.1f}% fraud probability",
        "tag":   f"fraud-{txn_id or int(datetime.utcnow().timestamp())}",
        "url":   "/fraud/alerts",
        "icon":  "/static/images/icon-192.png",
        "badge": "/static/images/icon-192.png",
        "vibrate": [200, 100, 200],
        "requireInteraction": prob > 0.65,
        "renotify": True,
        "ts": datetime.utcnow().isoformat(),
    }

    q = PushSubscription.query
    if recipients:
        q = q.filter(PushSubscription.user_id.in_(recipients))
    subs = q.all()

    sent = failed = expired = 0
    for s in subs:
        ok, info = send_push(s.to_dict(), payload)
        if ok:
            sent += 1
            s.last_sent = datetime.utcnow()
            s.failures  = 0
        else:
            # Subscription gone — clean it up
            if "404" in info or "410" in info:
                db.session.delete(s)
                expired += 1
            else:
                s.failures = (s.failures or 0) + 1
                failed    += 1

    db.session.commit()
    return {"sent": sent, "failed": failed, "expired_deleted": expired,
            "total_subs": len(subs)}
