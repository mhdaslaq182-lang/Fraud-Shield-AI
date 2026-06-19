
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from api_routes import api
from fraud_routes import fraud
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database import db, User, LoginLog, PushSubscription
from face_utils import register_face_from_image, verify_face_from_image, is_face_registered
from datetime import datetime
import os
import sys
from pathlib import Path
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = Flask(__name__)

# ── Secrets & session config (read from environment) ──
app.config["SECRET_KEY"] = os.environ.get(
    "FLASK_SECRET_KEY",
    "sl_fraud_detection_2026_cg02_g07"  # fallback for local dev only
)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///fraud_users.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ── Hardened cookie flags ──
_is_prod = os.environ.get("FLASK_ENV", "development").lower() == "production"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"]   = _is_prod   # require HTTPS in prod
app.config["REMEMBER_COOKIE_HTTPONLY"] = True
app.config["REMEMBER_COOKIE_SAMESITE"] = "Lax"
app.config["REMEMBER_COOKIE_SECURE"]   = _is_prod

db.init_app(app)
app.register_blueprint(api)
app.register_blueprint(fraud)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Create tables ──
with app.app_context():
    db.create_all()
    # Create default admin user
    if not User.query.filter_by(username="admin").first():
        admin = User(
            username  = "admin",
            email     = "admin@frauddetect.lk",
            full_name = "System Administrator",
            role      = "admin"
        )
        admin.set_password("admin123")
        db.session.add(admin)
        db.session.commit()
        print("Default admin created: admin / admin123")


# ══════════════════════════════════
# PWA: Service Worker + Manifest at root scope
# ══════════════════════════════════
@app.route("/sw.js")
def pwa_service_worker():
    """Serve the service worker from root so its scope is the whole site."""
    response = app.send_static_file("sw.js")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Content-Type"] = "application/javascript"
    return response

@app.route("/manifest.json")
def pwa_manifest():
    response = app.send_static_file("manifest.json")
    response.headers["Content-Type"] = "application/manifest+json"
    return response

@app.route("/offline")
def pwa_offline():
    """Fallback page shown by the service worker when the network is unreachable."""
    return render_template("offline.html")


# ══════════════════════════════════
# PWA: Web Push (subscribe / unsubscribe / public key / test)
# ══════════════════════════════════
@app.route("/api/push/public-key", methods=["GET"])
def push_public_key():
    """Return the server's VAPID public key — client uses it to call pushManager.subscribe()."""
    return jsonify({"publicKey": os.environ.get("VAPID_PUBLIC_KEY", "")})


@app.route("/api/push/subscribe", methods=["POST"])
@login_required
def push_subscribe():
    """Save a PushSubscription returned by the browser's pushManager."""
    sub = request.get_json(silent=True) or {}
    endpoint = sub.get("endpoint")
    keys     = sub.get("keys") or {}
    p256dh, auth = keys.get("p256dh"), keys.get("auth")

    if not endpoint or not p256dh or not auth:
        return jsonify({"success": False, "message": "Invalid subscription"}), 400

    existing = PushSubscription.query.filter_by(endpoint=endpoint).first()
    if existing:
        existing.user_id    = current_user.id
        existing.p256dh     = p256dh
        existing.auth       = auth
        existing.user_agent = request.headers.get("User-Agent", "")[:300]
        existing.failures   = 0
    else:
        existing = PushSubscription(
            user_id    = current_user.id,
            endpoint   = endpoint,
            p256dh     = p256dh,
            auth       = auth,
            user_agent = request.headers.get("User-Agent", "")[:300],
        )
        db.session.add(existing)
    db.session.commit()
    return jsonify({"success": True, "id": existing.id})


@app.route("/api/push/unsubscribe", methods=["POST"])
@login_required
def push_unsubscribe():
    endpoint = (request.get_json(silent=True) or {}).get("endpoint")
    if not endpoint:
        return jsonify({"success": False}), 400
    PushSubscription.query.filter_by(endpoint=endpoint).delete()
    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/push/test", methods=["POST"])
@login_required
def push_test():
    """Send a test notification to the logged-in user's devices. Useful for the
    'Send test notification' button on the profile page."""
    from push_notify import broadcast_fraud_alert
    result = broadcast_fraud_alert(
        prob       = 0.92,
        amount_lkr = 250000,
        fraud_type = "Test Alert",
        txn_id     = f"TEST-{int(datetime.utcnow().timestamp())}",
        recipients = [current_user.id],
    )
    return jsonify({"success": True, **result})


# ══════════════════════════════════
# HOME / LANDING PAGE
# ══════════════════════════════════
@app.route("/")
def home():
    return render_template("home.html")


# ══════════════════════════════════
# REGISTER
# ══════════════════════════════════
@app.route("/register", methods=["GET","POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username  = request.form.get("username",  "").strip()
        email     = request.form.get("email",     "").strip()
        full_name = request.form.get("full_name", "").strip()
        password  = request.form.get("password",  "")
        confirm   = request.form.get("confirm",   "")

        if not all([username, email, full_name, password]):
            flash("All fields are required!", "error")
            return render_template("register.html")

        if password != confirm:
            flash("Passwords do not match!", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters!", "error")
            return render_template("register.html")

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "error")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "error")
            return render_template("register.html")

        user = User(username=username, email=email, full_name=full_name)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Account created! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# ══════════════════════════════════
# LOGIN
# ══════════════════════════════════
@app.route("/login", methods=["GET","POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","")
        remember = request.form.get("remember", False)

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user, remember=remember)
            user.last_login = datetime.utcnow()
            log = LoginLog(user_id=user.id, username=username,
                          method="password", status="success",
                          ip_address=request.remote_addr)
            db.session.add(log)
            db.session.commit()
            flash(f"Welcome back, {user.full_name}!", "success")
            return redirect(url_for("dashboard"))
        else:
            log = LoginLog(username=username, method="password",
                          status="failed", ip_address=request.remote_addr)
            db.session.add(log)
            db.session.commit()
            flash("Invalid username or password!", "error")

    return render_template("login.html")


# ══════════════════════════════════
# FACE LOGIN
# ══════════════════════════════════
@app.route("/face-login")
def face_login():
    return render_template("face_login.html")

@app.route("/api/face-verify", methods=["POST"])
def api_face_verify():
    data     = request.get_json()
    username = data.get("username","")
    image    = data.get("image","")

    if not username or not image:
        return jsonify({"success":False, "message":"Missing data"})

    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"success":False, "message":"User not found"})

    success, message = verify_face_from_image(username, image)

    log = LoginLog(user_id=user.id if user else None,
                  username=username, method="face",
                  status="success" if success else "failed",
                  ip_address=request.remote_addr)
    db.session.add(log)

    if success:
        login_user(user)
        user.last_login = datetime.utcnow()
        db.session.commit()
        return jsonify({"success":True, "message":message, "redirect": url_for("dashboard")})
    else:
        db.session.commit()
        return jsonify({"success":False, "message":message})


# ══════════════════════════════════
# FACE REGISTER
# ══════════════════════════════════
@app.route("/api/face-register", methods=["POST"])
@login_required
def api_face_register():
    data  = request.get_json()
    image = data.get("image","")

    if not image:
        return jsonify({"success":False, "message":"No image provided"})

    success, message = register_face_from_image(current_user.username, image)

    if success:
        current_user.face_registered = True
        db.session.commit()

    return jsonify({"success":success, "message":message})


# ══════════════════════════════════
# DASHBOARD
# ══════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)


# ══════════════════════════════════
# ABOUT US
# ══════════════════════════════════
@app.route("/about")
def about():
    return render_template("about.html")


# ══════════════════════════════════
# PROFILE
# ══════════════════════════════════
@app.route("/profile")
@login_required
def profile():
    face_status = is_face_registered(current_user.username)
    return render_template("profile.html",
                          user=current_user,
                          face_status=face_status)


# ══════════════════════════════════
# LOGOUT
# ══════════════════════════════════
@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("RAILWAY_ENVIRONMENT") is None
    print("="*50)
    print("  Fraud Shield AI — Sri Lanka")
    print(f"  Running on port {port}")
    print("="*50)
    if debug:
        try:
            from pyngrok import ngrok
            public_url = ngrok.connect(port)
            print(f"  PUBLIC URL: {public_url.public_url}")
        except Exception:
            pass
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
