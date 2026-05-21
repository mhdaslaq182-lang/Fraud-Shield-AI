
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from api_routes import api
from fraud_routes import fraud
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database import db, User, LoginLog
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
app.config["SECRET_KEY"]        = "sl_fraud_detection_2026_cg02_g07"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fraud_users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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
