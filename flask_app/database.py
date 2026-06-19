from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id               = db.Column(db.Integer,     primary_key=True)
    username         = db.Column(db.String(80),  unique=True, nullable=False)
    email            = db.Column(db.String(120), unique=True, nullable=False)
    password_hash    = db.Column(db.String(200), nullable=False)
    full_name        = db.Column(db.String(120), nullable=False)
    role             = db.Column(db.String(20),  default="analyst")
    face_registered  = db.Column(db.Boolean,     default=False)
    created_at       = db.Column(db.DateTime,    default=datetime.utcnow)
    last_login       = db.Column(db.DateTime,    nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"

class LoginLog(db.Model):
    id         = db.Column(db.Integer,  primary_key=True)
    user_id    = db.Column(db.Integer,  db.ForeignKey("user.id"), nullable=True)
    username   = db.Column(db.String(80))
    login_time = db.Column(db.DateTime, default=datetime.utcnow)
    method     = db.Column(db.String(20))
    status     = db.Column(db.String(20))
    ip_address = db.Column(db.String(50))

    def __repr__(self):
        return f"<Login {self.username} {self.status}>"


class PushSubscription(db.Model):
    """Stores a Web Push subscription returned by the browser.
    One row per (user, device). `endpoint` is unique per browser/device."""
    id          = db.Column(db.Integer,    primary_key=True)
    user_id     = db.Column(db.Integer,    db.ForeignKey("user.id"), nullable=False)
    endpoint    = db.Column(db.String(500), unique=True, nullable=False)
    p256dh      = db.Column(db.String(200), nullable=False)
    auth        = db.Column(db.String(100), nullable=False)
    user_agent  = db.Column(db.String(300))
    created_at  = db.Column(db.DateTime,   default=datetime.utcnow)
    last_sent   = db.Column(db.DateTime,   nullable=True)
    failures    = db.Column(db.Integer,    default=0)

    def to_dict(self):
        return {
            "endpoint": self.endpoint,
            "keys": {"p256dh": self.p256dh, "auth": self.auth},
        }

    def __repr__(self):
        return f"<PushSub user={self.user_id} endpoint={self.endpoint[:32]}…>"
