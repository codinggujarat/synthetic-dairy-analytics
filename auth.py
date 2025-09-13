# auth.py
import random
import datetime
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from flask_login import login_user, login_required, logout_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Message
from extensions import db, mail  # shared DB and Mail

auth_bp = Blueprint("auth", __name__)

# --------------------
# USER MODEL
# --------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)
    otp_code = db.Column(db.String(6), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# --------------------
# HELPERS
# --------------------
def generate_otp():
    return f"{random.randint(100000, 999999)}"

def send_otp_email(recipient, otp):
    msg = Message("Your OTP Code", recipients=[recipient])
    msg.body = f"Your OTP code is {otp}. It will expire in 5 minutes."
    mail.send(msg)

# --------------------
# ROUTES
# --------------------
@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].lower()
        existing = User.query.filter_by(email=email).first()
        if existing:
            flash("Email already registered. Please log in.", "danger")
            return redirect(url_for("auth.login"))

        otp = generate_otp()
        expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        session["signup_email"] = email
        session["signup_otp"] = otp
        session["signup_expiry"] = expiry.isoformat()

        try:
            send_otp_email(email, otp)
            flash("OTP sent to your email. Please verify.", "info")
        except Exception as e:
            current_app.logger.error(f"Email send failed: {e}")
            flash("Could not send OTP email. Check mail settings.", "danger")
            return redirect(url_for("auth.signup"))

        return redirect(url_for("auth.verify_signup"))

    return render_template("signup.html")

@auth_bp.route("/verify-signup", methods=["GET", "POST"])
def verify_signup():
    if request.method == "POST":
        otp_entered = request.form["otp"]
        email = session.get("signup_email")
        otp = session.get("signup_otp")
        expiry = session.get("signup_expiry")

        if not email or not otp:
            flash("Session expired. Start signup again.", "danger")
            return redirect(url_for("auth.signup"))

        if datetime.datetime.utcnow() > datetime.datetime.fromisoformat(expiry):
            flash("OTP expired. Start signup again.", "danger")
            return redirect(url_for("auth.signup"))

        if otp_entered == otp:
            user = User(email=email, is_verified=True)
            password = request.form.get("password")
            if password:
                user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Signup successful. Please log in.", "success")
            session.pop("signup_email", None)
            session.pop("signup_otp", None)
            return redirect(url_for("auth.login"))

        flash("Invalid OTP", "danger")

    return render_template("verify_signup.html")

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].lower()
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid credentials", "danger")
            return redirect(url_for("auth.login"))

        otp = generate_otp()
        user.otp_code = otp
        user.otp_expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=5)
        db.session.commit()

        try:
            send_otp_email(user.email, otp)
            flash("OTP sent to your email.", "info")
        except Exception as e:
            current_app.logger.error(f"Email send failed: {e}")
            flash("Could not send OTP email. Try again.", "danger")
            return redirect(url_for("auth.login"))

        session["login_email"] = email
        return redirect(url_for("auth.verify_login"))

    return render_template("login.html")

@auth_bp.route("/verify-login", methods=["GET", "POST"])
def verify_login():
    email = session.get("login_email")
    user = User.query.filter_by(email=email).first() if email else None
    if not user:
        flash("Session expired. Please login again.", "danger")
        return redirect(url_for("auth.login"))

    if request.method == "POST":
        otp_entered = request.form["otp"]
        if datetime.datetime.utcnow() > user.otp_expiry:
            flash("OTP expired. Please login again.", "danger")
            return redirect(url_for("auth.login"))

        if otp_entered == user.otp_code:
            login_user(user)
            flash("Logged in successfully", "success")
            user.otp_code = None
            db.session.commit()
            session.pop("login_email", None)
            return redirect(url_for("welcome"))

        flash("Invalid OTP", "danger")

    return render_template("verify_login.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out", "info")
    return redirect(url_for("auth.login"))
