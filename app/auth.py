# auth.py - User Authentication Module

from flask import request, render_template, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import sqlite3
import os
import secrets

_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'users.db')
_REGISTER_TEMPLATE = 'auth/register.html'


def _get_conn():
    return sqlite3.connect(_DB_PATH)


def init_db():
    conn = _get_conn()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS diagnosis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symptoms TEXT,
            predicted_disease TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_feedback TEXT,
            rating INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()


class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

    @staticmethod
    def get(user_id):
        conn = _get_conn()
        c = conn.cursor()
        c.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,))
        row = c.fetchone()
        conn.close()
        return User(row[0], row[1], row[2]) if row else None

    @staticmethod
    def get_by_email(email):
        conn = _get_conn()
        c = conn.cursor()
        c.execute('SELECT id, username, email, password_hash FROM users WHERE email = ?', (email,))
        row = c.fetchone()
        conn.close()
        return row

    @staticmethod
    def create_user(username, email, password):
        try:
            conn = _get_conn()
            c = conn.cursor()
            c.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, generate_password_hash(password))
            )
            conn.commit()
            user_id = c.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            return None

    def save_diagnosis(self, symptoms, predicted_disease, confidence=None):
        conn = _get_conn()
        c = conn.cursor()
        symptoms_str = ','.join(symptoms) if isinstance(symptoms, list) else str(symptoms)
        c.execute(
            'INSERT INTO diagnosis_history (user_id, symptoms, predicted_disease, confidence) VALUES (?, ?, ?, ?)',
            (self.id, symptoms_str, predicted_disease, confidence)
        )
        conn.commit()
        diagnosis_id = c.lastrowid
        conn.close()
        return diagnosis_id

    def save_feedback(self, diagnosis_id, rating, feedback_text):
        conn = _get_conn()
        c = conn.cursor()
        c.execute(
            'UPDATE diagnosis_history SET rating = ?, user_feedback = ? WHERE id = ? AND user_id = ?',
            (rating, feedback_text, diagnosis_id, self.id)
        )
        conn.commit()
        conn.close()

    def get_diagnosis_history(self, limit=10):
        conn = _get_conn()
        c = conn.cursor()
        c.execute('''
            SELECT id, symptoms, predicted_disease, confidence, timestamp, rating, user_feedback
            FROM diagnosis_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (self.id, limit))
        history = c.fetchall()
        conn.close()
        return history


def setup_auth(app):
    secret_key = os.environ.get('SECRET_KEY')
    if not secret_key:
        # Generate a random key for development; warn if no env var is set
        secret_key = secrets.token_hex(32)
        import warnings
        warnings.warn(
            'SECRET_KEY env var not set — using a random key. Sessions will be '
            'invalidated on restart. Set SECRET_KEY in production.',
            stacklevel=2
        )
    app.config['SECRET_KEY'] = secret_key

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access the medical recommendation system.'
    login_manager.login_message_category = 'info'

    @login_manager.user_loader
    def load_user(user_id):
        return User.get(int(user_id))

    init_db()
    return login_manager


def add_auth_routes(app):

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')

            if not username or not email or not password:
                flash('All fields are required.', 'danger')
                return render_template(_REGISTER_TEMPLATE)

            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template(_REGISTER_TEMPLATE)

            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'danger')
                return render_template(_REGISTER_TEMPLATE)

            if User.get_by_email(email):
                flash('Email already registered. Please login instead.', 'warning')
                return redirect(url_for('login'))

            user_id = User.create_user(username, email, password)
            if user_id:
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Username already taken. Please choose another.', 'danger')
                return render_template(_REGISTER_TEMPLATE)

        return render_template(_REGISTER_TEMPLATE)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '')
            remember = bool(request.form.get('remember'))

            user_data = User.get_by_email(email)

            if user_data and check_password_hash(user_data[3], password):
                user = User(user_data[0], user_data[1], user_data[2])
                login_user(user, remember=remember)

                conn = _get_conn()
                c = conn.cursor()
                c.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now(), user.id))
                conn.commit()
                conn.close()

                next_page = request.args.get('next')
                flash(f'Welcome back, {user.username}!', 'success')
                return redirect(next_page) if next_page else redirect(url_for('home'))
            else:
                flash('Invalid email or password.', 'danger')

        return render_template('auth/login.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out successfully.', 'info')
        return redirect(url_for('login'))

    @app.route('/profile')
    @login_required
    def profile():
        history = current_user.get_diagnosis_history()
        return render_template('auth/profile.html', user=current_user, history=history)

    @app.route('/dashboard')
    @login_required
    def dashboard():
        recent_diagnoses = current_user.get_diagnosis_history(5)
        return render_template('auth/dashboard.html',
                               user=current_user,
                               recent_diagnoses=recent_diagnoses)
