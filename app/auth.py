# auth.py - User Authentication Module

from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import sqlite3
import os

# Database setup
def init_db():
    """Initialize the user database"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Create users table
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
    
    # Create diagnosis history table
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
    """User class for Flask-Login"""
    
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email
    
    @staticmethod
    def get(user_id):
        """Get user by ID"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,))
        user_data = c.fetchone()
        conn.close()
        
        if user_data:
            return User(user_data[0], user_data[1], user_data[2])
        return None
    
    @staticmethod
    def get_by_email(email):
        """Get user by email"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('SELECT id, username, email, password_hash FROM users WHERE email = ?', (email,))
        user_data = c.fetchone()
        conn.close()
        
        return user_data
    
    @staticmethod
    def create_user(username, email, password):
        """Create new user"""
        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            password_hash = generate_password_hash(password)
            c.execute('''
                INSERT INTO users (username, email, password_hash) 
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            conn.commit()
            user_id = c.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            return None  # User already exists
    
    def save_diagnosis(self, symptoms, predicted_disease, confidence=None):
        """Save diagnosis to user history"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        symptoms_str = ','.join(symptoms) if isinstance(symptoms, list) else str(symptoms)
        
        c.execute('''
            INSERT INTO diagnosis_history (user_id, symptoms, predicted_disease, confidence) 
            VALUES (?, ?, ?, ?)
        ''', (self.id, symptoms_str, predicted_disease, confidence))
        
        conn.commit()
        diagnosis_id = c.lastrowid
        conn.close()
        return diagnosis_id
    
    def get_diagnosis_history(self, limit=10):
        """Get user's diagnosis history"""
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute('''
            SELECT symptoms, predicted_disease, confidence, timestamp, rating, user_feedback 
            FROM diagnosis_history 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (self.id, limit))
        
        history = c.fetchall()
        conn.close()
        return history

def setup_auth(app):
    """Setup authentication for Flask app"""
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
    
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Please log in to access the medical recommendation system.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.get(int(user_id))
    
    # Initialize database
    init_db()
    
    return login_manager

# Authentication routes
def add_auth_routes(app):
    """Add authentication routes to Flask app"""
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            
            # Validation
            if not username or not email or not password:
                flash('All fields are required.', 'danger')
                return render_template('auth/register.html')
            
            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template('auth/register.html')
            
            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'danger')
                return render_template('auth/register.html')
            
            # Check if user exists
            existing_user = User.get_by_email(email)
            if existing_user:
                flash('Email already registered. Please login instead.', 'warning')
                return redirect(url_for('login'))
            
            # Create user
            user_id = User.create_user(username, email, password)
            if user_id:
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
            else:
                flash('Registration failed. Please try again.', 'danger')
        
        return render_template('auth/register.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            remember = bool(request.form.get('remember'))
            
            user_data = User.get_by_email(email)
            
            if user_data and check_password_hash(user_data[3], password):
                user = User(user_data[0], user_data[1], user_data[2])
                login_user(user, remember=remember)
                
                # Update last login
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                         (datetime.now(), user.id))
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