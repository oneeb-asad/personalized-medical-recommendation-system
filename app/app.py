from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from flask_login import login_required, current_user
import pickle
import pandas as pd
import numpy as np
import ast
import os
import json

# Import authentication system
from auth import setup_auth, add_auth_routes

# Import symptom icons
from symptom_icons import get_all_symptom_icons

# Get the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '..', 'models')
data_dir = os.path.join(base_dir, '..', 'data')
results_dir = os.path.join(base_dir, '..', 'results')

# === Load Trained Model and Encoder ===
rf_model = pickle.load(open(os.path.join(models_dir, 'random_forest.pkl'), 'rb'))
disease_encoder = pickle.load(open(os.path.join(models_dir, 'disease_encoder.pkl'), 'rb'))

# Try to load other models if they exist
try:
    dt_model = pickle.load(open(os.path.join(models_dir, 'decision_tree.pkl'), 'rb'))
    nn_model = pickle.load(open(os.path.join(models_dir, 'neural_network.pkl'), 'rb'))
    all_models_loaded = True
except:
    dt_model = None
    nn_model = None
    all_models_loaded = False

# === Load Supporting CSV Files ===
desc_df = pd.read_csv(os.path.join(data_dir, 'description.csv'))
med_df = pd.read_csv(os.path.join(data_dir, 'medications.csv'))
diet_df = pd.read_csv(os.path.join(data_dir, 'diets.csv'))
prec_df = pd.read_csv(os.path.join(data_dir, 'precautions_df.csv'))
work_df = pd.read_csv(os.path.join(data_dir, 'workout_df.csv'))
training_df = pd.read_csv(os.path.join(data_dir, 'Training.csv'))

# === Extract List of Symptoms from Training Data ===
symptom_list = list(training_df.columns[:-1])

# === Initialize Flask App ===
app = Flask(__name__)

# Setup authentication
login_manager = setup_auth(app)
add_auth_routes(app)

# === Recommendation Function ===
def get_all_recommendations(disease_name):
    result = {}

    try:
        result['description'] = desc_df.loc[desc_df['Disease'] == disease_name, 'Description'].values[0]
    except:
        result['description'] = 'No description available.'

    try:
        meds = med_df.loc[med_df['Disease'] == disease_name, 'Medication'].values[0]
        result['medications'] = ast.literal_eval(meds)
    except:
        result['medications'] = []

    try:
        diet = diet_df.loc[diet_df['Disease'] == disease_name, 'Diet'].values[0]
        result['diet'] = ast.literal_eval(diet)
    except:
        result['diet'] = []

    try:
        prec_row = prec_df.loc[prec_df['Disease'] == disease_name].iloc[:, 2:].values.flatten()
        result['precautions'] = [p for p in prec_row if str(p) != 'nan']
    except:
        result['precautions'] = []

    try:
        workouts = work_df.loc[work_df['disease'] == disease_name, 'workout'].tolist()
        result['workouts'] = workouts
    except:
        result['workouts'] = []

    return result

def get_prediction_confidence(input_vector):
    """Get prediction confidence from the model"""
    try:
        probabilities = rf_model.predict_proba([input_vector])[0]
        confidence = np.max(probabilities) * 100
        return round(confidence, 2)
    except:
        return None

def get_all_model_predictions(input_vector):
    """Get predictions from all available models"""
    predictions = {}
    
    # Random Forest
    try:
        rf_pred = rf_model.predict([input_vector])[0]
        rf_disease = disease_encoder.inverse_transform([rf_pred])[0]
        rf_conf = get_prediction_confidence(input_vector)
        predictions['Random Forest'] = {
            'disease': rf_disease,
            'confidence': rf_conf
        }
    except:
        pass
    
    # Decision Tree
    if dt_model:
        try:
            dt_pred = dt_model.predict([input_vector])[0]
            dt_disease = disease_encoder.inverse_transform([dt_pred])[0]
            dt_proba = dt_model.predict_proba([input_vector])[0]
            dt_conf = round(np.max(dt_proba) * 100, 2)
            predictions['Decision Tree'] = {
                'disease': dt_disease,
                'confidence': dt_conf
            }
        except:
            pass
    
    # Neural Network
    if nn_model:
        try:
            nn_pred = nn_model.predict([input_vector])[0]
            nn_disease = disease_encoder.inverse_transform([nn_pred])[0]
            nn_proba = nn_model.predict_proba([input_vector])[0]
            nn_conf = round(np.max(nn_proba) * 100, 2)
            predictions['Neural Network'] = {
                'disease': nn_disease,
                'confidence': nn_conf
            }
        except:
            pass
    
    return predictions

# === Routes ===
@app.route('/')
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    symptom_icons = get_all_symptom_icons(symptom_list)
    
    return render_template('index.html', 
                         symptoms=symptom_list, 
                         symptom_icons=symptom_icons,
                         user=current_user)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    
    if not selected_symptoms:
        symptom_icons = get_all_symptom_icons(symptom_list)
        return render_template('index.html', 
                             symptoms=symptom_list, 
                             symptom_icons=symptom_icons,
                             user=current_user,
                             error="Please select at least one symptom before predicting.")

    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

    try:
        # Get predictions from all models
        all_predictions = get_all_model_predictions(input_vector)
        
        # Use Random Forest as primary prediction
        primary_prediction = all_predictions.get('Random Forest', {})
        disease_name = primary_prediction.get('disease')
        confidence = primary_prediction.get('confidence')

        # Get recommendations
        recs = get_all_recommendations(disease_name)

        # Save diagnosis to user history
        current_user.save_diagnosis(selected_symptoms, disease_name, confidence)

        # Show result page with all model predictions
        return render_template('result.html', 
                             disease=disease_name, 
                             recs=recs,
                             confidence=confidence,
                             all_predictions=all_predictions,
                             selected_symptoms=selected_symptoms,
                             user=current_user)
    
    except Exception as e:
        symptom_icons = get_all_symptom_icons(symptom_list)
        return render_template('index.html', 
                             symptoms=symptom_list, 
                             symptom_icons=symptom_icons,
                             user=current_user,
                             error=f"An error occurred during prediction: {str(e)}")

@app.route('/evaluation')
@login_required
def evaluation():
    """Display model evaluation results"""
    
    # Check if evaluation results exist
    charts_exist = all([
        os.path.exists(os.path.join(results_dir, 'accuracy_comparison.png')),
        os.path.exists(os.path.join(results_dir, 'metrics_comparison.png')),
        os.path.exists(os.path.join(results_dir, 'confusion_matrices.png')),
        os.path.exists(os.path.join(results_dir, 'feature_importance.png'))
    ])
    
    # Load evaluation metrics if available
    evaluation_data = {
        'charts_available': charts_exist,
        'models_loaded': all_models_loaded,
        'models': ['Random Forest', 'Decision Tree', 'Neural Network'] if all_models_loaded else ['Random Forest'],
        'dataset_size': len(training_df),
        'num_symptoms': len(symptom_list),
        'num_diseases': training_df['prognosis'].nunique()
    }
    
    return render_template('evaluation.html', 
                         evaluation=evaluation_data,
                         user=current_user)

# @app.route('/feedback', methods=['POST'])
# @login_required
# def submit_feedback():
#     """Handle user feedback on predictions"""
#     try:
#         diagnosis_id = request.form.get('diagnosis_id')
#         rating = request.form.get('rating')
#         feedback_text = request.form.get('feedback')
        
#         flash('Thank you for your feedback! It helps improve our system.', 'success')
        
#     except Exception as e:
#         flash('Error submitting feedback. Please try again.', 'danger')
    
#     return redirect(url_for('profile'))

@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve files from results directory"""
    return send_from_directory(results_dir, filename)

# === Demo routes ===
@app.route('/demo')
def demo():
    """Demo access without authentication"""
    symptom_icons = get_all_symptom_icons(symptom_list)
    return render_template('index.html', 
                         symptoms=symptom_list, 
                         symptom_icons=symptom_icons,
                         demo_mode=True)

@app.route('/demo-predict', methods=['POST'])
def demo_predict():
    """Demo prediction without authentication"""
    selected_symptoms = request.form.getlist('symptoms')
    
    if not selected_symptoms:
        symptom_icons = get_all_symptom_icons(symptom_list)
        return render_template('index.html', 
                             symptoms=symptom_list, 
                             symptom_icons=symptom_icons,
                             demo_mode=True,
                             error="Please select at least one symptom before predicting.")

    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

    try:
        prediction = rf_model.predict([input_vector])[0]
        disease_name = disease_encoder.inverse_transform([prediction])[0]
        confidence = get_prediction_confidence(input_vector)
        recs = get_all_recommendations(disease_name)

        return render_template('result.html', 
                             disease=disease_name, 
                             recs=recs,
                             confidence=confidence,
                             selected_symptoms=selected_symptoms,
                             demo_mode=True)
    
    except Exception as e:
        symptom_icons = get_all_symptom_icons(symptom_list)
        return render_template('index.html', 
                             symptoms=symptom_list, 
                             symptom_icons=symptom_icons,
                             demo_mode=True,
                             error=f"An error occurred during prediction: {str(e)}")

# === API Endpoints ===
@app.route('/api/symptoms')
def api_symptoms():
    """API endpoint to get all symptoms with their icons"""
    symptom_icons = get_all_symptom_icons(symptom_list)
    return {
        'symptoms': symptom_list,
        'icons': symptom_icons,
        'total_symptoms': len(symptom_list)
    }

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': rf_model is not None,
        'total_symptoms': len(symptom_list),
        'all_models_loaded': all_models_loaded
    }


# === Start Server ===
if __name__ == '__main__':
    auth_template_dir = os.path.join('templates', 'auth')
    if not os.path.exists(auth_template_dir):
        os.makedirs(auth_template_dir)
    
    app.run(debug=True)