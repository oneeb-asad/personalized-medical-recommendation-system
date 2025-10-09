# from flask import Flask, request, render_template
# import pickle
# import pandas as pd
# import numpy as np
# import ast
# import os

# # # === Load Trained Model and Encoder ===
# # rf_model = pickle.load(open('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/models/rf_model.pkl', 'rb'))
# # disease_encoder = pickle.load(open('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/models/disease_encoder.pkl', 'rb'))

# # # === Load Supporting CSV Files ===
# # desc_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/description.csv')
# # med_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/medications.csv')
# # diet_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/diets.csv')
# # prec_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/precautions_df.csv')
# # work_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/workout_df.csv')
# # training_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/Training.csv')

# # Get the directory containing this script
# base_dir = os.path.dirname(os.path.abspath(__file__))
# models_dir = os.path.join(base_dir, '..', 'models')
# data_dir = os.path.join(base_dir, '..', 'data')

# # === Load Trained Model and Encoder ===
# rf_model = pickle.load(open(os.path.join(models_dir, 'rf_model.pkl'), 'rb'))
# disease_encoder = pickle.load(open(os.path.join(models_dir, 'disease_encoder.pkl'), 'rb'))

# # === Load Supporting CSV Files ===
# desc_df = pd.read_csv(os.path.join(data_dir, 'description.csv'))
# med_df = pd.read_csv(os.path.join(data_dir, 'medications.csv'))
# diet_df = pd.read_csv(os.path.join(data_dir, 'diets.csv'))
# prec_df = pd.read_csv(os.path.join(data_dir, 'precautions_df.csv'))
# work_df = pd.read_csv(os.path.join(data_dir, 'workout_df.csv'))
# training_df = pd.read_csv(os.path.join(data_dir, 'Training.csv'))

# # === Extract List of Symptoms from Training Data ===
# symptom_list = list(training_df.columns[:-1])

# # === Initialize Flask App ===
# app = Flask(__name__)

# # === Recommendation Function ===
# def get_all_recommendations(disease_name):
#     result = {}

#     try:
#         result['description'] = desc_df.loc[desc_df['Disease'] == disease_name, 'Description'].values[0]
#     except:
#         result['description'] = 'No description available.'

#     try:
#         meds = med_df.loc[med_df['Disease'] == disease_name, 'Medication'].values[0]
#         result['medications'] = ast.literal_eval(meds)
#     except:
#         result['medications'] = []

#     try:
#         diet = diet_df.loc[diet_df['Disease'] == disease_name, 'Diet'].values[0]
#         result['diet'] = ast.literal_eval(diet)
#     except:
#         result['diet'] = []

#     try:
#         prec_row = prec_df.loc[prec_df['Disease'] == disease_name].iloc[:, 2:].values.flatten()
#         result['precautions'] = [p for p in prec_row if str(p) != 'nan']
#     except:
#         result['precautions'] = []

#     try:
#         workouts = work_df.loc[work_df['disease'] == disease_name, 'workout'].tolist()
#         result['workouts'] = workouts
#     except:
#         result['workouts'] = []

#     return result

# # === Routes ===
# @app.route('/')
# def home():
#     return render_template('index.html', symptoms=symptom_list)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get selected symptoms from form
#     selected_symptoms = request.form.getlist('symptoms')

#     # Convert to binary input vector
#     input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

#     # Predict using model
#     prediction = rf_model.predict([input_vector])[0]
#     disease_name = disease_encoder.inverse_transform([prediction])[0]

#     # Get recommendations
#     recs = get_all_recommendations(disease_name)

#     # Show result page
#     return render_template('result.html', disease=disease_name, recs=recs)

# # === Start Server ===
# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, render_template
# import pickle
# import pandas as pd
# import numpy as np
# import ast
# import os
# from symptoms_icons import get_all_symptom_icons  # Import our new icon function

# # Get the directory containing this script
# base_dir = os.path.dirname(os.path.abspath(__file__))
# models_dir = os.path.join(base_dir, '..', 'models')
# data_dir = os.path.join(base_dir, '..', 'data')

# # === Load Trained Model and Encoder ===
# rf_model = pickle.load(open(os.path.join(models_dir, 'rf_model.pkl'), 'rb'))
# disease_encoder = pickle.load(open(os.path.join(models_dir, 'disease_encoder.pkl'), 'rb'))

# # === Load Supporting CSV Files ===
# desc_df = pd.read_csv(os.path.join(data_dir, 'description.csv'))
# med_df = pd.read_csv(os.path.join(data_dir, 'medications.csv'))
# diet_df = pd.read_csv(os.path.join(data_dir, 'diets.csv'))
# prec_df = pd.read_csv(os.path.join(data_dir, 'precautions_df.csv'))
# work_df = pd.read_csv(os.path.join(data_dir, 'workout_df.csv'))
# training_df = pd.read_csv(os.path.join(data_dir, 'Training.csv'))

# # === Extract List of Symptoms from Training Data ===
# symptom_list = list(training_df.columns[:-1])

# # === Initialize Flask App ===
# app = Flask(__name__)

# # === Recommendation Function ===
# def get_all_recommendations(disease_name):
#     result = {}

#     try:
#         result['description'] = desc_df.loc[desc_df['Disease'] == disease_name, 'Description'].values[0]
#     except:
#         result['description'] = 'No description available.'

#     try:
#         meds = med_df.loc[med_df['Disease'] == disease_name, 'Medication'].values[0]
#         result['medications'] = ast.literal_eval(meds)
#     except:
#         result['medications'] = []

#     try:
#         diet = diet_df.loc[diet_df['Disease'] == disease_name, 'Diet'].values[0]
#         result['diet'] = ast.literal_eval(diet)
#     except:
#         result['diet'] = []

#     try:
#         prec_row = prec_df.loc[prec_df['Disease'] == disease_name].iloc[:, 2:].values.flatten()
#         result['precautions'] = [p for p in prec_row if str(p) != 'nan']
#     except:
#         result['precautions'] = []

#     try:
#         workouts = work_df.loc[work_df['disease'] == disease_name, 'workout'].tolist()
#         result['workouts'] = workouts
#     except:
#         result['workouts'] = []

#     return result

# # === Routes ===
# @app.route('/')
# def home():
#     # Generate symptom-to-icon mapping
#     symptom_icons = get_all_symptom_icons(symptom_list)
    
#     # Pass both symptoms and their corresponding icons to the template
#     return render_template('index.html', 
#                          symptoms=symptom_list, 
#                          symptom_icons=symptom_icons)

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Validate input
#     selected_symptoms = request.form.getlist('symptoms')
    
#     # Check if any symptoms were selected
#     if not selected_symptoms:
#         # Generate symptom icons again for error display
#         symptom_icons = get_all_symptom_icons(symptom_list)
#         return render_template('index.html', 
#                              symptoms=symptom_list, 
#                              symptom_icons=symptom_icons,
#                              error="Please select at least one symptom before predicting.")

#     # Convert to binary input vector
#     input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

#     # Predict using model
#     try:
#         prediction = rf_model.predict([input_vector])[0]
#         disease_name = disease_encoder.inverse_transform([prediction])[0]

#         # Get recommendations
#         recs = get_all_recommendations(disease_name)

#         # Show result page
#         return render_template('result.html', 
#                              disease=disease_name, 
#                              recs=recs,
#                              selected_symptoms=selected_symptoms)
    
#     except Exception as e:
#         # Handle prediction errors
#         symptom_icons = get_all_symptom_icons(symptom_list)
#         return render_template('index.html', 
#                              symptoms=symptom_list, 
#                              symptom_icons=symptom_icons,
#                              error=f"An error occurred during prediction: {str(e)}")

# # === Optional: Add an API endpoint for testing ===
# @app.route('/api/symptoms')
# def api_symptoms():
#     """API endpoint to get all symptoms with their icons"""
#     symptom_icons = get_all_symptom_icons(symptom_list)
#     return {
#         'symptoms': symptom_list,
#         'icons': symptom_icons,
#         'total_symptoms': len(symptom_list)
#     }

# # === Start Server ===
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
import pickle
import pandas as pd
import numpy as np
import ast
import os

# Import our authentication system
from auth import setup_auth, add_auth_routes

# Import symptom icons
from symptoms_icons import get_all_symptom_icons

# Get the directory containing this script
base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, '..', 'models')
data_dir = os.path.join(base_dir, '..', 'data')

# === Load Trained Model and Encoder ===
rf_model = pickle.load(open(os.path.join(models_dir, 'rf_model.pkl'), 'rb'))
disease_encoder = pickle.load(open(os.path.join(models_dir, 'disease_encoder.pkl'), 'rb'))

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
        # Get prediction probabilities
        probabilities = rf_model.predict_proba([input_vector])[0]
        # Get the maximum probability as confidence
        confidence = np.max(probabilities) * 100
        return round(confidence, 2)
    except:
        return None

# === Routes ===
@app.route('/')
def home():
    # Redirect to login if not authenticated
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    # Generate symptom-to-icon mapping
    symptom_icons = get_all_symptom_icons(symptom_list)
    
    # Pass both symptoms and their corresponding icons to the template
    return render_template('index.html', 
                         symptoms=symptom_list, 
                         symptom_icons=symptom_icons,
                         user=current_user)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    # Validate input
    selected_symptoms = request.form.getlist('symptoms')
    
    # Check if any symptoms were selected
    if not selected_symptoms:
        symptom_icons = get_all_symptom_icons(symptom_list)
        return render_template('index.html', 
                             symptoms=symptom_list, 
                             symptom_icons=symptom_icons,
                             user=current_user,
                             error="Please select at least one symptom before predicting.")

    # Convert to binary input vector
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

    try:
        # Predict using model
        prediction = rf_model.predict([input_vector])[0]
        disease_name = disease_encoder.inverse_transform([prediction])[0]
        
        # Get confidence score
        confidence = get_prediction_confidence(input_vector)

        # Get recommendations
        recs = get_all_recommendations(disease_name)

        # Save diagnosis to user history
        current_user.save_diagnosis(selected_symptoms, disease_name, confidence)

        # Show result page
        return render_template('result.html', 
                             disease=disease_name, 
                             recs=recs,
                             confidence=confidence,
                             selected_symptoms=selected_symptoms,
                             user=current_user)
    
    except Exception as e:
        # Handle prediction errors
        symptom_icons = get_all_symptom_icons(symptom_list)
        return render_template('index.html', 
                             symptoms=symptom_list, 
                             symptom_icons=symptom_icons,
                             user=current_user,
                             error=f"An error occurred during prediction: {str(e)}")

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Handle user feedback on predictions"""
    try:
        diagnosis_id = request.form.get('diagnosis_id')
        rating = request.form.get('rating')
        feedback_text = request.form.get('feedback')
        
        # In a real system, you'd update the diagnosis record with feedback
        # For now, we'll just flash a success message
        flash('Thank you for your feedback! It helps improve our system.', 'success')
        
    except Exception as e:
        flash('Error submitting feedback. Please try again.', 'danger')
    
    return redirect(url_for('profile'))

# === Demo route for testing without full registration ===
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
        'total_symptoms': len(symptom_list)
    }

# === Error Handlers ===
@app.errorhandler(404)
def not_found_error(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('errors/500.html'), 500

# === Start Server ===
if __name__ == '__main__':
    # Create templates/auth directory if it doesn't exist
    auth_template_dir = os.path.join('templates', 'auth')
    if not os.path.exists(auth_template_dir):
        os.makedirs(auth_template_dir)
    
    app.run(debug=True)