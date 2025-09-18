from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import ast
import os

# # === Load Trained Model and Encoder ===
# rf_model = pickle.load(open('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/models/rf_model.pkl', 'rb'))
# disease_encoder = pickle.load(open('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/models/disease_encoder.pkl', 'rb'))

# # === Load Supporting CSV Files ===
# desc_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/description.csv')
# med_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/medications.csv')
# diet_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/diets.csv')
# prec_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/precautions_df.csv')
# work_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/workout_df.csv')
# training_df = pd.read_csv('C:/Users/Dell/Desktop/oneeb dissertation/Dataset/raw data/Training.csv')

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

# === Routes ===
@app.route('/')
def home():
    return render_template('index.html', symptoms=symptom_list)

@app.route('/predict', methods=['POST'])
def predict():
    # Get selected symptoms from form
    selected_symptoms = request.form.getlist('symptoms')

    # Convert to binary input vector
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]

    # Predict using model
    prediction = rf_model.predict([input_vector])[0]
    disease_name = disease_encoder.inverse_transform([prediction])[0]

    # Get recommendations
    recs = get_all_recommendations(disease_name)

    # Show result page
    return render_template('result.html', disease=disease_name, recs=recs)

# === Start Server ===
if __name__ == '__main__':
    app.run(debug=True)
