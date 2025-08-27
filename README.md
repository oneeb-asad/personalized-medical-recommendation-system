# 🧠 Personalized Medical Recommendation System with Machine Learning

This is a Flask-based web application that predicts diseases based on user-selected symptoms and provides personalized medical recommendations, including:

- 📋 Disease Description
- 💊 Medications
- 🥗 Diet Plans
- ✅ Precautions
- 🏃‍♂️ Workouts

---

## 🛠️ Technologies Used

- Python 3.x
- Pandas, NumPy, Scikit-learn
- Flask (Web Framework)
- Bootstrap 5 (Frontend)
- Jupyter Notebooks (for development & training)
- HTML5/CSS3

---

## 📂 Project Structure

```
MedicalRecommendationSystem/
├── app/
│   ├── app.py                  # Flask backend
│   ├── templates/
│   │   ├── index.html          # Symptom input form
│   │   ├── result.html         # Diagnosis & recommendations
├── models/
│   ├── rf_model.pkl            # Trained ML model
│   ├── disease_encoder.pkl     # Label encoder for disease names
├── data/
│   ├── Training.csv            # Training data
│   ├── description.csv
│   ├── medications.csv
│   ├── diets.csv
│   ├── precautions_df.csv
│   ├── workout_df.csv
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_recommendation_engine.ipynb
│   ├── 3_flask_app.ipynb
├── README.md
```

---

## 🚀 How to Run the Project

### 1. ✅ Download the Project

Download the project folder and unzip it to your system.

### 2. ✅ Install Required Libraries

Make sure you have Python 3.8+ installed. Then install dependencies:

pip install flask pandas scikit-learn numpy

pip install scikit-learn

Optional (for training notebooks):

pip install jupyter matplotlib seaborn

### 3. ✅ Train Model (Optional)

If you want to retrain the model:

cd notebooks
jupyter notebook

# Run `1_data_preprocessing.ipynb` and `2_recommendation_engine.ipynb`

### 4. ✅ Run the Web App

Navigate to the `app/` folder and run the Flask app:

cd ../app
python app.py

Visit your browser:

http://localhost:5000

## 💡 Features

- Intelligent prediction of disease from user-selected symptoms
- Fetches relevant medications, diet, precautions, and workouts
- Beautiful and responsive Bootstrap frontend
- Easy to extend, train, and deploy

## 📧 Contact

**Project by:** Oneeb Asad
**Course:** CO7100 – MSc Dissertation (2024–25)  
**Supervisor:** Paul Underhill
