# 🧠 Personalized Medical Recommendation System

A web-based medical recommendation system that predicts diseases from user symptoms and provides personalized health advice using machine learning.

---

## 📋 Overview

This Flask application uses three machine learning algorithms (Random Forest, Decision Tree, Neural Network) to predict diseases from 132 symptoms. It provides comprehensive recommendations including:

- Disease description and information
- Medication suggestions
- Diet recommendations
- Safety precautions
- Exercise plans

**Performance:** All models achieved 100% accuracy on test data (41 diseases, 4,920 training records)

---

## 🛠️ Technologies

- **Backend:** Python 3.8+, Flask, scikit-learn
- **Frontend:** HTML5, CSS3, Bootstrap 5
- **Database:** SQLite
- **ML Models:** Random Forest, Decision Tree, Neural Network

---

## 📂 Project Structure

```
MedicalRecommendationSystem/
├── app/
│   ├── app.py                  # Main Flask application
│   ├── templates/              # HTML files
│   └── static/                 # CSS files
├── data/                       # CSV files (training data, recommendations)
├── models/                     # Trained ML models (.pkl files)
├── results/                    # Evaluation charts
└── requirements.txt            # Python dependencies
```

---

## 🚀 Installation

### 1. Clone or Download

```bash
git clone https://git.chester.network/2420441/personalized-medical-recommendation-system-with-machine-learning.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:** Flask, scikit-learn, pandas, numpy, flask-login

### 3. Run Application

```bash
cd app
python app.py
```

### 4. Open in Browser

```
http://127.0.0.1:5000
```

---

## 💡 How to Use

1. **Register/Login** - Create account or login
2. **Select Symptoms** - Choose from 132 symptoms (use search to find quickly)
3. **Get Prediction** - Click "Analyze Symptoms"
4. **View Results** - See predicted disease and recommendations
5. **Check History** - View past predictions in your profile

---

## 📊 Model Performance

| Model          | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| Random Forest  | 100%     | 100%      | 100%   | 100%     |
| Decision Tree  | 100%     | 100%      | 100%   | 100%     |
| Neural Network | 100%     | 100%      | 100%   | 100%     |

**Primary Model:** Random Forest (selected for interpretability)

---

## ⚠️ Disclaimer

**This system is for educational purposes only. Not intended for medical diagnosis. Always consult healthcare professionals for medical advice.**

---

## 👨‍💻 Author

**Oneeb Asad**

- **Course:** MSc Data Science (2024-25)
- **University:** University of Chester
- **Supervisor:** Paul Underhill

## 📄 License

This project is for academic purposes.

---

**Last Updated:** 16th October 2025
