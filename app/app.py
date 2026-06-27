import ast
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, send_from_directory, url_for
from flask_login import current_user, login_required
from flask_wtf.csrf import CSRFProtect

from auth import add_auth_routes, setup_auth
from symptom_icons import get_all_symptom_icons

# ── String constants ───────────────────────────────────────────────────────────
RF_NAME        = 'Random Forest'
DT_NAME        = 'Decision Tree'
NN_NAME        = 'Neural Network'
INDEX_TEMPLATE = 'index.html'

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE    = os.path.dirname(os.path.abspath(__file__))
_MODELS  = os.path.join(_BASE, '..', 'models')
_DATA    = os.path.join(_BASE, '..', 'data')
_RESULTS = os.path.join(_BASE, '..', 'results')

# ── Load models ────────────────────────────────────────────────────────────────
def _load_pkl(name):
    with open(os.path.join(_MODELS, name), 'rb') as f:
        return pickle.load(f)

rf_model        = _load_pkl('random_forest.pkl')
disease_encoder = _load_pkl('disease_encoder.pkl')

try:
    dt_model = _load_pkl('decision_tree.pkl')
    nn_model = _load_pkl('neural_network.pkl')
    all_models_loaded = True
except Exception:
    dt_model = None
    nn_model = None
    all_models_loaded = False

# ── Load & index supporting data ───────────────────────────────────────────────
training_df  = pd.read_csv(os.path.join(_DATA, 'Training.csv'))
symptom_list = list(training_df.columns[:-1])

desc_df = pd.read_csv(os.path.join(_DATA, 'description.csv')).set_index('Disease')
med_df  = pd.read_csv(os.path.join(_DATA, 'medications.csv')).set_index('Disease')
diet_df = pd.read_csv(os.path.join(_DATA, 'diets.csv')).set_index('Disease')
prec_df = pd.read_csv(os.path.join(_DATA, 'precautions_df.csv'))
work_df = pd.read_csv(os.path.join(_DATA, 'workout_df.csv'))

# Severity weights: symptom → int weight (1-7); default 1 if missing
_sev_df  = pd.read_csv(os.path.join(_DATA, 'Symptom-severity.csv'))
SEVERITY = dict(zip(_sev_df['Symptom'], _sev_df['weight']))

# Random Forest feature importances (for per-prediction explanation)
RF_IMPORTANCE = dict(zip(symptom_list, rf_model.feature_importances_))

# Valid symptom set for input validation
VALID_SYMPTOMS = set(symptom_list)

# ── Flask app ──────────────────────────────────────────────────────────────────
app          = Flask(__name__)
csrf         = CSRFProtect()
login_manager = setup_auth(app)
csrf.init_app(app)
add_auth_routes(app)

# ── Helpers ────────────────────────────────────────────────────────────────────
def _build_input_vector(selected):
    return [1 if s in selected else 0 for s in symptom_list]


def _severity_score(selected):
    return sum(SEVERITY.get(s, 1) for s in selected)


def _top_symptoms(selected, top_n=5):
    ranked = sorted(
        [(s, RF_IMPORTANCE.get(s, 0)) for s in selected],
        key=lambda x: x[1], reverse=True
    )
    return [{'symptom': s.replace('_', ' ').title(), 'importance': round(imp * 100, 1)}
            for s, imp in ranked[:top_n]]


def get_all_recommendations(disease_name):
    result = {}

    try:
        result['description'] = desc_df.loc[disease_name, 'Description']
    except Exception:
        result['description'] = 'No description available.'

    try:
        result['medications'] = ast.literal_eval(med_df.loc[disease_name, 'Medication'])
    except Exception:
        result['medications'] = []

    try:
        result['diet'] = ast.literal_eval(diet_df.loc[disease_name, 'Diet'])
    except Exception:
        result['diet'] = []

    try:
        prec_row = prec_df.loc[prec_df['Disease'] == disease_name].iloc[:, 2:].values.flatten()
        result['precautions'] = [p for p in prec_row if str(p) != 'nan']
    except Exception:
        result['precautions'] = []

    try:
        result['workouts'] = work_df.loc[work_df['disease'] == disease_name, 'workout'].tolist()
    except Exception:
        result['workouts'] = []

    return result


def _confidence(model, vec):
    try:
        return round(float(np.max(model.predict_proba([vec])[0])) * 100, 2)
    except Exception:
        return None


def get_all_model_predictions(input_vector):
    predictions = {}

    try:
        pred = rf_model.predict([input_vector])[0]
        predictions[RF_NAME] = {
            'disease':    disease_encoder.inverse_transform([pred])[0],
            'confidence': _confidence(rf_model, input_vector),
        }
    except Exception:
        pass

    if dt_model:
        try:
            pred = dt_model.predict([input_vector])[0]
            predictions[DT_NAME] = {
                'disease':    disease_encoder.inverse_transform([pred])[0],
                'confidence': _confidence(dt_model, input_vector),
            }
        except Exception:
            pass

    if nn_model:
        try:
            pred = nn_model.predict([input_vector])[0]
            predictions[NN_NAME] = {
                'disease':    disease_encoder.inverse_transform([pred])[0],
                'confidence': _confidence(nn_model, input_vector),
            }
        except Exception:
            pass

    return predictions


def _ensemble_vote(all_predictions):
    votes = {}
    for info in all_predictions.values():
        d = info['disease']
        votes[d] = votes.get(d, 0) + 1

    max_votes  = max(votes.values())
    candidates = [d for d, v in votes.items() if v == max_votes]

    if len(candidates) == 1:
        return candidates[0]

    rf_pick = all_predictions.get(RF_NAME, {}).get('disease')
    return rf_pick if rf_pick in candidates else candidates[0]


def _index_ctx(user=None, demo_mode=False, error=None):
    ctx = {
        'symptoms':      symptom_list,
        'symptom_icons': get_all_symptom_icons(symptom_list),
        'demo_mode':     demo_mode,
    }
    if user is not None:
        ctx['user'] = user
    if error:
        ctx['error'] = error
    return ctx


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    return render_template(INDEX_TEMPLATE, **_index_ctx(user=current_user))


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    selected = [s for s in request.form.getlist('symptoms') if s in VALID_SYMPTOMS]

    if not selected:
        return render_template(INDEX_TEMPLATE,
                               **_index_ctx(user=current_user,
                                            error="Please select at least one valid symptom."))

    vec             = _build_input_vector(selected)
    all_predictions = get_all_model_predictions(vec)

    if not all_predictions:
        return render_template(INDEX_TEMPLATE,
                               **_index_ctx(user=current_user,
                                            error="Prediction failed. Please try again."))

    disease_name = _ensemble_vote(all_predictions)
    confidence   = all_predictions.get(RF_NAME, {}).get('confidence')
    diagnosis_id = current_user.save_diagnosis(selected, disease_name, confidence)

    return render_template('result.html',
                           disease=disease_name,
                           recs=get_all_recommendations(disease_name),
                           confidence=confidence,
                           all_predictions=all_predictions,
                           selected_symptoms=selected,
                           top_symptoms=_top_symptoms(selected),
                           severity_score=_severity_score(selected),
                           diagnosis_id=diagnosis_id,
                           user=current_user)


@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    try:
        diagnosis_id  = int(request.form.get('diagnosis_id', 0))
        rating        = int(request.form.get('rating', 0))
        feedback_text = request.form.get('feedback', '').strip()

        if diagnosis_id and 1 <= rating <= 5:
            current_user.save_feedback(diagnosis_id, rating, feedback_text)
            flash('Thank you for your feedback!', 'success')
        else:
            flash('Please provide a rating between 1 and 5.', 'warning')
    except Exception:
        flash('Error submitting feedback. Please try again.', 'danger')

    return redirect(url_for('profile'))


@app.route('/evaluation', methods=['GET'])
@login_required
def evaluation():
    charts_exist = all(
        os.path.exists(os.path.join(_RESULTS, f))
        for f in ('accuracy_comparison.png', 'metrics_comparison.png',
                  'confusion_matrices.png', 'feature_importance.png')
    )
    return render_template('evaluation.html',
                           evaluation={
                               'charts_available': charts_exist,
                               'models_loaded':    all_models_loaded,
                               'models':           [RF_NAME, DT_NAME, NN_NAME]
                                                   if all_models_loaded else [RF_NAME],
                               'dataset_size':     len(training_df),
                               'num_symptoms':     len(symptom_list),
                               'num_diseases':     training_df['prognosis'].nunique(),
                           },
                           user=current_user)


@app.route('/results/<path:filename>', methods=['GET'])
def serve_results(filename):
    return send_from_directory(_RESULTS, filename)


# ── Demo routes (unauthenticated) ──────────────────────────────────────────────
@app.route('/demo', methods=['GET'])
def demo():
    return render_template(INDEX_TEMPLATE, **_index_ctx(demo_mode=True))


@app.route('/demo-predict', methods=['POST'])
def demo_predict():
    selected = [s for s in request.form.getlist('symptoms') if s in VALID_SYMPTOMS]

    if not selected:
        return render_template(INDEX_TEMPLATE,
                               **_index_ctx(demo_mode=True,
                                            error="Please select at least one valid symptom."))

    vec             = _build_input_vector(selected)
    all_predictions = get_all_model_predictions(vec)

    if not all_predictions:
        return render_template(INDEX_TEMPLATE,
                               **_index_ctx(demo_mode=True,
                                            error="Prediction failed. Please try again."))

    disease_name = _ensemble_vote(all_predictions)
    confidence   = all_predictions.get(RF_NAME, {}).get('confidence')

    return render_template('result.html',
                           disease=disease_name,
                           recs=get_all_recommendations(disease_name),
                           confidence=confidence,
                           all_predictions=all_predictions,
                           selected_symptoms=selected,
                           top_symptoms=_top_symptoms(selected),
                           severity_score=_severity_score(selected),
                           demo_mode=True)


# ── API ────────────────────────────────────────────────────────────────────────
@app.route('/api/symptoms', methods=['GET'])
def api_symptoms():
    return {
        'symptoms':       symptom_list,
        'icons':          get_all_symptom_icons(symptom_list),
        'total_symptoms': len(symptom_list),
    }


@app.route('/api/health', methods=['GET'])
def api_health():
    return {
        'status':            'healthy',
        'model_loaded':      rf_model is not None,
        'total_symptoms':    len(symptom_list),
        'all_models_loaded': all_models_loaded,
    }


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(os.path.join('templates', 'auth'), exist_ok=True)
    app.run(debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true')
