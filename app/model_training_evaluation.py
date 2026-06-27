"""
Medical Recommendation System - ML Model Training & Evaluation
Trains and evaluates three models: Random Forest, Decision Tree, Neural Network.
Uses 5-fold stratified cross-validation for robust performance estimates.
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

RF_NAME = 'Random Forest'
DT_NAME = 'Decision Tree'
NN_NAME = 'Neural Network'


class MedicalModelEvaluator:
    """Train and evaluate multiple ML models for disease prediction."""

    def __init__(self, data_path, severity_path=None):
        self.data_path = data_path
        self.severity_path = severity_path
        self.models = {}
        self.results = {}
        self.cv_results = {}
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.disease_encoder = None
        self.severity_weights = {}

    def load_and_preprocess_data(self):
        """Load dataset, optionally apply symptom severity weights, and split."""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Number of diseases: {df['prognosis'].nunique()}")

        # Load severity weights if provided
        if self.severity_path and os.path.exists(self.severity_path):
            sev_df = pd.read_csv(self.severity_path)
            self.severity_weights = dict(zip(sev_df['Symptom'], sev_df['weight']))
            print(f"Loaded severity weights for {len(self.severity_weights)} symptoms.")

        X = df.drop('prognosis', axis=1)
        y = df['prognosis']

        # Apply severity weighting: multiply binary feature columns by their weight
        if self.severity_weights:
            for col in X.columns:
                weight = self.severity_weights.get(col, 1)
                X[col] = X[col] * weight
            print("Applied symptom severity weights to features.")

        self.disease_encoder = LabelEncoder()
        y_encoded = self.disease_encoder.fit_transform(y)

        # 70 / 15 / 15 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        n = len(df)
        print("\nData split:")
        print(f"  Training:   {len(X_train)} samples ({len(X_train)/n*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/n*100:.1f}%)")
        print(f"  Test:       {len(X_test)} samples ({len(X_test)/n*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------
    # Cross-validation (call before or after train_* methods)
    # ------------------------------------------------------------------

    def run_cross_validation(self, n_splits=5):
        """5-fold stratified cross-validation on all three model architectures.

        NOTE on 100 % accuracy: this dataset is synthetic and highly structured,
        so perfect scores are expected. Real-world deployments would require a
        more diverse, noisy dataset to obtain meaningful generalisation estimates.
        """
        print("\n" + "="*60)
        print(f"Running {n_splits}-Fold Stratified Cross-Validation")
        print("="*60)

        import numpy as np
        from sklearn.model_selection import StratifiedKFold, cross_validate

        # Combine train+val for CV (keep test set untouched)
        X_cv = pd.concat([self.X_train, self.X_val])
        y_cv = np.concatenate([self.y_train, self.y_val])

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        model_defs = {
            RF_NAME: RandomForestClassifier(
                n_estimators=100, min_samples_leaf=1, max_features='sqrt',
                random_state=42, n_jobs=-1),
            DT_NAME: DecisionTreeClassifier(ccp_alpha=0.0, random_state=42),
            NN_NAME: MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                max_iter=500, random_state=42, early_stopping=True),
        }

        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

        for name, model in model_defs.items():
            scores = cross_validate(model, X_cv, y_cv, cv=cv, scoring=scoring, n_jobs=-1)
            self.cv_results[name] = {
                'accuracy_mean':   scores['test_accuracy'].mean(),
                'accuracy_std':    scores['test_accuracy'].std(),
                'precision_mean':  scores['test_precision_weighted'].mean(),
                'recall_mean':     scores['test_recall_weighted'].mean(),
                'f1_mean':         scores['test_f1_weighted'].mean(),
                'fold_accuracies': scores['test_accuracy'].tolist(),
            }
            print(f"\n{name}:")
            print(f"  Accuracy:  {self.cv_results[name]['accuracy_mean']:.4f} "
                  f"± {self.cv_results[name]['accuracy_std']:.4f}")
            print(f"  Precision: {self.cv_results[name]['precision_mean']:.4f}")
            print(f"  Recall:    {self.cv_results[name]['recall_mean']:.4f}")
            print(f"  F1-Score:  {self.cv_results[name]['f1_mean']:.4f}")

        print("\n[Note] Perfect scores reflect the synthetic, structured nature of this")
        print("dataset. See dissertation Section 4 for a critical discussion.")

        return self.cv_results

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------

    def _record_metrics(self, name, train_pred, val_pred, test_pred):
        self.results[name] = {
            'train_accuracy': accuracy_score(self.y_train, train_pred),
            'val_accuracy':   accuracy_score(self.y_val,   val_pred),
            'test_accuracy':  accuracy_score(self.y_test,  test_pred),
            'precision':  precision_score(self.y_test, test_pred, average='weighted'),
            'recall':     recall_score(self.y_test,    test_pred, average='weighted'),
            'f1_score':   f1_score(self.y_test,        test_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(self.y_test, test_pred),
            'predictions': test_pred,
        }
        r = self.results[name]
        print(f"  Train accuracy:      {r['train_accuracy']:.4f}")
        print(f"  Validation accuracy: {r['val_accuracy']:.4f}")
        print(f"  Test accuracy:       {r['test_accuracy']:.4f}")

    def train_random_forest(self):
        print("\n" + "="*60)
        print("Training Random Forest Classifier...")
        print("="*60)
        model = RandomForestClassifier(
            n_estimators=100, min_samples_leaf=1, max_features='sqrt',
            random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models[RF_NAME] = model
        self._record_metrics(RF_NAME,
                             model.predict(self.X_train),
                             model.predict(self.X_val),
                             model.predict(self.X_test))
        return model

    def train_decision_tree(self):
        print("\n" + "="*60)
        print("Training Decision Tree Classifier...")
        print("="*60)
        model = DecisionTreeClassifier(ccp_alpha=0.0, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models[DT_NAME] = model
        self._record_metrics(DT_NAME,
                             model.predict(self.X_train),
                             model.predict(self.X_val),
                             model.predict(self.X_test))
        return model

    def train_neural_network(self):
        print("\n" + "="*60)
        print("Training Neural Network (MLP)...")
        print("="*60)
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=500, random_state=42, early_stopping=True,
            validation_fraction=0.1)
        model.fit(self.X_train, self.y_train)
        self.models[NN_NAME] = model
        self._record_metrics(NN_NAME,
                             model.predict(self.X_train),
                             model.predict(self.X_val),
                             model.predict(self.X_test))
        return model

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def create_comparison_table(self):
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        rows = []
        for name, m in self.results.items():
            cv = self.cv_results.get(name, {})
            rows.append({
                'Model':         name,
                'Test Accuracy': f"{m['test_accuracy']:.4f}",
                'Precision':     f"{m['precision']:.4f}",
                'Recall':        f"{m['recall']:.4f}",
                'F1-Score':      f"{m['f1_score']:.4f}",
                'CV Accuracy':   f"{cv.get('accuracy_mean', 0):.4f} ± {cv.get('accuracy_std', 0):.4f}"
                                 if cv else 'N/A',
            })
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_accuracy_comparison(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        models = list(self.results.keys())
        train_acc = [self.results[m]['train_accuracy'] for m in models]
        val_acc   = [self.results[m]['val_accuracy']   for m in models]
        test_acc  = [self.results[m]['test_accuracy']  for m in models]
        x = np.arange(len(models))
        w = 0.25
        ax.bar(x - w, train_acc, w, label='Training',   alpha=0.8)
        ax.bar(x,     val_acc,   w, label='Validation', alpha=0.8)
        ax.bar(x + w, test_acc,  w, label='Test',       alpha=0.8)
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        for i, v in enumerate(test_acc):
            ax.text(i + w, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: accuracy_comparison.png")
        return fig

    def plot_cv_results(self):
        """Box plot of per-fold accuracy across models from cross-validation."""
        if not self.cv_results:
            print("Run cross-validation first.")
            return None
        fig, ax = plt.subplots(figsize=(10, 6))
        data   = [self.cv_results[m]['fold_accuracies'] for m in self.cv_results]
        labels = list(self.cv_results.keys())
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ['#4a90e2', '#27ae60', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('5-Fold Cross-Validation Accuracy per Model',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig('cv_results.png', dpi=300, bbox_inches='tight')
        print("Saved: cv_results.png")
        return fig

    def plot_metrics_comparison(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        models = list(self.results.keys())
        metric_keys   = ['test_accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(models))
        w = 0.2
        for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
            values = [self.results[m][key] for m in models]
            ax.bar(x + i*w, values, w, label=label, alpha=0.8)
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comprehensive Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: metrics_comparison.png")
        return fig

    def plot_confusion_matrices(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            cm_display = cm[:20, :20] if cm.shape[0] > 20 else cm
            suffix = ' (First 20 classes)' if cm.shape[0] > 20 else ''
            sns.heatmap(cm_display, annot=False, fmt='d', cmap='Blues',
                        ax=axes[idx], cbar=True, square=True)
            axes[idx].set_title(f'{name}{suffix}', fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontweight='bold')
            axes[idx].set_ylabel('Actual',    fontweight='bold')
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Saved: confusion_matrices.png")
        return fig

    def plot_feature_importance(self, top_n=15):
        if RF_NAME not in self.models:
            print("Random Forest model not trained yet.")
            return None
        rf = self.models[RF_NAME]
        importance = rf.feature_importances_
        names = self.X_train.columns
        indices = np.argsort(importance)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importance[indices], alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance (Random Forest)',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance.png")
        return fig

    # ------------------------------------------------------------------
    # Save models
    # ------------------------------------------------------------------

    def save_models(self, output_dir='../models'):
        os.makedirs(output_dir, exist_ok=True)

        model_files = {
            RF_NAME: 'random_forest.pkl',   # matches app.py load name
            DT_NAME: 'decision_tree.pkl',
            NN_NAME: 'neural_network.pkl',
        }

        for name, filename in model_files.items():
            if name in self.models:
                path = os.path.join(output_dir, filename)
                with open(path, 'wb') as f:
                    pickle.dump(self.models[name], f)
                print(f"Saved: {path}")

        encoder_path = os.path.join(output_dir, 'disease_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.disease_encoder, f)
        print(f"Saved: {encoder_path}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_full_report(self):
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        for name, m in self.results.items():
            cv = self.cv_results.get(name, {})
            print(f"\n{name}:")
            print(f"  Training Accuracy:   {m['train_accuracy']:.4f}")
            print(f"  Validation Accuracy: {m['val_accuracy']:.4f}")
            print(f"  Test Accuracy:       {m['test_accuracy']:.4f}")
            print(f"  Precision:           {m['precision']:.4f}")
            print(f"  Recall:              {m['recall']:.4f}")
            print(f"  F1-Score:            {m['f1_score']:.4f}")
            if cv:
                print(f"  CV Accuracy (5-fold):{cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
        print("\n[Note] All models achieve 100% on this synthetic dataset.")
        print("See the dissertation for a critical discussion of this limitation.")


if __name__ == "__main__":
    print("Medical Recommendation System - Model Training & Evaluation")
    print("="*60)

    base = os.path.dirname(os.path.abspath(__file__))
    evaluator = MedicalModelEvaluator(
        data_path=os.path.join(base, '..', 'data', 'Training.csv'),
        severity_path=os.path.join(base, '..', 'data', 'Symptom-severity.csv'),
    )

    evaluator.load_and_preprocess_data()

    # Cross-validation first (uses train+val folds, test set untouched)
    evaluator.run_cross_validation(n_splits=5)

    # Train final models on the full training set
    evaluator.train_random_forest()
    evaluator.train_decision_tree()
    evaluator.train_neural_network()

    evaluator.create_comparison_table()

    print("\nGenerating visualisations...")
    evaluator.plot_accuracy_comparison()
    evaluator.plot_cv_results()
    evaluator.plot_metrics_comparison()
    evaluator.plot_confusion_matrices()
    evaluator.plot_feature_importance()

    evaluator.generate_full_report()
    evaluator.save_models(output_dir=os.path.join(base, '..', 'models'))

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
