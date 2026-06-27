"""
Medical Recommendation System - ML Model Training & Evaluation
This script trains and evaluates three models: Random Forest, Decision Tree, and Neural Network
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MedicalModelEvaluator:
    """
    A class to train and evaluate multiple ML models for disease prediction
    """
    
    def __init__(self, data_path):
        """Initialize with dataset path"""
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.disease_encoder = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the training data"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Number of diseases: {df['prognosis'].nunique()}")
        
        # Separate features and target
        X = df.drop('prognosis', axis=1)
        y = df['prognosis']
        
        # Encode disease labels
        self.disease_encoder = LabelEncoder()
        y_encoded = self.disease_encoder.fit_transform(y)
        
        # Split data: 70% train, 15% validation, 15% test
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
        )
        
        # Second split: 15% validation, 15% test from the 30% temp
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"\nData split completed:")
        print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
        print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
        print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self):
        """Train Random Forest Classifier"""
        print("\n" + "="*60)
        print("Training Random Forest Classifier...")
        print("="*60)
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf_model
        
        # Predictions
        train_pred = rf_model.predict(self.X_train)
        val_pred = rf_model.predict(self.X_val)
        test_pred = rf_model.predict(self.X_test)
        
        # Calculate metrics
        self.results['Random Forest'] = {
            'train_accuracy': accuracy_score(self.y_train, train_pred),
            'val_accuracy': accuracy_score(self.y_val, val_pred),
            'test_accuracy': accuracy_score(self.y_test, test_pred),
            'precision': precision_score(self.y_test, test_pred, average='weighted'),
            'recall': recall_score(self.y_test, test_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, test_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(self.y_test, test_pred),
            'predictions': test_pred
        }
        
        print(f"Train Accuracy: {self.results['Random Forest']['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {self.results['Random Forest']['val_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['Random Forest']['test_accuracy']:.4f}")
        
        return rf_model
    
    def train_decision_tree(self):
        """Train Decision Tree Classifier"""
        print("\n" + "="*60)
        print("Training Decision Tree Classifier...")
        print("="*60)
        
        dt_model = DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        dt_model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = dt_model
        
        # Predictions
        train_pred = dt_model.predict(self.X_train)
        val_pred = dt_model.predict(self.X_val)
        test_pred = dt_model.predict(self.X_test)
        
        # Calculate metrics
        self.results['Decision Tree'] = {
            'train_accuracy': accuracy_score(self.y_train, train_pred),
            'val_accuracy': accuracy_score(self.y_val, val_pred),
            'test_accuracy': accuracy_score(self.y_test, test_pred),
            'precision': precision_score(self.y_test, test_pred, average='weighted'),
            'recall': recall_score(self.y_test, test_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, test_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(self.y_test, test_pred),
            'predictions': test_pred
        }
        
        print(f"Train Accuracy: {self.results['Decision Tree']['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {self.results['Decision Tree']['val_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['Decision Tree']['test_accuracy']:.4f}")
        
        return dt_model
    
    def train_neural_network(self):
        """Train Multi-Layer Perceptron Neural Network"""
        print("\n" + "="*60)
        print("Training Neural Network (MLP)...")
        print("="*60)
        
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        mlp_model.fit(self.X_train, self.y_train)
        self.models['Neural Network'] = mlp_model
        
        # Predictions
        train_pred = mlp_model.predict(self.X_train)
        val_pred = mlp_model.predict(self.X_val)
        test_pred = mlp_model.predict(self.X_test)
        
        # Calculate metrics
        self.results['Neural Network'] = {
            'train_accuracy': accuracy_score(self.y_train, train_pred),
            'val_accuracy': accuracy_score(self.y_val, val_pred),
            'test_accuracy': accuracy_score(self.y_test, test_pred),
            'precision': precision_score(self.y_test, test_pred, average='weighted'),
            'recall': recall_score(self.y_test, test_pred, average='weighted'),
            'f1_score': f1_score(self.y_test, test_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(self.y_test, test_pred),
            'predictions': test_pred
        }
        
        print(f"Train Accuracy: {self.results['Neural Network']['train_accuracy']:.4f}")
        print(f"Validation Accuracy: {self.results['Neural Network']['val_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['Neural Network']['test_accuracy']:.4f}")
        
        return mlp_model
    
    def create_comparison_table(self):
        """Create comparison table of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        return df_comparison
    
    def plot_accuracy_comparison(self):
        """Plot accuracy comparison across all models"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(self.results.keys())
        train_acc = [self.results[m]['train_accuracy'] for m in models]
        val_acc = [self.results[m]['val_accuracy'] for m in models]
        test_acc = [self.results[m]['test_accuracy'] for m in models]
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, train_acc, width, label='Training', alpha=0.8)
        ax.bar(x, val_acc, width, label='Validation', alpha=0.8)
        ax.bar(x + width, test_acc, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, v in enumerate(test_acc):
            ax.text(i + width, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print("\nSaved: accuracy_comparison.png")
        
        return fig
    
    def plot_metrics_comparison(self):
        """Plot all metrics comparison"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        models = list(self.results.keys())
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(['test_accuracy', 'precision', 'recall', 'f1_score']):
            values = [self.results[m][metric] for m in models]
            ax.bar(x + i*width, values, width, label=metrics_names[i], alpha=0.8)
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comprehensive Model Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved: metrics_comparison.png")
        
        return fig
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            
            # For readability, show only first 20 classes if there are many
            if cm.shape[0] > 20:
                cm_display = cm[:20, :20]
                title_suffix = " (First 20 classes)"
            else:
                cm_display = cm
                title_suffix = ""
            
            sns.heatmap(cm_display, annot=False, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=True, square=True)
            axes[idx].set_title(f'{model_name}{title_suffix}', fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontweight='bold')
            axes[idx].set_ylabel('Actual', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Saved: confusion_matrices.png")
        
        return fig
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance for Random Forest"""
        if 'Random Forest' not in self.models:
            print("Random Forest model not trained yet!")
            return
        
        rf_model = self.models['Random Forest']
        feature_importance = rf_model.feature_importances_
        feature_names = self.X_train.columns
        
        # Get top N features
        indices = np.argsort(feature_importance)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), feature_importance[indices], alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importance (Random Forest)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance.png")
        
        return fig
    
    def save_models(self, output_dir='../models'):
        """Save trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Random Forest
        with open(f'{output_dir}/rf_model.pkl', 'wb') as f:
            pickle.dump(self.models['Random Forest'], f)
        print(f"\nSaved: {output_dir}/rf_model.pkl")
        
        # Save Decision Tree
        with open(f'{output_dir}/decision_tree.pkl', 'wb') as f:
            pickle.dump(self.models['Decision Tree'], f)
        print(f"Saved: {output_dir}/decision_tree.pkl")
        
        # Save Neural Network
        with open(f'{output_dir}/neural_network.pkl', 'wb') as f:
            pickle.dump(self.models['Neural Network'], f)
        print(f"Saved: {output_dir}/neural_network.pkl")
        
        # Save encoder
        with open(f'{output_dir}/disease_encoder.pkl', 'wb') as f:
            pickle.dump(self.disease_encoder, f)
        print(f"Saved: {output_dir}/disease_encoder.pkl")
    
    def generate_full_report(self):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*60)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name}:")
            print(f"  Training Accuracy:   {metrics['train_accuracy']:.4f}")
            print(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"  Test Accuracy:       {metrics['test_accuracy']:.4f}")
            print(f"  Precision:           {metrics['precision']:.4f}")
            print(f"  Recall:              {metrics['recall']:.4f}")
            print(f"  F1-Score:            {metrics['f1_score']:.4f}")


# Main execution
if __name__ == "__main__":
    print("Medical Recommendation System - Model Training & Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = MedicalModelEvaluator(data_path='../data/Training.csv')
    
    # Load and preprocess data
    evaluator.load_and_preprocess_data()
    
    # Train all models
    evaluator.train_random_forest()
    evaluator.train_decision_tree()
    evaluator.train_neural_network()
    
    # Generate comparison table
    comparison_df = evaluator.create_comparison_table()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    evaluator.plot_accuracy_comparison()
    evaluator.plot_metrics_comparison()
    evaluator.plot_confusion_matrices()
    evaluator.plot_feature_importance()
    
    # Generate full report
    evaluator.generate_full_report()
    
    # Save models
    evaluator.save_models()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - accuracy_comparison.png")
    print("  - metrics_comparison.png")
    print("  - confusion_matrices.png")
    print("  - feature_importance.png")
    print("  - Models saved in ../models/ directory")