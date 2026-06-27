"""
Complete Evaluation Runner
Runs full ML pipeline: training, evaluation, visualization, and dashboard generation
"""

import sys
import os

# Get absolute paths based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Parent of app directory

# Define absolute paths
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'Training.csv')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Add parent directory to path
sys.path.append(SCRIPT_DIR)

from model_training_evaluation import MedicalModelEvaluator
from results_dashboard import generate_dashboard_from_evaluator

def main():
    """Run complete evaluation pipeline"""
    
    print("\n" + "="*70)
    print("MEDICAL RECOMMENDATION SYSTEM - COMPLETE EVALUATION PIPELINE")
    print("="*70)
    
    print("\n📁 Configuration:")
    print(f"   Data: {DATA_PATH}")
    print(f"   Models output: {MODELS_DIR}")
    print(f"   Results output: {RESULTS_DIR}")
    
    try:
        # Step 1: Initialize evaluator
        print("\n" + "-"*70)
        print("STEP 1: Initializing Evaluator")
        print("-"*70)
        evaluator = MedicalModelEvaluator(data_path=DATA_PATH)
        
        # Step 2: Load and preprocess data
        print("\n" + "-"*70)
        print("STEP 2: Loading and Preprocessing Data")
        print("-"*70)
        evaluator.load_and_preprocess_data()
        
        # Step 3: Train all models
        print("\n" + "-"*70)
        print("STEP 3: Training Machine Learning Models")
        print("-"*70)
        print("\nThis may take a few minutes...")
        
        evaluator.train_random_forest()
        evaluator.train_decision_tree()
        evaluator.train_neural_network()
        
      # Step 4: Generate comparison table
        print("\n" + "-"*70)
        print("STEP 4: Generating Comparison Analysis")
        print("-"*70)
        comparison_df = evaluator.create_comparison_table()
        evaluator.comparison_df = comparison_df  # Store it in the evaluator object
        
        # Step 5: Create visualizations
        print("\n" + "-"*70)
        print("STEP 5: Creating Visualizations")
        print("-"*70)
        
        # Save current directory and change to results directory
        original_dir = os.getcwd()
        os.chdir(RESULTS_DIR)
        
        print("Creating accuracy comparison chart...")
        evaluator.plot_accuracy_comparison()
        
        print("Creating metrics comparison chart...")
        evaluator.plot_metrics_comparison()
        
        print("Creating confusion matrices...")
        evaluator.plot_confusion_matrices()
        
        print("Creating feature importance chart...")
        evaluator.plot_feature_importance()
        
        # Change back to original directory
        os.chdir(original_dir)
        
        # Step 6: Generate full report
        print("\n" + "-"*70)
        print("STEP 6: Generating Comprehensive Report")
        print("-"*70)
        os.chdir(RESULTS_DIR)
        evaluator.generate_full_report()
        os.chdir(original_dir)
        
        # Step 7: Save models
        print("\n" + "-"*70)
        print("STEP 7: Saving Trained Models")
        print("-"*70)
        evaluator.save_models(output_dir=MODELS_DIR)
        
        # Step 8: Generate HTML dashboard
        print("\n" + "-"*70)
        print("STEP 8: Generating HTML Dashboard")
        print("-"*70)
        os.chdir(RESULTS_DIR)
        generate_dashboard_from_evaluator(evaluator)
        os.chdir(original_dir)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find required file")
        print(f"   {str(e)}")
        print(f"\n   Make sure Training.csv exists in: {DATA_PATH}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: An unexpected error occurred")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()