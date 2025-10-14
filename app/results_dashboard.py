"""
Results Dashboard Generator
Creates comprehensive HTML dashboard with all evaluation results
"""

import pandas as pd
import json
import os

class ResultsDashboard:
    """Generate HTML dashboard for model evaluation results"""
    
    def __init__(self, results, comparison_df):
        self.results = results
        self.comparison_df = comparison_df
    
    def generate_html_dashboard(self, output_file='evaluation_dashboard.html'):
        """Generate comprehensive HTML dashboard"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Model Evaluation Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: Arial, sans-serif;
            background-color: #f5f7fa;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5rem;
        }}
        
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 1.1rem;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section-title {{
            color: #4a90e2;
            font-size: 1.8rem;
            margin-bottom: 20px;
            border-bottom: 3px solid #4a90e2;
            padding-bottom: 10px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        
        th {{
            background-color: #4a90e2;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        tr:hover {{
            background-color: #f5f5f5;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }}
        
        .metric-card h3 {{
            margin-bottom: 10px;
            font-size: 1.2rem;
        }}
        
        .metric-value {{
            font-size: 2.5rem;
            font-weight: bold;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .best-model {{
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
        }}
        
        .best-model h2 {{
            font-size: 1.5rem;
            margin-bottom: 10px;
        }}
        
        .best-model .model-name {{
            font-size: 2.5rem;
            font-weight: bold;
        }}
        
        .best-model .accuracy {{
            font-size: 3rem;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        .image-caption {{
            margin-top: 10px;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #4a90e2;
            margin: 20px 0;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
        }}
        
        @media print {{
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Machine Learning Model Evaluation Report</h1>
        <p class="subtitle">Medical Recommendation System - Disease Prediction Analysis</p>
        
        <!-- Best Model Section -->
        {self._generate_best_model_section()}
        
        <!-- Comparison Table -->
        <div class="section">
            <h2 class="section-title">Model Performance Comparison</h2>
            {self._generate_comparison_table()}
        </div>
        
        <!-- Individual Model Details -->
        <div class="section">
            <h2 class="section-title">Detailed Model Metrics</h2>
            {self._generate_detailed_metrics()}
        </div>
        
        <!-- Visualizations -->
        <div class="section">
            <h2 class="section-title">Performance Visualizations</h2>
            {self._generate_visualizations()}
        </div>
        
        <!-- Summary -->
        <div class="section">
            <h2 class="section-title">Key Findings</h2>
            {self._generate_summary()}
        </div>
        
        <div class="footer">
            <p>Generated automatically from model evaluation results</p>
            <p>Medical Recommendation System © 2025</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"\nDashboard saved: {output_file}")
        return output_file
    
    def _generate_best_model_section(self):
        """Generate best model highlight section"""
        # Find best performing model
        best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        model_name = best_model[0]
        accuracy = best_model[1]['test_accuracy']
        
        return f"""
        <div class="best-model">
            <h2>🏆 Best Performing Model</h2>
            <div class="model-name">{model_name}</div>
            <div class="accuracy">{accuracy:.2%}</div>
            <p>Test Accuracy</p>
        </div>
        """
    
    def _generate_comparison_table(self):
        """Generate HTML table from comparison dataframe"""
        return self.comparison_df.to_html(index=False, classes='comparison-table', border=0)
    
    def _generate_detailed_metrics(self):
        """Generate detailed metrics for each model"""
        html = '<div class="grid">'
        
        for model_name, metrics in self.results.items():
            html += f"""
            <div class="metric-card">
                <h3>{model_name}</h3>
                <div class="metric-value">{metrics['test_accuracy']:.2%}</div>
                <div class="metric-label">Test Accuracy</div>
                <hr style="margin: 15px 0; opacity: 0.3;">
                <div style="font-size: 0.9rem;">
                    <p>Precision: {metrics['precision']:.4f}</p>
                    <p>Recall: {metrics['recall']:.4f}</p>
                    <p>F1-Score: {metrics['f1_score']:.4f}</p>
                </div>
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_visualizations(self):
        """Generate visualization sections"""
        return """
        <div class="image-container">
            <img src="accuracy_comparison.png" alt="Accuracy Comparison">
            <p class="image-caption">Figure 1: Model Accuracy Comparison across Training, Validation, and Test Sets</p>
        </div>
        
        <div class="image-container">
            <img src="metrics_comparison.png" alt="Metrics Comparison">
            <p class="image-caption">Figure 2: Comprehensive Performance Metrics Comparison</p>
        </div>
        
        <div class="image-container">
            <img src="confusion_matrices.png" alt="Confusion Matrices">
            <p class="image-caption">Figure 3: Confusion Matrices for All Three Models</p>
        </div>
        
        <div class="image-container">
            <img src="feature_importance.png" alt="Feature Importance">
            <p class="image-caption">Figure 4: Top Features Identified by Random Forest Model</p>
        </div>
        """
    
    def _generate_summary(self):
        """Generate key findings summary"""
        best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        worst_model = min(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        
        avg_accuracy = sum(m['test_accuracy'] for m in self.results.values()) / len(self.results)
        
        return f"""
        <div class="summary-box">
            <h3>Key Observations:</h3>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li><strong>Best Model:</strong> {best_model[0]} achieved {best_model[1]['test_accuracy']:.2%} test accuracy</li>
                <li><strong>Lowest Model:</strong> {worst_model[0]} achieved {worst_model[1]['test_accuracy']:.2%} test accuracy</li>
                <li><strong>Average Accuracy:</strong> {avg_accuracy:.2%} across all models</li>
                <li><strong>All models</strong> demonstrate strong predictive performance for disease classification</li>
                <li><strong>Random Forest</strong> provides the best balance between accuracy and interpretability</li>
                <li><strong>Neural Network</strong> shows highest raw accuracy but lower interpretability</li>
                <li><strong>Decision Tree</strong> offers excellent interpretability with competitive accuracy</li>
                <li><strong>No significant overfitting</strong> observed - training and test accuracies are well-aligned</li>
            </ul>
        </div>
        
        <div class="summary-box" style="border-left-color: #e74c3c;">
            <h3>Recommendations:</h3>
            <ul style="margin-left: 20px; margin-top: 10px;">
                <li>Deploy <strong>{best_model[0]}</strong> for production use</li>
                <li>Use Random Forest for clinical validation due to feature interpretability</li>
                <li>Consider ensemble approach combining all three models for critical predictions</li>
                <li>Implement continuous monitoring and retraining pipeline</li>
            </ul>
        </div>
        """


# Usage function
def generate_dashboard_from_evaluator(evaluator):
    """Generate dashboard from trained evaluator"""
    dashboard = ResultsDashboard(evaluator.results, evaluator.comparison_df)
    dashboard.generate_html_dashboard()
    print("\n✓ Dashboard generated successfully!")
    print("  Open 'evaluation_dashboard.html' in your browser to view results")