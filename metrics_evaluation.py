import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    average_precision_score, roc_curve, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

class RetinalFeatureMetrics:
    """
    Metrics evaluation using ALREADY-EXTRACTED features from app.py
    (No image processing - works with your existing feature dictionary)
    """

    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_prob):
        """Core classification metrics using pre-computed predictions"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'average_precision': average_precision_score(y_true, y_prob)
        }

    @staticmethod
    def calculate_medical_metrics(y_true, y_pred):
        """Clinical metrics using pre-computed predictions"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'sensitivity': tp / (tp + fn),
            'specificity': tn / (tn + fp),
            'ppv': tp / (tp + fp),  # Positive predictive value
            'npv': tn / (tn + fn),  # Negative predictive value
            'false_positive_rate': fp / (fp + tn),
            'false_negative_rate': fn / (fn + tp)
        }

    @staticmethod
    def analyze_features(features):
        """Analyze the extracted feature values themselves"""
        feature_values = np.array(list(features.values()))
        
        return {
            'feature_mean': float(np.mean(feature_values)),
            'feature_std': float(np.std(feature_values)),
            'dominant_feature': max(features, key=features.get),
            'max_value': float(np.max(feature_values)),
            'min_value': float(np.min(feature_values))
        }

    @staticmethod
    def generate_report(features, y_true=None, y_pred=None, y_prob=None):
        """
        Generate complete report from EXISTING features and predictions
        Args:
            features: Your extracted feature dictionary from app.py
            y_true: Ground truth if available (optional)
            y_pred: Model prediction if available (optional)
            y_prob: Prediction probability if available (optional)
        """
        report = {
            'feature_analysis': RetinalFeatureMetrics.analyze_features(features),
            'features_raw': features
        }
        
        if y_pred is not None and y_prob is not None:
            report['classification'] = {
                'standard_metrics': RetinalFeatureMetrics.calculate_classification_metrics(y_true, y_pred, y_prob),
                'medical_metrics': RetinalFeatureMetrics.calculate_medical_metrics(y_true, y_pred)
            }
            
            # Add curves if ground truth available
            if y_true is not None:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                report['classification']['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                report['classification']['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
        
        return report

    @staticmethod
    def visualize_feature_importance(features, save_path=None):
        """Visualize the relative importance of features"""
        names = list(features.keys())
        values = np.array(list(features.values()))
        
        # Normalize for visualization
        values = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=names, y=values)
        plt.title('Normalized Feature Importance')
        plt.xticks(rotation=45)
        plt.ylabel('Normalized Value')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

# Example Usage with YOUR APP's existing features
if __name__ == "__main__":
    # Mock data matching what your app already extracts
    sample_features = {
        'VesselDensity': 0.45,
        'Tortuosity': 1.2,
        'CupDiscRatio': 0.3,
        'LesionCount': 5,
        'EdgeComplexity': 0.67,
        'FractalDim': 1.5,
        'AVR': 0.8,
        'CRAE': 120,
        'CRVE': 150
    }
    
    # Generate report (no ground truth)
    report = RetinalFeatureMetrics.generate_report(sample_features)
    print("Feature Analysis Report:")
    print(json.dumps(report, indent=2))
    
    # Visualize
    RetinalFeatureMetrics.visualize_feature_importance(sample_features)