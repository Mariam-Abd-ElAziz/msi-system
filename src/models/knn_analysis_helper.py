import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class KNNAnalyzer:
    """KNN model analysis and visualization"""
    
    def __init__(self, model_dir="saved_models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Load KNN model and config"""
        try:
            with open(self.model_dir / "knn_model.pkl", "rb") as f:
                self.model = pickle.load(f)
            
            with open(self.model_dir / "knn_config.json") as f:
                self.config = json.load(f)
            print("[OK] KNN model and config loaded")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"KNN model files not found. Run knn_training.py first. Error: {e}")
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, save=True):
        """Visualize confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': 'Count'})
        plt.title("KNN - Confusion Matrix", fontsize=14, fontweight='bold')
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.model_dir / "knn_confusion_matrix.png", dpi=300, bbox_inches='tight')
            print(f"[OK] Confusion matrix saved to {self.model_dir / 'knn_confusion_matrix.png'}")
        plt.show()
        
        return cm
    
    def get_model_info(self):
        """Print model information"""
        print(f"\n{'='*50}")
        print(f"KNN MODEL INFORMATION")
        print(f"{'='*50}")
        print(f"Model Type: {self.config['model_type']}")
        print(f"Feature Dimension: {self.config['feature_dimension']}")
        print(f"\nBest Parameters:")
        for param, value in self.config['best_params'].items():
            print(f"  {param}: {value}")
        print(f"\nMetrics:")
        for metric, value in self.config['metrics'].items():
            if metric != 'confusion_matrix':
                print(f"  {metric}: {value:.4f}")
        print(f"{'='*50}\n")
    
    def compare_models(self, svm_config_path="saved_models/svm_config.json"):
        """Compare KNN vs SVM performance"""
        try:
            with open(svm_config_path) as f:
                svm_config = json.load(f)
        except FileNotFoundError:
            print("[WARNING] SVM config not found. Skipping comparison.")
            return
        
        models = ['KNN', 'SVM']
        train_acc = [
            self.config['metrics']['train_accuracy'],
            svm_config['metrics']['train_accuracy']
        ]
        val_acc = [
            self.config['metrics']['val_accuracy'],
            svm_config['metrics']['val_accuracy']
        ]
        test_acc = [
            self.config['metrics']['test_accuracy'],
            svm_config['metrics']['test_accuracy']
        ]
        
        x = np.arange(len(models))
        width = 0.25
        
        plt.figure(figsize=(10, 6))
        plt.bar(x - width, train_acc, width, label='Train', alpha=0.8)
        plt.bar(x, val_acc, width, label='Validation', alpha=0.8)
        plt.bar(x + width, test_acc, width, label='Test', alpha=0.8)
        
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('KNN vs SVM Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, models)
        plt.legend()
        plt.ylim([0, 1.0])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.model_dir / "knn_vs_svm_comparison.png", dpi=300, bbox_inches='tight')
        print(f"[OK] Comparison plot saved to {self.model_dir / 'knn_vs_svm_comparison.png'}")
        plt.show()
        
        # Print comparison table
        print(f"\n{'='*60}")
        print(f"{'Metric':<20} {'KNN':<15} {'SVM':<15}")
        print(f"{'='*60}")
        print(f"{'Train Accuracy':<20} {train_acc[0]:<15.4f} {train_acc[1]:<15.4f}")
        print(f"{'Val Accuracy':<20} {val_acc[0]:<15.4f} {val_acc[1]:<15.4f}")
        print(f"{'Test Accuracy':<20} {test_acc[0]:<15.4f} {test_acc[1]:<15.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    analyzer = KNNAnalyzer()
    analyzer.get_model_info()
    analyzer.compare_models()
