import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class KNNTrainer:
    """KNN model training with hyperparameter tuning"""
    
    def __init__(self, data_dir="data/features", model_dir="saved_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """Load pre-extracted features from Phase 2"""
        try:
            X_train = np.load(self.data_dir / "X_train.npy")
            X_val = np.load(self.data_dir / "X_val.npy")
            X_test = np.load(self.data_dir / "X_test.npy")
            y_train = np.load(self.data_dir / "y_train.npy")
            y_val = np.load(self.data_dir / "y_val.npy")
            y_test = np.load(self.data_dir / "y_test.npy")
            
            print(f"[OK] Data loaded - Features shape: {X_train.shape}")
            return X_train, X_val, X_test, y_train, y_val, y_test
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Features not found. Run phase2_feature_extraction.py first. Error: {e}")
    
    def train_with_grid_search(self, X_train, y_train, X_val=None, y_val=None):
        """Hyperparameter tuning for KNN"""
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        print("\n[TUNING] Tuning KNN hyperparameters...")
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"\n[OK] Best Parameters: {grid_search.best_params_}")
        print(f"[OK] Best CV Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_model(self, model, X_train, X_val, X_test, y_train, y_val, y_test, class_names=None):
        """Evaluate KNN on all sets"""
        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        print(f"\n{'='*50}")
        print(f"KNN MODEL PERFORMANCE")
        print(f"{'='*50}")
        print(f"Training Accuracy:   {train_acc:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Test Accuracy:       {test_acc:.4f}")
        print(f"{'='*50}")
        
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, model.predict(X_test), target_names=class_names))
        
        cm = confusion_matrix(y_test, model.predict(X_test))
        
        return {
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, model, best_params, metrics, feature_dim):
        """Save KNN model and config"""
        with open(self.model_dir / "knn_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        config = {
            'model_type': 'KNN',
            'best_params': best_params,
            'metrics': metrics,
            'feature_dimension': feature_dim
        }
        
        with open(self.model_dir / "knn_config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"[OK] KNN model saved to {self.model_dir / 'knn_model.pkl'}")
        print(f"[OK] Config saved to {self.model_dir / 'knn_config.json'}")

    def train(self):
        """Full training pipeline"""
        # Load features
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # Train with hyperparameter tuning
        best_model, best_params = self.train_with_grid_search(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = self.evaluate_model(best_model, X_train, X_val, X_test, 
                                     y_train, y_val, y_test,
                                     class_names=['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash', 'unknown'])
        
        # Save
        self.save_model(best_model, best_params, metrics, X_train.shape[1])


if __name__ == "__main__":
    trainer = KNNTrainer()
    trainer.train()
