import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from config import *

class ModelTrainer:
    """Train KNN and Random Forest models for rockfall prediction with optimized performance"""
    
    def __init__(self):
        self.knn_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.tabular_data = None
        self.labels = None
        self.feature_selector = None
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(KNN_MODEL_PATH), exist_ok=True)
        
    def load_data(self):
        """Load and preprocess training data efficiently"""
        
        if not os.path.exists(TABULAR_DATA_PATH):
            raise FileNotFoundError("Training data not found. Please generate data first.")
        
        # Load tabular data
        self.tabular_data, self.labels = self._load_tabular()
        
        print(f"Loaded {len(self.tabular_data)} samples")
        print(f"Tabular data shape: {self.tabular_data.shape}")
        print(f"Label distribution: {np.unique(self.labels, return_counts=True)}")
    
    def _load_tabular(self):
        """Load and preprocess tabular data efficiently"""
        df = pd.read_csv(TABULAR_DATA_PATH)
        feature_cols = [col for col in df.columns if col not in ['risk_level', 'probability']]
        return df[feature_cols].values, df['risk_level'].values
    
    def train_knn_model(self):
        """Train KNN model with grid search for optimal parameters"""
        if self.tabular_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Scale data
        X_scaled = self.scaler.fit_transform(self.tabular_data)
        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(self.labels)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        # Grid search for optimal parameters
        print("Performing grid search for KNN parameters...")
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        grid_search = GridSearchCV(
            KNeighborsClassifier(),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        # Get best model
        self.knn_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best KNN parameters: {best_params}")
        # Evaluate model
        y_pred = self.knn_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        # Calculate precision, recall, f1-score (macro average)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('KNN Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(encoded_labels)))
        plt.xticks(tick_marks, self.label_encoder.classes_, rotation=45)
        plt.yticks(tick_marks, self.label_encoder.classes_)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('knn_confusion_matrix.png')
        # Return metrics as dict for UI compatibility
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'best_params': best_params
        }
    
    def train_models(self):
        """Train both KNN and RF models"""
        print("Starting model training...")
        
        # Load data if not already loaded
        if self.tabular_data is None:
            self.load_data()
        
        # Train models
        knn_accuracy = self.train_knn_model()
        rf_accuracy = self.train_rf_model()
        
        print(f"KNN Accuracy: {knn_accuracy:.4f}")
        print(f"RF Accuracy: {rf_accuracy:.4f}")
        
        # Save models
        self.save_models()
        
        return knn_accuracy, rf_accuracy
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        print("Saving models...")
        
        # Save KNN model
        joblib.dump(self.knn_model, KNN_MODEL_PATH)
        
        # Save RF model
        joblib.dump(self.rf_model, RF_MODEL_PATH)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        if self.feature_selector:
            joblib.dump(self.feature_selector, FEATURE_SELECTOR_PATH)
        
        print(f"Models saved to {os.path.dirname(KNN_MODEL_PATH)}")
        
    def plot_feature_importance(self):
        """Plot feature importance from Random Forest model"""
        if self.rf_model is None:
            raise ValueError("Random Forest model not trained. Call train_rf_model() first.")
        
        # Get feature names
        df = pd.read_csv(TABULAR_DATA_PATH)
        feature_cols = [col for col in df.columns if col not in ['risk_level', 'probability']]
        
        # Get feature importance
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
    def train_rf_model(self, n_estimators=100):
        """Train Random Forest model with optimized parameters"""
        if self.tabular_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        # Scale data
        X_scaled = self.scaler.fit_transform(self.tabular_data)
        # Feature selection
        print("Performing feature selection...")
        self.feature_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=50, random_state=42),
            max_features=10
        )
        X_selected = self.feature_selector.fit_transform(X_scaled, self.labels)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        # Train Random Forest with optimized parameters
        print("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train, y_train)
        # Evaluate model
        y_pred = self.rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        # Return metrics as dict for UI compatibility
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist()
        }
        
        # Feature importance analysis
        feature_names = [
            'temperature', 'humidity', 'rainfall', 'wind_speed',
            'soil_moisture', 'soil_density', 'cohesion', 'friction_angle',
            'slope_angle', 'fracture_density', 'rock_quality',
            'vibration_level', 'groundwater_level'
        ]
        
        selected_features = [feature_names[i] for i in range(len(feature_names)) 
                           if self.feature_selector.get_support()[i]]
        
        print("\nSelected features:", selected_features)
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'feature_importance': self.rf_model.feature_importances_,
            'selected_features': selected_features,
            'classification_report': report
        }
        
        return metrics
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        if self.knn_model:
            joblib.dump(self.knn_model, KNN_MODEL_PATH)
            print(f"KNN model saved to {KNN_MODEL_PATH}")
            
        if self.rf_model:
            joblib.dump(self.rf_model, RF_MODEL_PATH)
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
            if self.feature_selector:
                joblib.dump(self.feature_selector, 
                          os.path.join(MODELS_DIR, 'feature_selector.pkl'))
            print(f"Random Forest model and preprocessors saved to {MODELS_DIR}")


def run_complete_training():
    """Run complete training pipeline with optimized parameters"""
    
    print("Starting optimized training pipeline...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data with parallel processing
    print("Loading and preprocessing data...")
    trainer.load_data()
    
    # Train CNN model with optimized parameters
    print("\nTraining CNN model with optimized parameters...")
    cnn_history = trainer.train_cnn_model(
        epochs=15,          # Reduced epochs with early stopping
        batch_size=64,      # Increased batch size for faster training
        learning_rate=0.001 # Initial learning rate with adaptive reduction
    )
    
    # Train Random Forest model with optimized parameters
    print("\nTraining Random Forest model with feature selection...")
    rf_metrics = trainer.train_rf_model(n_estimators=100)
    
    # Save models and preprocessors
    print("\nSaving models and preprocessors...")
    trainer.save_models()
    
    print("\nTraining pipeline completed successfully!")
    print(f"Models and logs saved in {trainer.log_dir}")
    
    # Print performance metrics
    print("\nModel Performance Metrics:")
    print("-" * 30)
    print("CNN Final Validation Accuracy:", 
          f"{cnn_history['val_accuracy'][-1]:.4f}")
    print("\nRandom Forest Metrics:")
    for metric, value in rf_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
    
    print("\nSelected Features for Random Forest:")
    print(rf_metrics['selected_features'])
    
    return trainer, cnn_history, rf_metrics

if __name__ == "__main__":
    # Enable memory growth for GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    # Run optimized training pipeline
    trainer, cnn_history, rf_metrics = run_complete_training()
