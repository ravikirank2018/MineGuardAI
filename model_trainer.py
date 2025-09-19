import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
import matplotlib.pyplot as plt
from config import *

class ModelTrainer:
    """Train CNN and Random Forest models for rockfall prediction"""
    
    def __init__(self):
        self.cnn_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.images = None
        self.tabular_data = None
        self.labels = None
        
    def load_data(self):
        """Load training data from files"""
        
        if not os.path.exists(IMAGE_DATA_PATH) or not os.path.exists(TABULAR_DATA_PATH):
            raise FileNotFoundError("Training data not found. Please generate data first.")
        
        # Load images
        self.images = np.load(IMAGE_DATA_PATH)
        
        # Load tabular data
        df = pd.read_csv(TABULAR_DATA_PATH)
        
        # Extract features and labels
        feature_cols = [col for col in df.columns if col not in ['risk_level', 'probability']]
        self.tabular_data = df[feature_cols].values
        self.labels = df['risk_level'].values
        
        # Normalize image data
        self.images = self.images.astype('float32') / 255.0
        
        print(f"Loaded {len(self.images)} samples")
        print(f"Image shape: {self.images.shape}")
        print(f"Tabular data shape: {self.tabular_data.shape}")
        print(f"Label distribution: {np.unique(self.labels, return_counts=True)}")
        
    def build_cnn_model(self, input_shape, num_classes):
        """Build CNN architecture for image analysis"""
        
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_cnn_model(self, epochs=20, batch_size=32, learning_rate=0.001):
        """Train the CNN model for image classification"""
        
        if self.images is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Encode labels for CNN
        encoded_labels = self.label_encoder.fit_transform(self.labels)
        categorical_labels = to_categorical(encoded_labels)
        
        # Split data
        X_train_img, X_val_img, y_train_cat, y_val_cat = train_test_split(
            self.images, categorical_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # Build model
        input_shape = self.images.shape[1:]
        num_classes = len(np.unique(self.labels))
        
        self.cnn_model = self.build_cnn_model(input_shape, num_classes)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.cnn_model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
        
        # Train model
        print("Training CNN model...")
        history = self.cnn_model.fit(
            X_train_img, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_img, y_val_cat),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy = self.cnn_model.evaluate(X_val_img, y_val_cat, verbose=0)
        print(f"CNN Validation Accuracy: {val_accuracy:.4f}")
        
        return history.history
    
    def train_rf_model(self, n_estimators=100):
        """Train Random Forest model for tabular data"""
        
        if self.tabular_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Scale tabular data
        X_scaled = self.scaler.fit_transform(self.tabular_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        
        # Get detailed metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
        feature_names = [
            'temperature', 'humidity', 'rainfall', 'wind_speed',
            'soil_moisture', 'soil_density', 'cohesion', 'friction_angle',
            'slope_angle', 'fracture_density', 'rock_quality',
            'vibration_level', 'groundwater_level'
        ]
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'feature_importance': self.rf_model.feature_importances_,
            'feature_names': feature_names,
            'classification_report': report
        }
        
        return metrics
    
    def save_models(self):
        """Save trained models and preprocessing objects"""
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save CNN model
        if self.cnn_model is not None:
            self.cnn_model.save(CNN_MODEL_PATH)
            print(f"CNN model saved to {CNN_MODEL_PATH}")
        
        # Save Random Forest model
        if self.rf_model is not None:
            joblib.dump(self.rf_model, RF_MODEL_PATH)
            print(f"Random Forest model saved to {RF_MODEL_PATH}")
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, SCALER_PATH)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
        
        print("All models and preprocessors saved successfully!")
    
    def load_models(self):
        """Load trained models and preprocessing objects"""
        
        try:
            # Load CNN model
            if os.path.exists(CNN_MODEL_PATH):
                self.cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
                print("CNN model loaded successfully")
            
            # Load Random Forest model
            if os.path.exists(RF_MODEL_PATH):
                self.rf_model = joblib.load(RF_MODEL_PATH)
                print("Random Forest model loaded successfully")
            
            # Load preprocessors
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                print("Scaler loaded successfully")
            
            if os.path.exists(LABEL_ENCODER_PATH):
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                print("Label encoder loaded successfully")
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
    
    def evaluate_ensemble(self, test_images, test_tabular, test_labels):
        """Evaluate the ensemble of CNN and Random Forest models"""
        
        if self.cnn_model is None or self.rf_model is None:
            raise ValueError("Models not trained or loaded")
        
        # Get CNN predictions
        cnn_probs = self.cnn_model.predict(test_images)
        cnn_pred_classes = np.argmax(cnn_probs, axis=1)
        
        # Get RF predictions
        test_tabular_scaled = self.scaler.transform(test_tabular)
        rf_probs = self.rf_model.predict_proba(test_tabular_scaled)
        rf_pred_classes = self.rf_model.predict(test_tabular_scaled)
        
        # Ensemble predictions (weighted average)
        cnn_weight = 0.6
        rf_weight = 0.4
        
        ensemble_probs = cnn_weight * cnn_probs + rf_weight * rf_probs
        ensemble_pred_classes = np.argmax(ensemble_probs, axis=1)
        
        # Convert test labels to encoded format
        test_labels_encoded = self.label_encoder.transform(test_labels)
        
        # Calculate metrics
        cnn_accuracy = accuracy_score(test_labels_encoded, cnn_pred_classes)
        rf_accuracy = accuracy_score(test_labels_encoded, rf_pred_classes)
        ensemble_accuracy = accuracy_score(test_labels_encoded, ensemble_pred_classes)
        
        results = {
            'cnn_accuracy': cnn_accuracy,
            'rf_accuracy': rf_accuracy,
            'ensemble_accuracy': ensemble_accuracy,
            'cnn_predictions': cnn_pred_classes,
            'rf_predictions': rf_pred_classes,
            'ensemble_predictions': ensemble_pred_classes,
            'ensemble_probabilities': ensemble_probs
        }
        
        print(f"CNN Accuracy: {cnn_accuracy:.4f}")
        print(f"RF Accuracy: {rf_accuracy:.4f}")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        
        return results
    
    def plot_training_history(self, history):
        """Plot training history"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Training pipeline
def run_complete_training():
    """Run complete training pipeline"""
    
    print("Starting complete training pipeline...")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    trainer.load_data()
    
    # Train CNN model
    cnn_history = trainer.train_cnn_model(epochs=20, batch_size=32)
    
    # Train Random Forest model
    rf_metrics = trainer.train_rf_model(n_estimators=100)
    
    # Save models
    trainer.save_models()
    
    print("Training pipeline completed successfully!")
    
    return trainer, cnn_history, rf_metrics

if __name__ == "__main__":
    # Run training pipeline
    trainer, cnn_history, rf_metrics = run_complete_training()
    
    # Plot training history
    trainer.plot_training_history(cnn_history)
    
    print("\nRandom Forest Metrics:")
    for metric, value in rf_metrics.items():
        if metric not in ['feature_importance', 'feature_names', 'classification_report']:
            print(f"{metric}: {value:.4f}")
