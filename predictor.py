import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import *

class RockfallPredictor:
    """Make predictions using trained CNN and Random Forest models"""
    
    def __init__(self):
        self.cnn_model = None
        self.rf_model = None
        self.scaler = None
        self.label_encoder = None
        self.load_models()
        
    def load_models(self):
        """Load all trained models and preprocessors"""
        
        try:
            # Load CNN model
            if os.path.exists(CNN_MODEL_PATH):
                self.cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
                print("CNN model loaded successfully")
            else:
                print(f"Warning: CNN model not found at {CNN_MODEL_PATH}")
            
            # Load Random Forest model
            if os.path.exists(RF_MODEL_PATH):
                self.rf_model = joblib.load(RF_MODEL_PATH)
                print("Random Forest model loaded successfully")
            else:
                print(f"Warning: RF model not found at {RF_MODEL_PATH}")
            
            # Load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                print("Scaler loaded successfully")
            else:
                print(f"Warning: Scaler not found at {SCALER_PATH}")
            
            # Load label encoder
            if os.path.exists(LABEL_ENCODER_PATH):
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                print("Label encoder loaded successfully")
            else:
                print(f"Warning: Label encoder not found at {LABEL_ENCODER_PATH}")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        """Preprocess image for CNN prediction"""
        
        if image is None:
            raise ValueError("Image cannot be None")
        
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        
        # Ensure correct shape (add batch dimension if needed)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        return image
    
    def preprocess_tabular(self, tabular_data):
        """Preprocess tabular data for Random Forest prediction"""
        
        if tabular_data is None:
            raise ValueError("Tabular data cannot be None")
        
        # Convert to numpy array if needed
        if not isinstance(tabular_data, np.ndarray):
            tabular_data = np.array(tabular_data)
        
        # Ensure correct shape
        if len(tabular_data.shape) == 1:
            tabular_data = tabular_data.reshape(1, -1)
        
        # Scale data
        if self.scaler is not None:
            tabular_data = self.scaler.transform(tabular_data)
        
        return tabular_data
    
    def predict_cnn(self, image):
        """Make prediction using CNN model"""
        
        if self.cnn_model is None:
            raise ValueError("CNN model not loaded")
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        cnn_probs = self.cnn_model.predict(processed_image, verbose=0)
        
        return cnn_probs[0]  # Return single prediction
    
    def predict_rf(self, tabular_data):
        """Make prediction using Random Forest model"""
        
        if self.rf_model is None:
            raise ValueError("Random Forest model not loaded")
        
        # Preprocess tabular data
        processed_data = self.preprocess_tabular(tabular_data)
        
        # Make prediction
        rf_probs = self.rf_model.predict_proba(processed_data)
        
        return rf_probs[0]  # Return single prediction
    
    def predict_ensemble(self, image, tabular_data, cnn_weight=0.6, rf_weight=0.4):
        """Make ensemble prediction combining CNN and Random Forest"""
        
        # Get individual predictions
        cnn_probs = self.predict_cnn(image)
        rf_probs = self.predict_rf(tabular_data)
        
        # Combine predictions using weighted average
        ensemble_probs = cnn_weight * cnn_probs + rf_weight * rf_probs
        
        return ensemble_probs, cnn_probs, rf_probs
    
    def predict(self, image, tabular_data, return_details=False):
        """Make final rockfall prediction with risk level and probability"""
        
        try:
            # Get ensemble prediction
            ensemble_probs, cnn_probs, rf_probs = self.predict_ensemble(image, tabular_data)
            
            # Get predicted class
            predicted_class_idx = np.argmax(ensemble_probs)
            
            # Convert to risk level
            if self.label_encoder is not None:
                risk_levels = self.label_encoder.classes_
                predicted_risk = risk_levels[predicted_class_idx]
            else:
                # Fallback mapping
                risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
                predicted_risk = risk_mapping.get(predicted_class_idx, 'Unknown')
            
            # Get probability score (max probability from ensemble)
            probability_score = np.max(ensemble_probs)
            
            # Alternative: use weighted average of high-risk probabilities
            if len(ensemble_probs) >= 3:  # Assuming Low, Medium, High
                # Weight probabilities: Low=0.1, Medium=0.5, High=0.9
                risk_weights = np.array([0.1, 0.5, 0.9])
                probability_score = np.sum(ensemble_probs * risk_weights)
            
            if return_details:
                details = {
                    'risk_level': predicted_risk,
                    'probability': probability_score,
                    'ensemble_probabilities': ensemble_probs,
                    'cnn_probabilities': cnn_probs,
                    'rf_probabilities': rf_probs,
                    'predicted_class_idx': predicted_class_idx,
                    'individual_scores': {
                        'cnn_max_prob': np.max(cnn_probs),
                        'rf_max_prob': np.max(rf_probs)
                    }
                }
                return details
            
            return probability_score, predicted_risk
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Return default values in case of error
            return 0.0, 'Unknown'
    
    def batch_predict(self, images, tabular_data_list):
        """Make predictions for multiple samples"""
        
        predictions = []
        
        for i, (image, tabular_data) in enumerate(zip(images, tabular_data_list)):
            try:
                probability, risk_level = self.predict(image, tabular_data)
                predictions.append({
                    'sample_id': i,
                    'probability': probability,
                    'risk_level': risk_level
                })
            except Exception as e:
                print(f"Error predicting sample {i}: {str(e)}")
                predictions.append({
                    'sample_id': i,
                    'probability': 0.0,
                    'risk_level': 'Unknown'
                })
        
        return predictions
    
    def get_risk_threshold_classification(self, probability):
        """Classify risk based on probability thresholds"""
        
        if probability >= HIGH_RISK_THRESHOLD:
            return 'High'
        elif probability >= MEDIUM_RISK_THRESHOLD:
            return 'Medium'
        else:
            return 'Low'
    
    def analyze_prediction_confidence(self, ensemble_probs):
        """Analyze confidence of the prediction"""
        
        # Calculate entropy (uncertainty measure)
        entropy = -np.sum(ensemble_probs * np.log(ensemble_probs + 1e-8))
        max_entropy = np.log(len(ensemble_probs))
        normalized_entropy = entropy / max_entropy
        
        # Calculate confidence (1 - normalized_entropy)
        confidence = 1 - normalized_entropy
        
        # Alternative confidence measure: difference between top 2 probabilities
        sorted_probs = np.sort(ensemble_probs)[::-1]
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        return {
            'confidence': confidence,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'margin': margin,
            'max_probability': np.max(ensemble_probs)
        }
    
    def get_feature_importance_explanation(self, tabular_data):
        """Get explanation of which features contribute most to the prediction"""
        
        if self.rf_model is None:
            return None
        
        # Get feature importance from Random Forest
        feature_names = [
            'temperature', 'humidity', 'rainfall', 'wind_speed',
            'soil_moisture', 'soil_density', 'cohesion', 'friction_angle',
            'slope_angle', 'fracture_density', 'rock_quality',
            'vibration_level', 'groundwater_level'
        ]
        
        feature_importance = self.rf_model.feature_importances_
        
        # Combine with actual values
        processed_data = self.preprocess_tabular(tabular_data)
        feature_values = processed_data[0] if len(processed_data) > 0 else tabular_data
        
        # Create explanation
        explanation = []
        for i, (name, importance, value) in enumerate(zip(feature_names, feature_importance, feature_values)):
            explanation.append({
                'feature': name,
                'importance': importance,
                'value': value,
                'contribution': importance * abs(value)  # Simplified contribution
            })
        
        # Sort by contribution
        explanation.sort(key=lambda x: x['contribution'], reverse=True)
        
        return explanation[:5]  # Return top 5 contributing features

def test_predictor():
    """Test the predictor with sample data"""
    
    try:
        predictor = RockfallPredictor()
        
        # Create sample data
        sample_image = np.random.rand(64, 64, 3) * 255
        sample_tabular = np.random.rand(13)  # 13 features
        
        # Make prediction
        probability, risk_level = predictor.predict(sample_image, sample_tabular)
        
        print(f"Sample prediction:")
        print(f"Probability: {probability:.4f}")
        print(f"Risk Level: {risk_level}")
        
        # Get detailed prediction
        details = predictor.predict(sample_image, sample_tabular, return_details=True)
        print(f"Detailed prediction: {details}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Test the predictor
    test_predictor()
