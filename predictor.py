import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import *

class RockfallPredictor:
    """Make predictions using trained KNN and Random Forest models"""
    
    def __init__(self):
        self.knn_model = None
        self.rf_model = None
        self.scaler = None
        self.label_encoder = None
        self.load_models()
        
    def load_models(self):
        """Load all trained models and preprocessors"""
        
        try:
            # Load KNN model
            if os.path.exists(KNN_MODEL_PATH):
                try:
                    self.knn_model = joblib.load(KNN_MODEL_PATH)
                    print("KNN model loaded successfully")
                except Exception as e:
                    print(f"Error loading KNN model: {str(e)}")
                    self.knn_model = None
            else:
                print(f"Warning: KNN model not found at {KNN_MODEL_PATH}")
            
            # Load Random Forest model
            if os.path.exists(RF_MODEL_PATH):
                try:
                    self.rf_model = joblib.load(RF_MODEL_PATH)
                    print("Random Forest model loaded successfully")
                except Exception as e:
                    print(f"Error loading RF model: {str(e)}")
                    self.rf_model = None
            else:
                print(f"Warning: RF model not found at {RF_MODEL_PATH}")
            
            # Load scaler
            if os.path.exists(SCALER_PATH):
                try:
                    self.scaler = joblib.load(SCALER_PATH)
                    print("Scaler loaded successfully")
                except Exception as e:
                    print(f"Error loading scaler: {str(e)}")
                    self.scaler = None
            else:
                print(f"Warning: Scaler not found at {SCALER_PATH}")
            
            # Load label encoder
            if os.path.exists(LABEL_ENCODER_PATH):
                try:
                    self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                    print("Label encoder loaded successfully")
                except Exception as e:
                    print(f"Error loading label encoder: {str(e)}")
                    self.label_encoder = None
            else:
                print(f"Warning: Label encoder not found at {LABEL_ENCODER_PATH}")
                
        except Exception as e:
            print(f"Error in load_models: {str(e)}")
    
    def preprocess_tabular(self, tabular_data):
        """Preprocess tabular data for model prediction"""
        
        # Define feature names based on updated rockfall parameters
        feature_names = [
            # Slope Geometry Parameters
            'slope_height', 'slope_angle', 'slope_length', 'surface_roughness', 'slope_profile',
            # Rock/Block Parameters
            'block_size', 'block_shape', 'material_strength', 'restitution_coefficient', 
            'friction_coefficient',
            # Environmental & Triggering Parameters
            'moisture_content', 'freeze_thaw_cycles', 'vegetation', 'temperature_changes',
            'seismic_events',
            # Initial Conditions
            'release_point', 'initial_velocity',
            # Simulation & Output Parameters
            'runout_distance', 'impact_velocity', 'kinetic_energy', 'bounce_height',
            'lateral_dispersion', 'number_of_bounces',
            # Historical & Statistical Data
            'rockfall_frequency', 'block_volume_distribution'
        ]
        
        if tabular_data is None:
            # Return default features if no tabular data provided
            return np.zeros((1, len(feature_names)))
        
        # Handle dictionary input
        if isinstance(tabular_data, dict):
            # Convert dictionary to array in the correct order
            values = []
            for feature in feature_names:
                values.append(tabular_data.get(feature, 0.5))
            tabular_data = np.array(values)
        
        # Convert to numpy array if needed
        if not isinstance(tabular_data, np.ndarray):
            tabular_data = np.array(tabular_data)
        
        # Ensure correct shape
        if len(tabular_data.shape) == 1:
            tabular_data = tabular_data.reshape(1, -1)
        
        # Scale data if scaler is available
        if self.scaler is not None:
            try:
                tabular_data = self.scaler.transform(tabular_data)
            except Exception as e:
                print(f"Error scaling data: {str(e)}")
                # Return default scaled features
                return np.zeros((1, len(feature_names)))
        
        # Apply feature selection if available
        feature_selector_path = os.path.join(MODELS_DIR, "feature_selector.pkl")
        if os.path.exists(feature_selector_path):
            try:
                feature_selector = joblib.load(feature_selector_path)
                tabular_data = feature_selector.transform(tabular_data)
            except Exception as e:
                print(f"Error applying feature selection: {str(e)}")
                return np.zeros((1, tabular_data.shape[1]))
        
        return tabular_data
    
    def predict_knn(self, tabular_data):
        """Make prediction using KNN model"""
        
        if self.knn_model is None:
            # Return default probabilities if model not available
            return np.array([0.8, 0.15, 0.05])  # Conservative default: high probability of low risk
        
        try:
            # Preprocess tabular data
            processed_data = self.preprocess_tabular(tabular_data)
            
            # Make prediction
            knn_probs = self.knn_model.predict_proba(processed_data)
            
            return knn_probs[0]  # Return single prediction
        except Exception as e:
            print(f"Error in KNN prediction: {str(e)}")
            return np.array([0.8, 0.15, 0.05])
    
    def predict_rf(self, tabular_data):
        """Make prediction using Random Forest model"""
        
        if self.rf_model is None:
            # Return default probabilities if model not available
            return np.array([0.8, 0.15, 0.05])
        
        try:
            # Preprocess tabular data
            processed_data = self.preprocess_tabular(tabular_data)
            
            # Make prediction
            rf_probs = self.rf_model.predict_proba(processed_data)
            
            return rf_probs[0]  # Return single prediction
        except Exception as e:
            print(f"Error in RF prediction: {str(e)}")
            return np.array([0.8, 0.15, 0.05])
    
    def predict_ensemble(self, tabular_data, knn_weight=0.6, rf_weight=0.4):
        """Make ensemble prediction combining KNN and Random Forest"""
        
        # Get individual predictions
        knn_probs = self.predict_knn(tabular_data)
        rf_probs = self.predict_rf(tabular_data)
        
        # Combine predictions using weighted average
        ensemble_probs = knn_weight * knn_probs + rf_weight * rf_probs
        
        return ensemble_probs, knn_probs, rf_probs
    
    def predict(self, tabular_data=None, return_details=False):
        """Make final rockfall prediction with risk level and probability"""
        
        try:
            # Get ensemble prediction
            ensemble_probs, knn_probs, rf_probs = self.predict_ensemble(tabular_data)
            
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
                    'knn_probabilities': knn_probs,
                    'rf_probabilities': rf_probs,
                    'predicted_class_idx': predicted_class_idx,
                    'individual_scores': {
                        'knn_max_prob': np.max(knn_probs),
                        'rf_max_prob': np.max(rf_probs)
                    }
                }
                return details
            
            return probability_score, predicted_risk
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            # Return default values in case of error
            return 0.1, 'Low'  # Conservative default
    
    def batch_predict(self, tabular_data_list):
        """Make predictions for multiple samples"""
        
        predictions = []
        
        for i, tabular_data in enumerate(tabular_data_list):
            try:
                probability, risk_level = self.predict(tabular_data)
                predictions.append({
                    'sample_id': i,
                    'probability': probability,
                    'risk_level': risk_level
                })
            except Exception as e:
                print(f"Error predicting sample {i}: {str(e)}")
                predictions.append({
                    'sample_id': i,
                    'probability': 0.1,
                    'risk_level': 'Low'  # Conservative default
                })
        
        return predictions
    
    def get_risk_threshold_classification(self, probability):
        """Convert probability score to risk threshold classification"""
        
        if probability < config.LOW_RISK_THRESHOLD:
            return "Low Risk"
        elif probability < config.MEDIUM_RISK_THRESHOLD:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def analyze_prediction_confidence(self, prediction_details):
        """Analyze confidence of prediction based on model agreement"""
        
        # Extract probabilities
        knn_probs = prediction_details['knn_probabilities']
        rf_probs = prediction_details['rf_probabilities']
        ensemble_probs = prediction_details['ensemble_probabilities']
        
        # Get predicted classes for each model
        knn_class = np.argmax(knn_probs)
        rf_class = np.argmax(rf_probs)
        ensemble_class = np.argmax(ensemble_probs)
        
        # Check if models agree on prediction
        models_agree = (knn_class == rf_class)
        
        # Calculate confidence metrics
        confidence_metrics = {
            'models_agree': models_agree,
            'knn_confidence': np.max(knn_probs),
            'rf_confidence': np.max(rf_probs),
            'ensemble_confidence': np.max(ensemble_probs),
            'confidence_gap': np.max(ensemble_probs) - np.sort(ensemble_probs)[-2]
        }
        
        # Generate confidence assessment
        if models_agree and confidence_metrics['ensemble_confidence'] > 0.8:
            confidence_assessment = "High confidence prediction"
        elif models_agree and confidence_metrics['ensemble_confidence'] > 0.6:
            confidence_assessment = "Moderate confidence prediction"
        elif not models_agree and confidence_metrics['ensemble_confidence'] > 0.7:
            confidence_assessment = "Models disagree but ensemble is confident"
        else:
            confidence_assessment = "Low confidence prediction"
        
        confidence_metrics['assessment'] = confidence_assessment
        return confidence_metrics
    
    def get_feature_importance_explanation(self, tabular_data):
        """Generate explanation of prediction based on feature importance"""
        
        if self.rf_model is None or not hasattr(self.rf_model, 'feature_importances_'):
            return "Feature importance analysis not available"
        
        try:
            # Preprocess tabular data
            processed_data = self.preprocess_tabular(tabular_data)
            
            # Get feature names
            feature_names = [
                'slope_height', 'slope_angle', 'slope_length', 'surface_roughness', 'slope_profile',
                'block_size', 'block_shape', 'material_strength', 'restitution_coefficient', 'friction_coefficient',
                'moisture_content', 'freeze_thaw_cycles', 'vegetation', 'temperature_changes', 'seismic_events',
                'release_point', 'initial_velocity', 'runout_distance', 'impact_velocity', 'kinetic_energy',
                'bounce_height', 'lateral_dispersion', 'number_of_bounces', 'rockfall_frequency', 'block_volume_distribution'
            ]
            
            # Get feature importances from RF model
            importances = self.rf_model.feature_importances_
            
            # Create dictionary of feature importances
            feature_importances = dict(zip(feature_names, importances))
            
            # Sort features by importance
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
            
            # Get top 5 most important features
            top_features = sorted_features[:5]
            
            # Generate explanation
            explanation = "Top factors influencing this prediction:\n"
            for feature, importance in top_features:
                # Format feature name for display
                display_name = ' '.join(word.capitalize() for word in feature.split('_'))
                
                # Get feature value
                feature_idx = feature_names.index(feature)
                if feature_idx < len(processed_data[0]):
                    feature_value = processed_data[0][feature_idx]
                    
                    # Determine if value is high, medium, or low (simplified)
                    value_assessment = "high" if feature_value > 0.7 else "medium" if feature_value > 0.3 else "low"
                    
                    explanation += f"- {display_name}: {value_assessment} (importance: {importance:.2f})\n"
                else:
                    explanation += f"- {display_name}: (importance: {importance:.2f})\n"
            
            return explanation
            
        except Exception as e:
            print(f"Error generating feature importance explanation: {str(e)}")
            return "Feature importance analysis failed"

def test_predictor():
    """Test the predictor with sample data"""
    
    try:
        predictor = RockfallPredictor()
        
        # Create sample data with all rockfall parameters
        sample_tabular = {
            # Slope Geometry Parameters
            'slope_height': 0.7,           # High slope
            'slope_angle': 0.8,            # Steep slope
            'slope_length': 0.6,           # Moderate length
            'surface_roughness': 0.3,      # Relatively smooth
            'slope_profile': 0.5,          # Mixed profile
            
            # Rock/Block Parameters
            'block_size': 0.7,             # Large block
            'block_shape': 0.2,            # More rounded
            'material_strength': 0.6,      # Moderate strength
            'restitution_coefficient': 0.7, # High bounce
            'friction_coefficient': 0.4,    # Moderate friction
            
            # Environmental & Triggering Parameters
            'moisture_content': 0.8,       # High moisture
            'freeze_thaw_cycles': 0.5,     # Moderate freeze-thaw
            'vegetation': 0.2,             # Low vegetation
            'temperature_changes': 0.6,    # Moderate temperature changes
            'seismic_events': 0.1,         # Low seismic activity
            
            # Initial Conditions
            'release_point': 0.9,          # High release point
            'initial_velocity': 0.3,       # Low initial velocity
            
            # Simulation & Output Parameters
            'runout_distance': 0.7,        # Long runout
            'impact_velocity': 0.8,        # High impact velocity
            'kinetic_energy': 0.9,         # High energy
            'bounce_height': 0.7,          # High bounce
            'lateral_dispersion': 0.5,     # Moderate dispersion
            'number_of_bounces': 0.6,      # Moderate number of bounces
            
            # Historical & Statistical Data
            'rockfall_frequency': 0.7,     # High frequency
            'block_volume_distribution': 0.6 # Moderate volume distribution
        }
        
        # Make prediction
        probability, risk_level = predictor.predict(sample_tabular)
        
        print(f"Sample prediction:")
        print(f"Probability: {probability:.4f}")
        print(f"Risk Level: {risk_level}")
        
        # Get detailed prediction
        details = predictor.predict(sample_tabular, return_details=True)
        print(f"Detailed prediction: {details}")
        
        # Get feature importance explanation
        explanation = predictor.get_feature_importance_explanation(sample_tabular)
        print(f"Feature importance explanation:")
        print(explanation)
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the predictor
    import traceback
    test_predictor()
