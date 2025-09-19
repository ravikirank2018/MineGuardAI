"""
Configuration file for AI-based Rockfall Prediction System
Contains all system constants, file paths, and configuration parameters
"""

import os

# ============================================================================
# SYSTEM CONSTANTS
# ============================================================================

# Dataset configuration
DATASET_SIZE = 5000  # Number of synthetic samples to generate
IMAGE_SIZE = (64, 64)  # Size of DEM images (height, width)
NUM_CLASSES = 3  # Low, Medium, High risk levels

# Risk thresholds for classification
LOW_RISK_THRESHOLD = 0.0
MEDIUM_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.7

# Model training parameters
CNN_EPOCHS = 20
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.001
RF_N_ESTIMATORS = 100
TEST_SPLIT_RATIO = 0.2

# Ensemble weights
CNN_WEIGHT = 0.6
RF_WEIGHT = 0.4

# Live monitoring settings
LIVE_UPDATE_INTERVAL = 10  # seconds
MAX_PREDICTION_HISTORY = 100

# ============================================================================
# FILE PATHS
# ============================================================================

# Data directories
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data file paths
IMAGE_DATA_PATH = os.path.join(DATA_DIR, "dem_images.npy")
TABULAR_DATA_PATH = os.path.join(DATA_DIR, "tabular_features.csv")
WEATHER_DATA_PATH = os.path.join(DATA_DIR, "weather_data.csv")
GEOLOGICAL_DATA_PATH = os.path.join(DATA_DIR, "geological_survey.csv")

# Model file paths
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_model.h5")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Log file paths
PREDICTION_LOG_PATH = os.path.join(LOGS_DIR, "predictions.log")
SYSTEM_LOG_PATH = os.path.join(LOGS_DIR, "system.log")
TRAINING_LOG_PATH = os.path.join(LOGS_DIR, "training.log")

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Tabular feature names in order
FEATURE_NAMES = [
    'temperature',          # Temperature in Celsius
    'humidity',             # Relative humidity percentage
    'rainfall',             # Rainfall in mm (24-hour)
    'wind_speed',           # Wind speed in km/h
    'soil_moisture',        # Soil moisture percentage
    'soil_density',         # Soil density in g/cm³
    'cohesion',             # Soil cohesion in kPa
    'friction_angle',       # Internal friction angle in degrees
    'slope_angle',          # Slope angle in degrees
    'fracture_density',     # Fracture density (0-1)
    'rock_quality',         # Rock quality index (0-1)
    'vibration_level',      # Ground vibration level
    'groundwater_level'     # Groundwater level in meters
]

# Feature ranges for validation
FEATURE_RANGES = {
    'temperature': (-20, 50),
    'humidity': (0, 100),
    'rainfall': (0, 300),
    'wind_speed': (0, 100),
    'soil_moisture': (0, 50),
    'soil_density': (1.0, 2.5),
    'cohesion': (0, 50),
    'friction_angle': (10, 50),
    'slope_angle': (0, 90),
    'fracture_density': (0, 1),
    'rock_quality': (0, 1),
    'vibration_level': (0, 10),
    'groundwater_level': (-30, 10)
}

# ============================================================================
# RISK LEVEL DEFINITIONS
# ============================================================================

RISK_LEVELS = ['Low', 'Medium', 'High']

# Risk level descriptions
RISK_DESCRIPTIONS = {
    'Low': {
        'description': 'Minimal rockfall risk - normal operations can continue',
        'color': '#28a745',  # Green
        'actions': [
            'Continue normal mining operations',
            'Maintain standard monitoring procedures',
            'Conduct routine safety inspections',
            'Monitor trends and environmental changes'
        ]
    },
    'Medium': {
        'description': 'Moderate rockfall risk - increased caution required',
        'color': '#ffc107',  # Yellow/Orange
        'actions': [
            'Reduce personnel in potentially affected areas',
            'Increase visual inspections and patrols',
            'Enhance sensor monitoring and data collection',
            'Review safety protocols and evacuation routes',
            'Brief all personnel on current risk status'
        ]
    },
    'High': {
        'description': 'High rockfall risk - immediate safety measures required',
        'color': '#dc3545',  # Red
        'actions': [
            'Evacuate personnel from high-risk areas immediately',
            'Alert emergency response team and mine safety officer',
            'Restrict access to unstable slopes and danger zones',
            'Increase monitoring frequency to continuous mode',
            'Document all actions taken for safety compliance'
        ]
    }
}

# ============================================================================
# SYNTHETIC DATA GENERATION PARAMETERS
# ============================================================================

# Risk distribution for balanced dataset
RISK_DISTRIBUTION = {
    'Low': 0.7,      # 70% low risk samples
    'Medium': 0.2,   # 20% medium risk samples
    'High': 0.1      # 10% high risk samples
}

# Weather parameter ranges by risk level
WEATHER_RANGES = {
    'Low': {
        'temperature': (15, 23, 3),  # (mean, range, std)
        'humidity': (30, 60, 10),
        'rainfall': (0, 20, 5),
        'wind_speed': (0, 15, 5)
    },
    'Medium': {
        'temperature': (18, 25, 5),
        'humidity': (50, 80, 15),
        'rainfall': (10, 60, 20),
        'wind_speed': (5, 25, 8)
    },
    'High': {
        'temperature': (22, 28, 10),
        'humidity': (70, 95, 10),
        'rainfall': (50, 200, 50),
        'wind_speed': (20, 60, 15)
    }
}

# Soil parameter ranges by risk level
SOIL_RANGES = {
    'Low': {
        'soil_moisture': (5, 20, 5),
        'soil_density': (1.6, 2.2, 0.2),
        'cohesion': (20, 40, 8),
        'friction_angle': (30, 45, 5)
    },
    'Medium': {
        'soil_moisture': (15, 30, 5),
        'soil_density': (1.4, 1.8, 0.2),
        'cohesion': (10, 25, 5),
        'friction_angle': (20, 35, 5)
    },
    'High': {
        'soil_moisture': (25, 40, 5),
        'soil_density': (1.2, 1.6, 0.2),
        'cohesion': (5, 15, 5),
        'friction_angle': (15, 25, 5)
    }
}

# Geological parameter ranges by risk level
GEOLOGICAL_RANGES = {
    'Low': {
        'slope_angle': (5, 30, 8),
        'fracture_density': (0.1, 0.4, 0.1),
        'rock_quality': (0.6, 1.0, 0.15),
        'vibration_level': (0, 2, 0.5),
        'groundwater_level': (-20, -5, 5)
    },
    'Medium': {
        'slope_angle': (25, 50, 8),
        'fracture_density': (0.3, 0.7, 0.15),
        'rock_quality': (0.3, 0.7, 0.15),
        'vibration_level': (0, 3, 1),
        'groundwater_level': (-15, 0, 5)
    },
    'High': {
        'slope_angle': (45, 75, 10),
        'fracture_density': (0.6, 1.0, 0.15),
        'rock_quality': (0.1, 0.4, 0.1),
        'vibration_level': (2, 5, 1),
        'groundwater_level': (-10, 5, 5)
    }
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Color schemes
COLOR_SCHEMES = {
    'risk_heatmap': 'RdYlGn_r',  # Red-Yellow-Green reversed
    'feature_importance': 'Viridis',
    'confusion_matrix': 'Blues',
    'time_series': ['#ff6b6b', '#4ecdc4', '#45b7d1']
}

# Plot dimensions
PLOT_DIMENSIONS = {
    'heatmap_size': (20, 20),
    'figure_height': 400,
    'figure_width': 800,
    'dashboard_height': 600
}

# Heatmap grid settings
HEATMAP_GRID = {
    'x_min': 0,
    'x_max': 1000,  # meters
    'y_min': 0,
    'y_max': 1000,  # meters
    'resolution': 50  # grid points per axis
}

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================

# CNN architecture
CNN_ARCHITECTURE = {
    'conv_layers': [
        {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
        {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'}
    ],
    'dense_layers': [
        {'units': 512, 'activation': 'relu', 'dropout': 0.5},
        {'units': 256, 'activation': 'relu', 'dropout': 0.3}
    ],
    'batch_normalization': True,
    'max_pooling': True
}

# Random Forest parameters
RF_PARAMETERS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================================
# ALERT SYSTEM CONFIGURATION
# ============================================================================

# Alert thresholds (can be overridden in UI)
ALERT_THRESHOLDS = {
    'high_risk': HIGH_RISK_THRESHOLD,
    'medium_risk': MEDIUM_RISK_THRESHOLD,
    'consecutive_high_risk': 3,  # Number of consecutive high-risk predictions
    'probability_spike': 0.2     # Minimum increase to trigger spike alert
}

# Alert message templates
ALERT_MESSAGES = {
    'high_risk': "🚨 HIGH RISK ALERT: Rockfall probability is {probability:.1%}. Immediate evacuation recommended.",
    'medium_risk': "⚠️ MEDIUM RISK WARNING: Rockfall probability is {probability:.1%}. Increase monitoring and reduce personnel.",
    'low_risk': "✅ LOW RISK: Current rockfall probability is {probability:.1%}. Normal operations can continue.",
    'spike_alert': "📈 PROBABILITY SPIKE: Risk increased by {increase:.1%} in the last update.",
    'system_error': "❌ SYSTEM ERROR: Unable to generate prediction. Check system status."
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Log levels
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# MINE SITE CONFIGURATION
# ============================================================================

# Mine site dimensions and coordinates
MINE_SITE = {
    'name': 'Demo Open-Pit Mine',
    'center_lat': 45.0,
    'center_lon': -110.0,
    'area_km2': 2.5,
    'max_depth': 150,  # meters
    'operational_levels': 8,
    'bench_height': 15,  # meters
    'safety_zones': [
        {'name': 'Equipment Staging', 'risk_multiplier': 0.5},
        {'name': 'Active Mining', 'risk_multiplier': 1.2},
        {'name': 'Waste Dump', 'risk_multiplier': 0.8},
        {'name': 'Access Roads', 'risk_multiplier': 0.7}
    ]
}

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'prediction_time_max': 5.0,    # seconds
    'model_accuracy_min': 0.85,    # minimum acceptable accuracy
    'data_freshness_max': 300,     # maximum data age in seconds
    'memory_usage_max': 1000,      # MB
    'cpu_usage_max': 80            # percentage
}

# Metrics to track
METRICS_TO_TRACK = [
    'prediction_accuracy',
    'prediction_time',
    'false_positive_rate',
    'false_negative_rate',
    'system_uptime',
    'data_quality_score'
]

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

# Optional environment variable overrides
def get_env_config():
    """Get configuration from environment variables with fallbacks"""
    
    config_overrides = {}
    
    # Dataset size override
    if 'DATASET_SIZE' in os.environ:
        try:
            config_overrides['DATASET_SIZE'] = int(os.environ['DATASET_SIZE'])
        except ValueError:
            pass
    
    # Risk thresholds override
    if 'HIGH_RISK_THRESHOLD' in os.environ:
        try:
            config_overrides['HIGH_RISK_THRESHOLD'] = float(os.environ['HIGH_RISK_THRESHOLD'])
        except ValueError:
            pass
    
    if 'MEDIUM_RISK_THRESHOLD' in os.environ:
        try:
            config_overrides['MEDIUM_RISK_THRESHOLD'] = float(os.environ['MEDIUM_RISK_THRESHOLD'])
        except ValueError:
            pass
    
    # Update interval override
    if 'LIVE_UPDATE_INTERVAL' in os.environ:
        try:
            config_overrides['LIVE_UPDATE_INTERVAL'] = int(os.environ['LIVE_UPDATE_INTERVAL'])
        except ValueError:
            pass
    
    return config_overrides

# Apply environment overrides
ENV_CONFIG = get_env_config()
for key, value in ENV_CONFIG.items():
    if key in globals():
        globals()[key] = value

# ============================================================================
# VERSION INFORMATION
# ============================================================================

VERSION = "1.0.0"
BUILD_DATE = "2024-12-19"
AUTHOR = "AI Rockfall Prediction System"
DESCRIPTION = "AI-based rockfall prediction and alert system for open-pit mine safety"

# System information
SYSTEM_INFO = {
    'version': VERSION,
    'build_date': BUILD_DATE,
    'author': AUTHOR,
    'description': DESCRIPTION,
    'python_requirements': [
        'streamlit>=1.28.0',
        'tensorflow>=2.12.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'plotly>=5.15.0',
        'opencv-python>=4.8.0',
        'Pillow>=10.0.0',
        'joblib>=1.3.0',
        'faker>=19.0.0'
    ]
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config():
    """Validate configuration parameters"""
    
    errors = []
    
    # Validate risk thresholds
    if not (0 <= LOW_RISK_THRESHOLD <= MEDIUM_RISK_THRESHOLD <= HIGH_RISK_THRESHOLD <= 1):
        errors.append("Risk thresholds must be in ascending order between 0 and 1")
    
    # Validate dataset size
    if DATASET_SIZE < 100:
        errors.append("Dataset size must be at least 100")
    
    # Validate image size
    if not all(dim > 0 for dim in IMAGE_SIZE):
        errors.append("Image size dimensions must be positive")
    
    # Validate ensemble weights
    if abs(CNN_WEIGHT + RF_WEIGHT - 1.0) > 0.001:
        errors.append("CNN and RF weights must sum to 1.0")
    
    # Validate feature ranges
    for feature, (min_val, max_val) in FEATURE_RANGES.items():
        if min_val >= max_val:
            errors.append(f"Feature range for {feature} is invalid: min >= max")
    
    return errors

def print_config_summary():
    """Print configuration summary for debugging"""
    
    print("=" * 60)
    print("AI ROCKFALL PREDICTION SYSTEM - CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Version: {VERSION}")
    print(f"Dataset Size: {DATASET_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Risk Thresholds: Low < {MEDIUM_RISK_THRESHOLD}, Medium < {HIGH_RISK_THRESHOLD}")
    print(f"Model Ensemble: CNN({CNN_WEIGHT}) + RF({RF_WEIGHT})")
    print(f"Update Interval: {LIVE_UPDATE_INTERVAL}s")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print("=" * 60)
    
    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        print("CONFIGURATION ERRORS:")
        for error in config_errors:
            print(f"  - {error}")
        print("=" * 60)
    else:
        print("Configuration validated successfully!")
        print("=" * 60)

if __name__ == "__main__":
    print_config_summary()
