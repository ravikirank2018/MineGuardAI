import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from config import *

def create_risk_heatmap(size=(20, 20), seed=None):
    """Create a synthetic risk heatmap for the mine site"""
    
    if seed:
        np.random.seed(seed)
    
    # Create base risk map
    heatmap = np.random.rand(*size) * 0.3  # Base low risk
    
    # Add some high-risk zones (geological instabilities)
    num_high_risk_zones = random.randint(2, 5)
    
    for _ in range(num_high_risk_zones):
        # Random high-risk zone center
        center_x = random.randint(2, size[0] - 3)
        center_y = random.randint(2, size[1] - 3)
        
        # Create circular high-risk zone
        radius = random.randint(2, 4)
        y, x = np.ogrid[:size[0], :size[1]]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Add risk intensity
        risk_intensity = random.uniform(0.6, 0.9)
        heatmap[mask] = np.maximum(heatmap[mask], risk_intensity)
    
    # Add some medium-risk corridors (water drainage, access roads)
    num_corridors = random.randint(1, 3)
    
    for _ in range(num_corridors):
        if random.choice([True, False]):  # Horizontal corridor
            row = random.randint(1, size[0] - 2)
            start_col = random.randint(0, size[1] // 2)
            end_col = random.randint(size[1] // 2, size[1] - 1)
            heatmap[row, start_col:end_col] = np.maximum(
                heatmap[row, start_col:end_col], 
                random.uniform(0.4, 0.6)
            )
        else:  # Vertical corridor
            col = random.randint(1, size[1] - 2)
            start_row = random.randint(0, size[0] // 2)
            end_row = random.randint(size[0] // 2, size[0] - 1)
            heatmap[start_row:end_row, col] = np.maximum(
                heatmap[start_row:end_row, col], 
                random.uniform(0.4, 0.6)
            )
    
    # Smooth the heatmap
    heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
    
    # Ensure values are in [0, 1] range
    heatmap = np.clip(heatmap, 0, 1)
    
    return heatmap

def get_risk_color(risk_level):
    """Get color code for risk level"""
    
    color_map = {
        'Low': '#28a745',     # Green
        'Medium': '#ffc107',  # Yellow/Orange
        'High': '#dc3545',    # Red
        'Unknown': '#6c757d'  # Gray
    }
    
    return color_map.get(risk_level, '#6c757d')

def format_probability(probability):
    """Format probability as percentage"""
    
    if probability is None:
        return "N/A"
    
    percentage = probability * 100
    return f"{percentage:.1f}%"

def classify_risk_by_probability(probability):
    """Classify risk level based on probability"""
    
    if probability >= HIGH_RISK_THRESHOLD:
        return 'High'
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return 'Medium'
    else:
        return 'Low'

def generate_weather_impact_score(weather_data):
    """Calculate weather impact score on rockfall risk"""
    
    score = 0.0
    
    # Temperature impact (extreme temperatures increase risk)
    temp = weather_data.get('temperature', 20)
    if temp > 35 or temp < -5:
        score += 0.2
    elif temp > 30 or temp < 0:
        score += 0.1
    
    # Rainfall impact (heavy rain increases risk)
    rainfall = weather_data.get('rainfall', 0)
    if rainfall > 100:
        score += 0.4
    elif rainfall > 50:
        score += 0.2
    elif rainfall > 20:
        score += 0.1
    
    # Wind impact (strong winds increase risk)
    wind_speed = weather_data.get('wind_speed', 0)
    if wind_speed > 50:
        score += 0.3
    elif wind_speed > 25:
        score += 0.1
    
    # Humidity impact (high humidity can affect soil stability)
    humidity = weather_data.get('humidity', 50)
    if humidity > 85:
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0

def generate_geological_stability_score(geological_data):
    """Calculate geological stability score"""
    
    score = 1.0  # Start with stable (1.0 = stable, 0.0 = unstable)
    
    # Slope angle impact
    slope_angle = geological_data.get('slope_angle', 30)
    if slope_angle > 60:
        score -= 0.4
    elif slope_angle > 45:
        score -= 0.2
    elif slope_angle > 35:
        score -= 0.1
    
    # Fracture density impact
    fracture_density = geological_data.get('fracture_density', 0.3)
    score -= fracture_density * 0.3
    
    # Rock quality impact
    rock_quality = geological_data.get('rock_quality', 0.7)
    score -= (1.0 - rock_quality) * 0.3
    
    # Soil cohesion impact
    cohesion = geological_data.get('cohesion', 25)
    if cohesion < 10:
        score -= 0.2
    elif cohesion < 15:
        score -= 0.1
    
    return max(score, 0.0)  # Cap at 0.0 minimum

def create_feature_importance_plot(feature_names, importance_values):
    """Create feature importance visualization"""
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=True)
    
    # Create horizontal bar plot
    fig = px.bar(
        df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Rockfall Prediction',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        xaxis_title="Importance Score",
        yaxis_title="Features"
    )
    
    return fig

def create_confusion_matrix_plot(y_true, y_pred, class_names):
    """Create confusion matrix visualization"""
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    fig = px.imshow(
        cm_normalized,
        text_auto=True,
        color_continuous_scale='Blues',
        title='Confusion Matrix (Normalized)',
        x=class_names,
        y=class_names
    )
    
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=400
    )
    
    return fig

def calculate_risk_metrics(predictions, true_labels):
    """Calculate various risk prediction metrics"""
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions, average='weighted'),
        'recall': recall_score(true_labels, predictions, average='weighted'),
        'f1_score': f1_score(true_labels, predictions, average='weighted')
    }
    
    # Calculate per-class metrics
    for class_name in np.unique(true_labels):
        class_mask = (true_labels == class_name)
        class_predictions = predictions[class_mask]
        class_true = true_labels[class_mask]
        
        if len(class_true) > 0:
            class_accuracy = accuracy_score(class_true, class_predictions)
            metrics[f'{class_name}_accuracy'] = class_accuracy
    
    return metrics

def create_time_series_plot(timestamps, values, title="Time Series", ylabel="Value"):
    """Create time series plot"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines+markers',
        name=ylabel,
        line=dict(width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=ylabel,
        hovermode='x unified'
    )
    
    return fig

def generate_synthetic_mine_coordinates():
    """Generate synthetic mine pit coordinates"""
    
    # Create a typical open-pit mine shape
    angles = np.linspace(0, 2*np.pi, 100)
    
    # Outer perimeter (elliptical)
    outer_a, outer_b = 500, 300  # Semi-major and semi-minor axes in meters
    outer_x = outer_a * np.cos(angles)
    outer_y = outer_b * np.sin(angles)
    
    # Inner pit (smaller ellipse)
    inner_a, inner_b = 200, 120
    inner_x = inner_a * np.cos(angles)
    inner_y = inner_b * np.sin(angles)
    
    # Benches (stepped levels)
    bench_levels = []
    for level in range(1, 6):  # 5 bench levels
        scale = 0.8 ** level
        bench_a = outer_a * scale
        bench_b = outer_b * scale
        bench_x = bench_a * np.cos(angles)
        bench_y = bench_b * np.sin(angles)
        bench_levels.append((bench_x, bench_y))
    
    return {
        'outer_perimeter': (outer_x, outer_y),
        'inner_pit': (inner_x, inner_y),
        'bench_levels': bench_levels
    }

def create_mine_site_visualization(risk_data):
    """Create 3D visualization of mine site with risk overlay"""
    
    # Generate mine coordinates
    mine_coords = generate_synthetic_mine_coordinates()
    
    # Create 3D surface plot
    fig = go.Figure()
    
    # Add risk heatmap as surface
    x_range = np.linspace(-600, 600, risk_data.shape[1])
    y_range = np.linspace(-400, 400, risk_data.shape[0])
    X, Y = np.meshgrid(x_range, y_range)
    
    fig.add_trace(go.Surface(
        x=X, y=Y, z=risk_data * 100,  # Scale for visualization
        colorscale='RdYlGn_r',
        name='Risk Level',
        showscale=True,
        colorbar=dict(title="Risk Score")
    ))
    
    fig.update_layout(
        title='3D Mine Site Risk Visualization',
        scene=dict(
            xaxis_title='East-West (m)',
            yaxis_title='North-South (m)',
            zaxis_title='Risk Score',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    return fig

def export_risk_report(prediction_data, filename=None):
    """Export comprehensive risk report"""
    
    if filename is None:
        filename = f"rockfall_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    # Convert prediction data to DataFrame
    df = pd.DataFrame(prediction_data)
    
    # Generate summary statistics
    summary_stats = {
        'total_predictions': len(df),
        'high_risk_count': (df['risk_level'] == 'High').sum(),
        'medium_risk_count': (df['risk_level'] == 'Medium').sum(),
        'low_risk_count': (df['risk_level'] == 'Low').sum(),
        'average_probability': df['probability'].mean(),
        'max_probability': df['probability'].max(),
        'latest_risk_level': df['risk_level'].iloc[-1] if not df.empty else 'Unknown'
    }
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rockfall Risk Assessment Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 10px; }}
            .summary {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                      background-color: #e9ecef; border-radius: 5px; }}
            .high-risk {{ color: #dc3545; font-weight: bold; }}
            .medium-risk {{ color: #ffc107; font-weight: bold; }}
            .low-risk {{ color: #28a745; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏔️ Rockfall Risk Assessment Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Total Assessments:</strong> {summary_stats['total_predictions']}
            </div>
            <div class="metric high-risk">
                <strong>High Risk Events:</strong> {summary_stats['high_risk_count']}
            </div>
            <div class="metric medium-risk">
                <strong>Medium Risk Events:</strong> {summary_stats['medium_risk_count']}
            </div>
            <div class="metric low-risk">
                <strong>Low Risk Events:</strong> {summary_stats['low_risk_count']}
            </div>
            <div class="metric">
                <strong>Average Risk Probability:</strong> {summary_stats['average_probability']:.1%}
            </div>
            <div class="metric">
                <strong>Peak Risk Probability:</strong> {summary_stats['max_probability']:.1%}
            </div>
        </div>
        
        <h2>Current Status</h2>
        <p><strong>Latest Risk Level:</strong> 
           <span class="{summary_stats['latest_risk_level'].lower()}-risk">
               {summary_stats['latest_risk_level']}
           </span>
        </p>
        
        <h2>Recommendations</h2>
        <ul>
            <li>Continue monitoring high-risk areas with increased frequency</li>
            <li>Ensure all safety protocols are up to date and practiced</li>
            <li>Review evacuation routes and emergency procedures</li>
            <li>Consider additional geological surveys in high-risk zones</li>
        </ul>
        
        <footer style="margin-top: 40px; font-size: 12px; color: #6c757d;">
            <p>This report was generated by the AI-based Rockfall Prediction System</p>
        </footer>
    </body>
    </html>
    """
    
    return html_content, summary_stats

def validate_model_input(image, tabular_data):
    """Validate input data for model prediction"""
    
    errors = []
    
    # Validate image
    if image is None:
        errors.append("Image data is required")
    else:
        if not isinstance(image, np.ndarray):
            errors.append("Image must be a numpy array")
        elif len(image.shape) != 3:
            errors.append("Image must be 3-dimensional (height, width, channels)")
        elif image.shape[2] != 3:
            errors.append("Image must have 3 channels (RGB)")
    
    # Validate tabular data
    if tabular_data is None:
        errors.append("Tabular data is required")
    else:
        if not isinstance(tabular_data, (list, np.ndarray)):
            errors.append("Tabular data must be a list or numpy array")
        elif len(tabular_data) != 13:  # Expected number of features
            errors.append(f"Tabular data must have exactly 13 features, got {len(tabular_data)}")
    
    return errors

# Utility functions for data preprocessing
def normalize_image(image):
    """Normalize image to [0, 1] range"""
    
    if image.max() > 1.0:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)

def denormalize_image(image):
    """Convert normalized image back to [0, 255] range"""
    
    if image.max() <= 1.0:
        return (image * 255).astype(np.uint8)
    return image.astype(np.uint8)

# Constants for visualization
RISK_COLORMAP = {
    0.0: '#00ff00',  # Green (no risk)
    0.3: '#ffff00',  # Yellow (low risk)
    0.6: '#ff8000',  # Orange (medium risk)
    1.0: '#ff0000'   # Red (high risk)
}

def get_risk_color_from_probability(probability):
    """Get color based on continuous probability value"""
    
    if probability <= 0.3:
        # Interpolate between green and yellow
        ratio = probability / 0.3
        return f"rgb({int(255*ratio)}, 255, 0)"
    elif probability <= 0.6:
        # Interpolate between yellow and orange
        ratio = (probability - 0.3) / 0.3
        return f"rgb(255, {int(255*(1-ratio*0.5))}, 0)"
    else:
        # Interpolate between orange and red
        ratio = (probability - 0.6) / 0.4
        return f"rgb(255, {int(128*(1-ratio))}, 0)"

if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test risk heatmap generation
    heatmap = create_risk_heatmap()
    print(f"Risk heatmap shape: {heatmap.shape}")
    print(f"Risk range: {heatmap.min():.3f} - {heatmap.max():.3f}")
    
    # Test weather impact calculation
    weather_data = {
        'temperature': 35,
        'rainfall': 75,
        'wind_speed': 30,
        'humidity': 90
    }
    weather_score = generate_weather_impact_score(weather_data)
    print(f"Weather impact score: {weather_score:.3f}")
    
    # Test geological stability calculation
    geo_data = {
        'slope_angle': 50,
        'fracture_density': 0.7,
        'rock_quality': 0.4,
        'cohesion': 8
    }
    geo_score = generate_geological_stability_score(geo_data)
    print(f"Geological stability score: {geo_score:.3f}")
    
    print("All utility functions tested successfully!")
