import streamlit as st
import os
import sys
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cv2
from PIL import Image
import joblib
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import RockfallDataGenerator
from model_trainer import ModelTrainer
from predictor import RockfallPredictor
from dashboard import Dashboard
from camera_page import CameraPage
from utils import create_risk_heatmap, format_probability, get_risk_color
from config import *

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = []
if 'live_mode' not in st.session_state:
    st.session_state.live_mode = False
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

def main():
    st.set_page_config(
        page_title="AI Rockfall Prediction System",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("AI-Based Rockfall Prediction & Alert System")
    st.markdown("*Advanced machine learning system for open-pit mine safety*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Cache page components to improve switching performance
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
        
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard", "Live Camera & Alerts", "Data Generation", "Model Training", "System Settings"],
        key="page_selector"
    )
    
    # Only reload components if page has changed
    if page != st.session_state.current_page:
        st.session_state.current_page = page
        
    if page == "Dashboard":
        show_dashboard()
    elif page == "Live Camera & Alerts":
        if 'camera_page' not in st.session_state:
            st.session_state.camera_page = CameraPage()
        st.session_state.camera_page.render_page()
    elif page == "Data Generation":
        show_data_generation()
    elif page == "Model Training":
        show_model_training()
    elif page == "System Settings":
        show_settings()

def show_dashboard():
    """Main dashboard view with real-time monitoring"""
    
    # Check if system is ready
    if not st.session_state.data_generated:
        st.warning("Please generate training data first in the Data Generation section.")
        return
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first in the Model Training section.")
        return
    
    # Live mode toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Real-time Rockfall Monitoring")
    with col2:
        if st.button("Refresh Data"):
            generate_live_prediction()
    with col3:
        live_mode = st.toggle("Live Mode (10s)", value=st.session_state.live_mode)
        st.session_state.live_mode = live_mode
    
    # Auto-refresh in live mode
    if st.session_state.live_mode:
        time_since_update = (datetime.now() - st.session_state.last_update).seconds
        if time_since_update >= 10:
            generate_live_prediction()
            st.rerun()
    
    # Current status overview
    show_current_status()
    
    # Risk heatmap and metrics
    show_risk_analysis()
    
    # Historical data and trends
    show_historical_trends()
    
    # Alert system
    show_alert_system()

def show_current_status():
    """Display current system status and latest predictions"""
    
    if not st.session_state.prediction_data:
        generate_live_prediction()
    
    latest_data = st.session_state.prediction_data[-1] if st.session_state.prediction_data else None
    
    if latest_data is None:
        st.error("No prediction data available")
        return
    
    # Status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        probability = latest_data['probability']
        risk_level = latest_data['risk_level']
        color = get_risk_color(risk_level)
        
        st.metric(
            label="Rockfall Probability",
            value=f"{probability:.1%}",
            delta=f"{risk_level} Risk"
        )
        st.markdown(f"<div style='background-color: {color}20; padding: 10px; border-radius: 5px; text-align: center;'><strong>{risk_level.upper()} RISK</strong></div>", unsafe_allow_html=True)
    
    with col2:
        temp = latest_data['temperature']
        humidity = latest_data['humidity']
        
        st.metric(
            label="Temperature",
            value=f"{temp:.1f}°C"
        )
        st.metric(
            label="Humidity",
            value=f"{humidity:.1f}%"
        )
    
    with col3:
        rainfall = latest_data['rainfall']
        wind_speed = latest_data['wind_speed']
        
        st.metric(
            label="Rainfall",
            value=f"{rainfall:.1f}mm"
        )
        st.metric(
            label="Wind Speed",
            value=f"{wind_speed:.1f}km/h"
        )
    
    with col4:
        soil_moisture = latest_data['soil_moisture']
        slope_angle = latest_data['slope_angle']
        
        st.metric(
            label="Soil Moisture",
            value=f"{soil_moisture:.1f}%"
        )
        st.metric(
            label="Slope Angle",
            value=f"{slope_angle:.1f}°"
        )

def show_risk_analysis():
    """Display risk heatmap and detailed analysis"""
    
    st.subheader("Risk Analysis")

    # Realistic 3D Mountain Visualizer with Dynamic Rock and Parameters
    st.subheader("3D Realistic Mountain & Rockfall Visualizer")
    import plotly.graph_objects as go
    import numpy as np

    # Get latest prediction data for dynamic parameters
    latest_data = st.session_state.prediction_data[-1] if st.session_state.prediction_data else None
    # Default values if no data
    temp = latest_data['temperature'] if latest_data else 25.0
    humidity = latest_data['humidity'] if latest_data else 50.0
    rainfall = latest_data['rainfall'] if latest_data else 0.0
    wind_speed = latest_data['wind_speed'] if latest_data else 5.0
    soil_moisture = latest_data['soil_moisture'] if latest_data else 20.0
    slope_angle = latest_data['slope_angle'] if latest_data else 30.0
    risk_level = latest_data['risk_level'] if latest_data else 'Low'
    probability = latest_data['probability'] if latest_data else 0.1

    # Generate a realistic mountain using a Gaussian hill
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    x, y = np.meshgrid(x, y)
    z = 12 * np.exp(-0.015 * (x**2 + y**2)) + 2 * np.sin(0.2 * x) * np.cos(0.2 * y)

    # Place a rock at the top, about to fall
    rock_x = 0
    rock_y = 0
    rock_z = 12  # Top of the mountain

    # Simulate a short predicted path down the slope
    path_x = np.linspace(rock_x, 12 * np.sin(np.radians(slope_angle)), 10)
    path_y = np.linspace(rock_y, 12 * np.cos(np.radians(slope_angle)), 10)
    path_z = 12 - np.linspace(0, 8, 10)  # Descending

    fig3d = go.Figure()
    # Mountain surface
    fig3d.add_trace(go.Surface(z=z, x=x, y=y, colorscale='Earth', opacity=0.85, name='Mountain'))
    # Rock (large marker)
    fig3d.add_trace(go.Scatter3d(
        x=[rock_x], y=[rock_y], z=[rock_z+0.5],
        mode='markers',
        marker=dict(size=16, color='gray', symbol='circle'),
        name='Rock (about to fall)'
    ))
    # Rockfall path
    fig3d.add_trace(go.Scatter3d(
        x=path_x, y=path_y, z=path_z,
        mode='lines+markers',
        marker=dict(size=6, color='red'),
        line=dict(color='red', width=5),
        name='Predicted Rockfall Path'
    ))
    # Annotate main parameters
    param_text = (
        f"<b>Parameters Affecting Rockfall:</b><br>"
        f"Temperature: {temp:.1f}°C<br>"
        f"Humidity: {humidity:.1f}%<br>"
        f"Rainfall: {rainfall:.1f} mm<br>"
        f"Wind Speed: {wind_speed:.1f} km/h<br>"
        f"Soil Moisture: {soil_moisture:.1f}%<br>"
        f"Slope Angle: {slope_angle:.1f}°<br>"
        f"Risk Level: {risk_level}<br>"
        f"Probability: {probability:.1%}"
    )
    # Place annotation near the rock
    fig3d.add_trace(go.Scatter3d(
        x=[rock_x+5], y=[rock_y+5], z=[rock_z+2],
        mode='text',
        text=[param_text],
        textposition='top right',
        showlegend=False
    ))
    fig3d.update_layout(
        scene=dict(
            xaxis_title='East-West (m)',
            yaxis_title='North-South (m)',
            zaxis_title='Elevation (m)',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.1))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=550,
        title="Interactive 3D Mountain, Rock, and Dynamic Parameters"
    )
    st.plotly_chart(fig3d, use_container_width=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Risk heatmap
        st.subheader("Risk Heatmap")
        heatmap_data = create_risk_heatmap()
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='RdYlGn_r',
            aspect='auto',
            title="Mine Site Risk Distribution"
        )
        fig.update_layout(
            xaxis_title="East-West (m)",
            yaxis_title="North-South (m)",
            coloraxis_colorbar_title="Risk Score"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk distribution
        if st.session_state.prediction_data:
            recent_data = st.session_state.prediction_data[-20:] if len(st.session_state.prediction_data) >= 20 else st.session_state.prediction_data
            risk_levels = [d['risk_level'] for d in recent_data]
            risk_counts = pd.Series(risk_levels).value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Recent Risk Distribution",
                color_discrete_map={
                    'Low': '#28a745',
                    'Medium': '#ffc107', 
                    'High': '#dc3545'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

def show_historical_trends():
    """Display historical trends and patterns"""
    
    st.subheader(" Historical Trends")
    
    if len(st.session_state.prediction_data) < 2:
        st.info("Generating more data points to show trends...")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(st.session_state.prediction_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time series plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Probability trend
        fig = px.line(
            df,
            x='timestamp',
            y='probability',
            title='Rockfall Probability Over Time',
            color_discrete_sequence=['#ff6b6b']
        )
        fig.update_yaxis(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Weather correlation
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Temperature & Humidity', 'Rainfall & Wind Speed'],
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['temperature'], name='Temperature (°C)', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['humidity'], name='Humidity (%)', line=dict(color='blue'), yaxis='y2'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['rainfall'], name='Rainfall (mm)', line=dict(color='cyan')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['wind_speed'], name='Wind Speed (km/h)', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(height=400, title_text="Weather Parameters")
        st.plotly_chart(fig, use_container_width=True)

def show_alert_system():
    """Display alert system and recommendations"""
    
    st.subheader("Alert System")
    
    if not st.session_state.prediction_data:
        return
    
    latest_data = st.session_state.prediction_data[-1]
    probability = latest_data['probability']
    risk_level = latest_data['risk_level']
    
    # Alert status
    if risk_level == 'High':
        st.error(f"HIGH RISK ALERT - Probability: {probability:.1%}")
        st.markdown("**Recommended Actions:**")
        st.markdown("- Evacuate personnel from high-risk areas immediately")
        st.markdown("-Alert emergency response team")
        st.markdown("- Restrict access to unstable slopes")
        st.markdown("-Increase monitoring frequency")
        
    elif risk_level == 'Medium':
        st.warning(f"MEDIUM RISK - Probability: {probability:.1%}")
        st.markdown("**Recommended Actions:**")
        st.markdown("- Reduce personnel in potentially affected areas")
        st.markdown("- Increase visual inspections")
        st.markdown("- Enhanced sensor monitoring")
        st.markdown("- Review safety protocols")
        
    else:
        st.success(f"LOW RISK - Probability: {probability:.1%}")
        st.markdown("**Current Status:**")
        st.markdown("- Normal operations can continue")
        st.markdown("- Maintain standard monitoring procedures")
        st.markdown("- Continue routine inspections")
    
    # Recent alerts log
    st.subheader("Recent Alerts Log")
    
    alert_data = []
    for data in st.session_state.prediction_data[-10:]:
        if data['risk_level'] in ['Medium', 'High']:
            alert_data.append({
                'Timestamp': data['timestamp'],
                'Risk Level': data['risk_level'],
                'Probability': f"{data['probability']:.1%}",
                'Action': get_recommended_action(data['risk_level'])
            })
    
    if alert_data:
        alert_df = pd.DataFrame(alert_data)
        st.dataframe(alert_df, use_container_width=True)
    else:
        st.info("No recent alerts - system operating normally")

def show_data_generation():
    """Data generation interface"""
    
    st.subheader("Synthetic Data Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This system generates synthetic data for training the rockfall prediction model:
        - **DEM Images**: Simulated digital elevation model images
        - **Soil Parameters**: Moisture, density, cohesion, friction angle
        - **Weather Data**: Temperature, humidity, rainfall, wind speed
        - **Geological Data**: Slope angle, rock type, fracture density
        """)
    
    with col2:
        if st.session_state.data_generated:
            st.success("Training data ready")
            st.info(f"Dataset: {DATASET_SIZE} samples")
        else:
            st.warning("No training data")
    
    # Data generation controls
    st.subheader("Generation Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dataset_size = st.number_input(
            "Dataset Size",
            min_value=1000,
            max_value=10000,
            value=DATASET_SIZE,
            step=1000
        )
    
    with col2:
        image_size = st.selectbox(
            "Image Size",
            options=[64, 128, 224],
            index=0
        )
    
    with col3:
        risk_distribution = st.selectbox(
            "Risk Distribution",
            options=["Balanced", "Low-Heavy", "High-Heavy"],
            index=0
        )
    
    # Generate data button
    if st.button("Generate Training Data", type="primary"):
        with st.spinner("Generating synthetic data..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize data generator
                generator = RockfallDataGenerator(
                    dataset_size=dataset_size,
                    image_size=(image_size, image_size)
                )
                
                # Generate data with progress updates
                def update_progress(current, total, stage):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"{stage}: {current}/{total}")
                
                generator.generate_complete_dataset(progress_callback=update_progress)
                
                st.session_state.data_generated = True
                st.success("Successfully generated {dataset_size} training samples!")
                
                # Show data preview
                show_data_preview(generator)
                
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
    
    # Data preview section
    if st.session_state.data_generated:
        st.subheader("Data Preview")
        show_data_summary()

def show_model_training():
    """Model training interface"""
    
    st.subheader("Model Training")
    
    if not st.session_state.data_generated:
        st.warning("Please generate training data first.")
        return
    
    # Training configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        The system uses a hybrid approach:
        - **CNN Model**: Analyzes DEM images for visual patterns
        - **Random Forest**: Processes tabular data (weather, soil, geological)
        - **Ensemble**: Combines predictions for final risk assessment
        """)
    
    with col2:
        if st.session_state.model_trained:
            st.success("Models trained and ready")
            if os.path.exists(CNN_MODEL_PATH) and os.path.exists(RF_MODEL_PATH):
                st.info("Model files saved")
        else:
            st.warning("Models not trained")
    
    # Training parameters
    st.subheader("Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input("CNN Epochs", min_value=5, max_value=100, value=20)
        batch_size = st.number_input("Batch Size", min_value=16, max_value=128, value=32)
    
    with col2:
        rf_estimators = st.number_input("RF Estimators", min_value=50, max_value=500, value=100)
        test_split = st.slider("Test Split", min_value=0.1, max_value=0.4, value=0.2)
    
    with col3:
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01, 0.1],
            value=0.001
        )
    
    # Training button
    if st.button("Train Models", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            
            try:
                # Initialize trainer
                trainer = ModelTrainer()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load data
                status_text.text("Loading training data...")
                progress_bar.progress(0.1)
                
                trainer.load_data()
                
                # Train KNN
                status_text.text("Training KNN model...")
                progress_bar.progress(0.2)
                
                knn_metrics = trainer.train_knn_model()
                
                progress_bar.progress(0.6)
                
                # Train Random Forest
                status_text.text("Training Random Forest model...")
                rf_metrics = trainer.train_rf_model(n_estimators=rf_estimators)
                
                progress_bar.progress(0.8)
                
                # Save models
                status_text.text("Saving trained models...")
                trainer.save_models()
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                
                st.session_state.model_trained = True
                st.success("Models trained successfully!")
                
                # Show training results
                show_training_results(knn_metrics, rf_metrics)
                
            except Exception as e:
                st.error(f"Training failed: {str(e)}")

def show_settings():
    """System settings and configuration"""
    
    st.subheader("System Settings")
    
    # Alert thresholds
    st.subheader("Alert Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        high_risk_threshold = st.slider(
            "High Risk Threshold",
            min_value=0.5,
            max_value=1.0,
            value=HIGH_RISK_THRESHOLD,
            step=0.05,
            format="%.2f"
        )
    
    with col2:
        medium_risk_threshold = st.slider(
            "Medium Risk Threshold", 
            min_value=0.1,
            max_value=0.8,
            value=MEDIUM_RISK_THRESHOLD,
            step=0.05,
            format="%.2f"
        )
    
    # Update interval
    st.subheader("Data Update Settings")
    
    update_interval = st.number_input(
        "Live Update Interval (seconds)",
        min_value=5,
        max_value=60,
        value=10
    )
    
    # Model retraining
    st.subheader("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Training Data"):
            if st.button("Confirm Clear Data", type="secondary"):
                clear_training_data()
                st.success("Training data cleared")
    
    with col2:
        if st.button("Reset Models"):
            if st.button("Confirm Reset Models", type="secondary"):
                reset_models()
                st.success("Models reset")
    
    # System information
    st.subheader("System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("**Data Status:**")
        st.write(f"- Training Data: {'Ready' if st.session_state.data_generated else 'Not Generated'}")
        st.write(f"- Models: {'Trained' if st.session_state.model_trained else 'Not Trained'}")
        st.write(f"- Prediction History: {len(st.session_state.prediction_data)} records")
    
    with info_col2:
        st.markdown("**File System:**")
        st.write(f"- CNN Model: {'Found' if os.path.exists(CNN_MODEL_PATH) else 'Missing'}")
        st.write(f"- RF Model: {'Found' if os.path.exists(RF_MODEL_PATH) else 'Missing'}")
        st.write(f"- Scaler: {'Found' if os.path.exists(SCALER_PATH) else 'Missing'}")

def show_data_preview(generator):
    """Show preview of generated data"""
    
    st.subheader("Generated Data Preview")
    
    # Sample images
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sample DEM Images:**")
        # Show a few sample images
        for i in range(min(3, len(generator.images))):
            st.image(generator.images[i], width=200, caption=f"Sample {i+1}")
    
    with col2:
        st.markdown("**Data Distribution:**")
        if hasattr(generator, 'labels'):
            risk_dist = pd.Series(generator.labels).value_counts()
            fig = px.bar(x=risk_dist.index, y=risk_dist.values, title="Risk Level Distribution")
            st.plotly_chart(fig, use_container_width=True)

def show_training_results(knn_metrics, rf_metrics):
    """Display training results and metrics"""
    
    st.subheader("Training Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**KNN Model Metrics:**")
        if knn_metrics:
            st.metric("Accuracy", f"{knn_metrics['accuracy']:.4f}")
            st.write("**Classification Report:**")
            st.text(knn_metrics['classification_report'])
            
            # Display confusion matrix
            st.write("**Confusion Matrix:**")
            if os.path.exists("knn_confusion_matrix.png"):
                st.image("knn_confusion_matrix.png")
    
    
    with col2:
        st.markdown("**Random Forest Metrics:**")
        if rf_metrics:
            st.metric("Accuracy", f"{rf_metrics['accuracy']:.3f}")
            st.metric("Precision", f"{rf_metrics['precision']:.3f}")
            st.metric("Recall", f"{rf_metrics['recall']:.3f}")
            st.metric("F1-Score", f"{rf_metrics['f1_score']:.3f}")
            
            # Feature importance
            if 'feature_importance' in rf_metrics:
                importance_df = pd.DataFrame({
                    'Feature': rf_metrics['feature_names'],
                    'Importance': rf_metrics['feature_importance']
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Feature Importance'
                )
                st.plotly_chart(fig, use_container_width=True)

def get_recommended_action(risk_level):
    """Get recommended action based on risk level"""
    
    actions = {
        'Low': 'Continue normal operations',
        'Medium': 'Increase monitoring, reduce personnel',
        'High': 'Evacuate area immediately'
    }
    return actions.get(risk_level, 'Unknown')

def clear_training_data():
    """Clear all training data"""
    
    st.session_state.data_generated = False
    # Remove data files if they exist
    for path in [TABULAR_DATA_PATH, IMAGE_DATA_PATH]:
        if os.path.exists(path):
            os.remove(path)

def reset_models():
    """Reset all trained models"""
    
    st.session_state.model_trained = False
    # Remove model files if they exist
    for path in [CNN_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH]:
        if os.path.exists(path):
            os.remove(path)

def generate_live_prediction():
    """Generate a new prediction with current timestamp"""
    
    try:
        predictor = RockfallPredictor()
        
        # Generate synthetic rockfall parameters
        rockfall_params = {
            'slope_angle': round(random.uniform(20, 70), 1),
            'rock_mass': round(random.uniform(10, 500), 1),
            'rock_density': round(random.uniform(2000, 3000), 1),
            'rock_shape': random.choice(['angular', 'rounded', 'irregular']),
            'friction_angle': round(random.uniform(20, 45), 1),
            'restitution_coefficient': round(random.uniform(0.3, 0.9), 2),
            'slope_roughness': round(random.uniform(0.01, 0.2), 2),
            'vegetation': random.choice(['none', 'light', 'moderate', 'heavy']),
            'water_content': round(random.uniform(0, 0.4), 2),
            'temperature': round(random.uniform(-10, 40), 1),
            'rainfall': round(random.uniform(0, 100), 1),
            'seismic_events': random.choice([0, 0, 0, 1, 2]),
            'release_point': round(random.uniform(10, 200), 1)
        }
        
        # Make prediction with tabular data only
        probability, risk_level = predictor.predict(tabular_data=rockfall_params)
        
        # Create prediction record with all required keys for dashboard
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'probability': probability,
            'risk_level': risk_level,
            'temperature': rockfall_params.get('temperature', 25.0),
            'humidity': rockfall_params.get('humidity', 50.0),
            'rainfall': rockfall_params.get('rainfall', 0.0),
            'wind_speed': rockfall_params.get('wind_speed', 5.0),
            'soil_moisture': rockfall_params.get('soil_moisture', 20.0),
            'slope_angle': rockfall_params.get('slope_angle', 30.0),
            'rock_mass': rockfall_params.get('rock_mass', 100.0),
            'vegetation': rockfall_params.get('vegetation', 'none'),
            'seismic_events': rockfall_params.get('seismic_events', 0)
        }
        
        # Store in session state
        st.session_state.prediction_data.append(prediction_record)
        
        # Keep only last 100 records
        if len(st.session_state.prediction_data) > 100:
            st.session_state.prediction_data = st.session_state.prediction_data[-100:]
        
        st.session_state.last_update = datetime.now()
        
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


def show_data_summary():
    """Display summary of generated data"""
    try:
        # Load and display image data summary
        images = np.load('data/dem_images.npy')
        st.write("**Image Data Summary:**")
        st.write(f"- Number of images: {len(images)}")
        st.write(f"- Image dimensions: {images.shape[1:3]}")
        st.write(f"- Image channels: {images.shape[3]}")
        
        # Display sample images
        st.write("**Sample Images:**")
        cols = st.columns(4)
        for i, col in enumerate(cols):
            if i < len(images):
                col.image(images[i], caption=f"Sample {i+1}", use_container_width=True)
        
        # Load and display tabular data summary
        df = pd.read_csv('data/tabular_features.csv')
        st.write("\n**Tabular Data Summary:**")
        st.write(f"- Number of samples: {len(df)}")
        st.write(f"- Number of features: {len(df.columns) - 2}")  # Excluding risk_level and probability
        
        # Display risk level distribution
        risk_dist = df['risk_level'].value_counts()
        st.write("\n**Risk Level Distribution:**")
        st.bar_chart(risk_dist)
        
        # Display feature statistics - exclude non-numeric columns
        st.write("\n**Feature Statistics:**")
        numeric_df = df.drop(columns=['risk_level'])
        st.dataframe(numeric_df.describe())
        
        # Display correlation heatmap - only for numeric columns
        st.write("\n**Feature Correlations:**")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Calculate correlation only for numeric columns
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {str(e)}")
        
    except Exception as e:
        st.error(f"Error displaying data summary: {str(e)}")
        st.info("Please generate data first using the options above.")


if __name__ == "__main__":
    main()
