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
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_generator import RockfallDataGenerator
from model_trainer import ModelTrainer
from predictor import RockfallPredictor
from dashboard import Dashboard
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
        page_icon="🏔️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏔️ AI-Based Rockfall Prediction & Alert System")
    st.markdown("*Advanced machine learning system for open-pit mine safety*")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dashboard", "Data Generation", "Model Training", "System Settings"]
    )
    
    if page == "Dashboard":
        show_dashboard()
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
        st.warning("⚠️ Please generate training data first in the Data Generation section.")
        return
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first in the Model Training section.")
        return
    
    # Live mode toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("🎯 Real-time Rockfall Monitoring")
    with col2:
        if st.button("🔄 Refresh Data"):
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
            label="🎯 Rockfall Probability",
            value=f"{probability:.1%}",
            delta=f"{risk_level} Risk"
        )
        st.markdown(f"<div style='background-color: {color}20; padding: 10px; border-radius: 5px; text-align: center;'><strong>{risk_level.upper()} RISK</strong></div>", unsafe_allow_html=True)
    
    with col2:
        temp = latest_data['temperature']
        humidity = latest_data['humidity']
        
        st.metric(
            label="🌡️ Temperature",
            value=f"{temp:.1f}°C"
        )
        st.metric(
            label="💧 Humidity",
            value=f"{humidity:.1f}%"
        )
    
    with col3:
        rainfall = latest_data['rainfall']
        wind_speed = latest_data['wind_speed']
        
        st.metric(
            label="🌧️ Rainfall",
            value=f"{rainfall:.1f}mm"
        )
        st.metric(
            label="💨 Wind Speed",
            value=f"{wind_speed:.1f}km/h"
        )
    
    with col4:
        soil_moisture = latest_data['soil_moisture']
        slope_angle = latest_data['slope_angle']
        
        st.metric(
            label="🌱 Soil Moisture",
            value=f"{soil_moisture:.1f}%"
        )
        st.metric(
            label="📐 Slope Angle",
            value=f"{slope_angle:.1f}°"
        )

def show_risk_analysis():
    """Display risk heatmap and detailed analysis"""
    
    st.subheader("📊 Risk Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Risk heatmap
        st.subheader("🗺️ Risk Heatmap")
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
    
    st.subheader("📈 Historical Trends")
    
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
    
    st.subheader("🚨 Alert System")
    
    if not st.session_state.prediction_data:
        return
    
    latest_data = st.session_state.prediction_data[-1]
    probability = latest_data['probability']
    risk_level = latest_data['risk_level']
    
    # Alert status
    if risk_level == 'High':
        st.error(f"🚨 HIGH RISK ALERT - Probability: {probability:.1%}")
        st.markdown("**Recommended Actions:**")
        st.markdown("- ⛔ Evacuate personnel from high-risk areas immediately")
        st.markdown("- 📞 Alert emergency response team")
        st.markdown("- 🔒 Restrict access to unstable slopes")
        st.markdown("- 📊 Increase monitoring frequency")
        
    elif risk_level == 'Medium':
        st.warning(f"⚠️ MEDIUM RISK - Probability: {probability:.1%}")
        st.markdown("**Recommended Actions:**")
        st.markdown("- 👥 Reduce personnel in potentially affected areas")
        st.markdown("- 🔍 Increase visual inspections")
        st.markdown("- 📡 Enhanced sensor monitoring")
        st.markdown("- 📋 Review safety protocols")
        
    else:
        st.success(f"✅ LOW RISK - Probability: {probability:.1%}")
        st.markdown("**Current Status:**")
        st.markdown("- ✅ Normal operations can continue")
        st.markdown("- 🔄 Maintain standard monitoring procedures")
        st.markdown("- 📅 Continue routine inspections")
    
    # Recent alerts log
    st.subheader("📋 Recent Alerts Log")
    
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
    
    st.subheader("🔧 Synthetic Data Generation")
    
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
            st.success("✅ Training data ready")
            st.info(f"Dataset: {DATASET_SIZE} samples")
        else:
            st.warning("⏳ No training data")
    
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
    if st.button("🎲 Generate Training Data", type="primary"):
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
                st.success(f"✅ Successfully generated {dataset_size} training samples!")
                
                # Show data preview
                show_data_preview(generator)
                
            except Exception as e:
                st.error(f"❌ Error generating data: {str(e)}")
    
    # Data preview section
    if st.session_state.data_generated:
        st.subheader("📊 Data Preview")
        show_data_summary()

def show_model_training():
    """Model training interface"""
    
    st.subheader("🤖 Model Training")
    
    if not st.session_state.data_generated:
        st.warning("⚠️ Please generate training data first.")
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
            st.success("✅ Models trained and ready")
            if os.path.exists(CNN_MODEL_PATH) and os.path.exists(RF_MODEL_PATH):
                st.info("📁 Model files saved")
        else:
            st.warning("⏳ Models not trained")
    
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
    if st.button("🚀 Train Models", type="primary"):
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
                
                # Train CNN
                status_text.text("Training CNN model...")
                progress_bar.progress(0.2)
                
                cnn_history = trainer.train_cnn_model(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
                
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
                st.success("✅ Models trained successfully!")
                
                # Show training results
                show_training_results(cnn_history, rf_metrics)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

def show_settings():
    """System settings and configuration"""
    
    st.subheader("⚙️ System Settings")
    
    # Alert thresholds
    st.subheader("🚨 Alert Thresholds")
    
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
    st.subheader("🔄 Data Update Settings")
    
    update_interval = st.number_input(
        "Live Update Interval (seconds)",
        min_value=5,
        max_value=60,
        value=10
    )
    
    # Model retraining
    st.subheader("🔄 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Training Data"):
            if st.button("Confirm Clear Data", type="secondary"):
                clear_training_data()
                st.success("Training data cleared")
    
    with col2:
        if st.button("🗑️ Reset Models"):
            if st.button("Confirm Reset Models", type="secondary"):
                reset_models()
                st.success("Models reset")
    
    # System information
    st.subheader("ℹ️ System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("**Data Status:**")
        st.write(f"- Training Data: {'✅ Ready' if st.session_state.data_generated else '❌ Not Generated'}")
        st.write(f"- Models: {'✅ Trained' if st.session_state.model_trained else '❌ Not Trained'}")
        st.write(f"- Prediction History: {len(st.session_state.prediction_data)} records")
    
    with info_col2:
        st.markdown("**File System:**")
        st.write(f"- CNN Model: {'✅ Found' if os.path.exists(CNN_MODEL_PATH) else '❌ Missing'}")
        st.write(f"- RF Model: {'✅ Found' if os.path.exists(RF_MODEL_PATH) else '❌ Missing'}")
        st.write(f"- Scaler: {'✅ Found' if os.path.exists(SCALER_PATH) else '❌ Missing'}")

def generate_live_prediction():
    """Generate a new prediction with current timestamp"""
    
    try:
        predictor = RockfallPredictor()
        
        # Generate synthetic current conditions
        data_generator = RockfallDataGenerator(dataset_size=1)
        image, tabular_data, _ = data_generator.generate_single_sample()
        
        # Make prediction
        probability, risk_level = predictor.predict(image, tabular_data)
        
        # Create prediction record
        prediction_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'probability': probability,
            'risk_level': risk_level,
            'temperature': tabular_data[0],
            'humidity': tabular_data[1], 
            'rainfall': tabular_data[2],
            'wind_speed': tabular_data[3],
            'soil_moisture': tabular_data[4],
            'slope_angle': tabular_data[5]
        }
        
        # Store in session state
        st.session_state.prediction_data.append(prediction_record)
        
        # Keep only last 100 records
        if len(st.session_state.prediction_data) > 100:
            st.session_state.prediction_data = st.session_state.prediction_data[-100:]
        
        st.session_state.last_update = datetime.now()
        
    except Exception as e:
        st.error(f"Error generating prediction: {str(e)}")

def show_data_preview(generator):
    """Show preview of generated data"""
    
    st.subheader("📊 Generated Data Preview")
    
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

def show_data_summary():
    """Show summary of existing training data"""
    
    try:
        # Load and display data statistics
        if os.path.exists(TABULAR_DATA_PATH):
            df = pd.read_csv(TABULAR_DATA_PATH)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Statistics:**")
                st.write(f"Total samples: {len(df)}")
                st.write(f"Features: {len(df.columns)-1}")
                
                if 'risk_level' in df.columns:
                    risk_counts = df['risk_level'].value_counts()
                    st.write("Risk distribution:")
                    for risk, count in risk_counts.items():
                        st.write(f"  - {risk}: {count} ({count/len(df)*100:.1f}%)")
            
            with col2:
                st.markdown("**Feature Correlations:**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, title="Feature Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data summary: {str(e)}")

def show_training_results(cnn_history, rf_metrics):
    """Display training results and metrics"""
    
    st.subheader("📈 Training Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**CNN Training History:**")
        if cnn_history:
            # Plot training history
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Model Accuracy', 'Model Loss']
            )
            
            epochs = range(1, len(cnn_history['accuracy']) + 1)
            
            fig.add_trace(
                go.Scatter(x=list(epochs), y=cnn_history['accuracy'], name='Training Accuracy'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=cnn_history['val_accuracy'], name='Validation Accuracy'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=list(epochs), y=cnn_history['loss'], name='Training Loss'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=list(epochs), y=cnn_history['val_loss'], name='Validation Loss'),
                row=2, col=1
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
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

if __name__ == "__main__":
    main()
