import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
from predictor import RockfallPredictor
from data_generator import RockfallDataGenerator
from utils import create_risk_heatmap, get_risk_color, format_probability
from config import *

class Dashboard:
    """Streamlit dashboard for rockfall prediction system"""
    
    def __init__(self):
        self.predictor = None
        self.initialize_predictor()
    
    def initialize_predictor(self):
        """Initialize the prediction system"""
        try:
            self.predictor = RockfallPredictor()
        except Exception as e:
            st.error(f"Failed to initialize predictor: {str(e)}")
            self.predictor = None
    
    def render_header(self):
        """Render dashboard header"""
        st.title("🏔️ AI Rockfall Prediction Dashboard")
        st.markdown("*Real-time monitoring and risk assessment for open-pit mine safety*")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if self.predictor and self.predictor.cnn_model is not None:
                st.success("🤖 AI Models: Ready")
            else:
                st.error("🤖 AI Models: Not Ready")
        
        with col2:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.info(f"🕐 Last Update: {current_time}")
        
        with col3:
            if 'prediction_data' in st.session_state and st.session_state.prediction_data:
                latest_data = st.session_state.prediction_data[-1]
                risk_level = latest_data['risk_level']
                color = get_risk_color(risk_level)
                st.markdown(f"🎯 Current Risk: <span style='color:{color}'><strong>{risk_level}</strong></span>", unsafe_allow_html=True)
            else:
                st.warning("🎯 Current Risk: No Data")
        
        with col4:
            if 'live_mode' in st.session_state and st.session_state.live_mode:
                st.success("📡 Live Mode: ON")
            else:
                st.info("📡 Live Mode: OFF")
    
    def render_current_conditions(self):
        """Render current environmental conditions"""
        
        st.subheader("🌡️ Current Environmental Conditions")
        
        if not st.session_state.prediction_data:
            st.warning("No current data available")
            return
        
        latest_data = st.session_state.prediction_data[-1]
        
        # Weather conditions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp = latest_data['temperature']
            st.metric(
                label="Temperature",
                value=f"{temp:.1f}°C",
                delta=f"{temp-20:.1f}°C from avg"
            )
        
        with col2:
            humidity = latest_data['humidity']
            st.metric(
                label="Humidity",
                value=f"{humidity:.1f}%",
                delta=f"{humidity-50:.1f}% from avg"
            )
        
        with col3:
            rainfall = latest_data['rainfall']
            st.metric(
                label="Rainfall (24h)",
                value=f"{rainfall:.1f}mm",
                delta="Heavy" if rainfall > 50 else "Normal"
            )
        
        with col4:
            wind_speed = latest_data['wind_speed']
            st.metric(
                label="Wind Speed",
                value=f"{wind_speed:.1f}km/h",
                delta="Strong" if wind_speed > 25 else "Normal"
            )
        
        # Soil and geological conditions
        st.subheader("🌱 Soil & Geological Conditions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            soil_moisture = latest_data['soil_moisture']
            st.metric(
                label="Soil Moisture",
                value=f"{soil_moisture:.1f}%",
                delta="High" if soil_moisture > 30 else "Normal"
            )
        
        with col2:
            slope_angle = latest_data['slope_angle']
            st.metric(
                label="Slope Angle",
                value=f"{slope_angle:.1f}°",
                delta="Steep" if slope_angle > 45 else "Moderate"
            )
        
        with col3:
            probability = latest_data['probability']
            risk_level = latest_data['risk_level']
            color = get_risk_color(risk_level)
            
            st.metric(
                label="Rockfall Probability",
                value=format_probability(probability),
                delta=risk_level
            )
            
            # Risk indicator bar
            st.markdown(f"""
                <div style='background: linear-gradient(90deg, 
                    green 0%, green 30%, 
                    yellow 30%, yellow 60%, 
                    red 60%, red 100%); 
                    height: 20px; border-radius: 10px; position: relative;'>
                    <div style='position: absolute; left: {probability*100:.1f}%; 
                        top: -5px; font-size: 30px;'>⬇️</div>
                </div>
            """, unsafe_allow_html=True)
    
    def render_risk_heatmap(self):
        """Render risk heatmap visualization"""
        
        st.subheader("🗺️ Mine Site Risk Heatmap")
        
        # Generate heatmap data
        heatmap_data = create_risk_heatmap()
        
        # Create heatmap plot
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='RdYlGn_r',
            aspect='auto',
            title="Current Risk Distribution Across Mine Site"
        )
        
        fig.update_layout(
            xaxis_title="East-West Distance (m)",
            yaxis_title="North-South Distance (m)",
            coloraxis_colorbar_title="Risk Score (0-1)",
            height=400
        )
        
        # Add annotations for high-risk areas
        high_risk_coords = np.where(heatmap_data > 0.7)
        for i in range(min(5, len(high_risk_coords[0]))):  # Limit to 5 annotations
            y, x = high_risk_coords[0][i], high_risk_coords[1][i]
            fig.add_annotation(
                x=x, y=y,
                text="⚠️",
                showarrow=False,
                font=dict(size=20)
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk area summary
        col1, col2, col3 = st.columns(3)
        
        high_risk_percent = (heatmap_data > HIGH_RISK_THRESHOLD).sum() / heatmap_data.size * 100
        medium_risk_percent = ((heatmap_data > MEDIUM_RISK_THRESHOLD) & (heatmap_data <= HIGH_RISK_THRESHOLD)).sum() / heatmap_data.size * 100
        low_risk_percent = 100 - high_risk_percent - medium_risk_percent
        
        with col1:
            st.metric("🟢 Low Risk Areas", f"{low_risk_percent:.1f}%")
        with col2:
            st.metric("🟡 Medium Risk Areas", f"{medium_risk_percent:.1f}%")
        with col3:
            st.metric("🔴 High Risk Areas", f"{high_risk_percent:.1f}%")
    
    def render_historical_trends(self):
        """Render historical trends and patterns"""
        
        st.subheader("📈 Historical Trends & Analysis")
        
        if len(st.session_state.prediction_data) < 5:
            st.info("Collecting more data points to show meaningful trends...")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.prediction_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time series analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['probability'],
                mode='lines+markers',
                name='Rockfall Probability',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            
            # Add risk threshold lines
            fig.add_hline(y=HIGH_RISK_THRESHOLD, line_dash="dash", line_color="red", annotation_text="High Risk")
            fig.add_hline(y=MEDIUM_RISK_THRESHOLD, line_dash="dash", line_color="orange", annotation_text="Medium Risk")
            
            fig.update_layout(
                title="Rockfall Probability Over Time",
                xaxis_title="Time",
                yaxis_title="Probability",
                yaxis=dict(tickformat='.1%'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level distribution
            risk_counts = df['risk_level'].value_counts()
            
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color_discrete_map={
                    'Low': '#28a745',
                    'Medium': '#ffc107',
                    'High': '#dc3545'
                }
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔍 Weather-Risk Correlation Analysis")
        
        # Create correlation heatmap
        numeric_cols = ['probability', 'temperature', 'humidity', 'rainfall', 'wind_speed', 'soil_moisture', 'slope_angle']
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            color_continuous_scale='RdBu_r',
            aspect='auto',
            title="Feature Correlation Matrix"
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        # Calculate some insights
        avg_probability = df['probability'].mean()
        max_probability = df['probability'].max()
        high_risk_periods = (df['probability'] > HIGH_RISK_THRESHOLD).sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"📊 Average Risk: {avg_probability:.1%}")
        with col2:
            st.warning(f"⚠️ Peak Risk: {max_probability:.1%}")
        with col3:
            st.error(f"🚨 High Risk Periods: {high_risk_periods}")
    
    def render_alert_system(self):
        """Render alert system and recommendations"""
        
        st.subheader("🚨 Alert System & Recommendations")
        
        if not st.session_state.prediction_data:
            st.warning("No data available for alerts")
            return
        
        latest_data = st.session_state.prediction_data[-1]
        probability = latest_data['probability']
        risk_level = latest_data['risk_level']
        
        # Current alert status
        if risk_level == 'High':
            st.error("🚨 **HIGH RISK ALERT**")
            st.markdown(f"**Current Probability:** {probability:.1%}")
            
            st.markdown("### Immediate Actions Required:")
            st.markdown("- ⛔ **EVACUATE** personnel from high-risk areas immediately")
            st.markdown("- 📞 **ALERT** emergency response team and mine safety officer")
            st.markdown("- 🔒 **RESTRICT** access to unstable slopes and danger zones")
            st.markdown("- 📡 **INCREASE** monitoring frequency to continuous mode")
            st.markdown("- 📋 **DOCUMENT** all actions taken for safety compliance")
            
        elif risk_level == 'Medium':
            st.warning("⚠️ **MEDIUM RISK WARNING**")
            st.markdown(f"**Current Probability:** {probability:.1%}")
            
            st.markdown("### Recommended Actions:")
            st.markdown("- 👥 **REDUCE** personnel in potentially affected areas")
            st.markdown("- 🔍 **INCREASE** visual inspections and patrols")
            st.markdown("- 📡 **ENHANCE** sensor monitoring and data collection")
            st.markdown("- 📋 **REVIEW** safety protocols and evacuation routes")
            st.markdown("- 💬 **BRIEF** all personnel on current risk status")
            
        else:
            st.success("✅ **NORMAL OPERATIONS**")
            st.markdown(f"**Current Probability:** {probability:.1%}")
            
            st.markdown("### Standard Procedures:")
            st.markdown("- ✅ **CONTINUE** normal mining operations")
            st.markdown("- 🔄 **MAINTAIN** standard monitoring procedures")
            st.markdown("- 📅 **CONDUCT** routine safety inspections")
            st.markdown("- 📊 **MONITOR** trends and environmental changes")
        
        # Recent alerts log
        st.subheader("📋 Recent Alerts History")
        
        # Filter recent high/medium risk events
        recent_alerts = []
        for data in st.session_state.prediction_data[-20:]:  # Last 20 entries
            if data['risk_level'] in ['Medium', 'High']:
                recent_alerts.append({
                    'Timestamp': data['timestamp'],
                    'Risk Level': data['risk_level'],
                    'Probability': f"{data['probability']:.1%}",
                    'Temperature': f"{data['temperature']:.1f}°C",
                    'Rainfall': f"{data['rainfall']:.1f}mm",
                    'Action Taken': self.get_recommended_action(data['risk_level'])
                })
        
        if recent_alerts:
            alerts_df = pd.DataFrame(recent_alerts)
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.success("✅ No recent alerts - system operating normally")
        
        # Alert statistics
        if len(st.session_state.prediction_data) >= 10:
            total_records = len(st.session_state.prediction_data)
            high_risk_count = sum(1 for d in st.session_state.prediction_data if d['risk_level'] == 'High')
            medium_risk_count = sum(1 for d in st.session_state.prediction_data if d['risk_level'] == 'Medium')
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Alerts", high_risk_count + medium_risk_count)
            with col2:
                st.metric("High Risk Events", high_risk_count)
            with col3:
                st.metric("Medium Risk Events", medium_risk_count)
    
    def render_system_status(self):
        """Render system status and diagnostics"""
        
        st.subheader("⚙️ System Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Status:**")
            if self.predictor:
                cnn_status = "✅ Ready" if self.predictor.cnn_model is not None else "❌ Not Ready"
                rf_status = "✅ Ready" if self.predictor.rf_model is not None else "❌ Not Ready"
                scaler_status = "✅ Ready" if self.predictor.scaler is not None else "❌ Not Ready"
                
                st.write(f"- CNN Model: {cnn_status}")
                st.write(f"- Random Forest: {rf_status}")
                st.write(f"- Data Scaler: {scaler_status}")
            else:
                st.write("- All Models: ❌ Not Initialized")
        
        with col2:
            st.markdown("**Data Status:**")
            prediction_count = len(st.session_state.prediction_data) if 'prediction_data' in st.session_state else 0
            st.write(f"- Prediction Records: {prediction_count}")
            st.write(f"- Live Mode: {'✅ Active' if st.session_state.get('live_mode', False) else '❌ Inactive'}")
            
            if prediction_count > 0:
                latest_time = st.session_state.prediction_data[-1]['timestamp']
                st.write(f"- Last Update: {latest_time}")
    
    def get_recommended_action(self, risk_level):
        """Get recommended action based on risk level"""
        
        actions = {
            'Low': 'Continue normal operations',
            'Medium': 'Increase monitoring, reduce personnel',
            'High': 'Evacuate area immediately'
        }
        return actions.get(risk_level, 'Unknown')
    
    def render_live_controls(self):
        """Render live monitoring controls"""
        
        st.sidebar.subheader("🎛️ Live Monitoring Controls")
        
        # Auto-refresh toggle
        live_mode = st.sidebar.toggle(
            "Enable Live Mode",
            value=st.session_state.get('live_mode', False),
            help="Automatically refresh predictions every 10 seconds"
        )
        st.session_state.live_mode = live_mode
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now", use_container_width=True):
            self.generate_new_prediction()
            st.rerun()
        
        # Clear history button
        if st.sidebar.button("🗑️ Clear History", use_container_width=True):
            st.session_state.prediction_data = []
            st.success("History cleared!")
            st.rerun()
        
        # Data export
        if st.sidebar.button("📥 Export Data", use_container_width=True):
            if 'prediction_data' in st.session_state and st.session_state.prediction_data:
                df = pd.DataFrame(st.session_state.prediction_data)
                csv = df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"rockfall_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # System settings
        st.sidebar.subheader("⚙️ Alert Settings")
        
        high_threshold = st.sidebar.slider(
            "High Risk Threshold",
            min_value=0.5,
            max_value=1.0,
            value=HIGH_RISK_THRESHOLD,
            step=0.05,
            format="%.2f"
        )
        
        medium_threshold = st.sidebar.slider(
            "Medium Risk Threshold",
            min_value=0.1,
            max_value=0.8,
            value=MEDIUM_RISK_THRESHOLD,
            step=0.05,
            format="%.2f"
        )
    
    def generate_new_prediction(self):
        """Generate a new prediction and add to history"""
        
        try:
            if self.predictor is None:
                st.error("Predictor not initialized")
                return
            
            # Generate synthetic current data
            generator = RockfallDataGenerator(dataset_size=1)
            image, tabular_data, _ = generator.generate_single_sample()
            
            # Make prediction
            probability, risk_level = self.predictor.predict(image, tabular_data)
            
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
                'slope_angle': tabular_data[8]  # slope_angle is at index 8
            }
            
            # Initialize prediction data if it doesn't exist
            if 'prediction_data' not in st.session_state:
                st.session_state.prediction_data = []
            
            # Add to history
            st.session_state.prediction_data.append(prediction_record)
            
            # Keep only last 100 records
            if len(st.session_state.prediction_data) > 100:
                st.session_state.prediction_data = st.session_state.prediction_data[-100:]
            
            # Update last update time
            st.session_state.last_update = datetime.now()
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")
    
    def run_dashboard(self):
        """Main dashboard execution function"""
        
        # Render header
        self.render_header()
        
        # Render live controls in sidebar
        self.render_live_controls()
        
        # Auto-refresh logic
        if st.session_state.get('live_mode', False):
            time_since_update = (datetime.now() - st.session_state.get('last_update', datetime.now())).seconds
            if time_since_update >= 10:  # 10-second refresh
                self.generate_new_prediction()
                st.rerun()
        
        # Generate initial prediction if no data exists
        if not st.session_state.get('prediction_data', []):
            self.generate_new_prediction()
        
        # Main dashboard content
        
        # Current conditions
        self.render_current_conditions()
        
        st.divider()
        
        # Risk heatmap
        self.render_risk_heatmap()
        
        st.divider()
        
        # Historical trends
        self.render_historical_trends()
        
        st.divider()
        
        # Alert system
        self.render_alert_system()
        
        st.divider()
        
        # System status
        self.render_system_status()

# Create dashboard instance
dashboard = Dashboard()
