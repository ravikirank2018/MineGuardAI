import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import folium
from streamlit_folium import folium_static
import time
from predictor import RockfallPredictor
import pyttsx3
import threading
from browser_tts import speak_text
from queue import Queue
import json

class CameraPage:
    def __init__(self):
        self.predictor = RockfallPredictor()
        self.voice_engine = None
        self.alert_queue = Queue()
        self.alert_thread = None
        self.initialize_voice_engine()
        
    def initialize_voice_engine(self):
        """Initialize text-to-speech engine"""
        try:
            self.voice_engine = pyttsx3.init()
            # Configure voice properties
            self.voice_engine.setProperty('rate', 150)  # Speed of speech
            self.voice_engine.setProperty('volume', 0.9)  # Volume (0-1)
            # Start alert processing thread
            self.alert_thread = threading.Thread(target=self.process_alerts, daemon=True)
            self.alert_thread.start()
            # Log successful initialization
            print("Voice engine initialized successfully")
        except Exception as e:
            st.error(f"Could not initialize voice system: {str(e)}")
            print(f"Voice engine initialization error: {str(e)}")
            self.voice_engine = None
    
    def process_alerts(self):
        """Process voice alerts in background thread"""
        while True:
            try:
                alert_text = self.alert_queue.get()
                if self.voice_engine:
                    self.voice_engine.say(alert_text)
                    self.voice_engine.runAndWait()
            except Exception as e:
                print(f"Error processing voice alert: {str(e)}")
            time.sleep(0.1)  # Prevent CPU overuse
    
    def speak_alert(self, text):
        """Speak alert using browser TTS (for web users) and server TTS (for local server)"""
        # Browser-based TTS for web users
        speak_text(text)
        # Optionally, keep server-side TTS for local speakers
        if self.voice_engine:
            print(f"Adding alert to queue: {text}")
            self.alert_queue.put(text)
        else:
            print("Voice engine not available, cannot speak alert")
    
    def generate_voice_alert(self, risk_level, location, time_to_impact):
        """Generate location-specific voice alert"""
        alerts = {
            'High': f"URGENT! High risk of rockfall detected at {location}. Estimated impact in {time_to_impact} minutes. Evacuate immediately following marked routes.",
            'Medium': f"Warning! Medium risk of rockfall at {location}. Estimated time to impact: {time_to_impact} minutes. Please prepare for evacuation.",
            'Low': f"Notice: Low risk of rockfall observed at {location}. Monitoring situation. No immediate action required."
        }
        return alerts.get(risk_level, "Alert: Unknown risk level detected. Please check monitoring system.")
    
    def setup_camera(self):
        """Initialize camera settings"""
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        if 'camera_source' not in st.session_state:
            st.session_state.camera_source = 0
            
    def process_frame(self, frame):
        """Process camera frame and return risk assessment"""
        try:
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Make prediction
            probability, risk_level = self.predictor.predict(frame)
            
            # Simulate location detection (replace with actual detection)
            location = "North Face, Section A-3"
            
            # Calculate time to impact based on probability
            time_to_impact = max(1, int((1 - probability) * 10))
            
            return frame, probability, risk_level, location, time_to_impact
            
        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            return frame, 0.1, "Low", "Unknown", 10
    
    def render_evacuation_map(self, location, risk_level, time_to_impact):
        """Render interactive evacuation map"""
        try:
            # Create map centered on mine location (example coordinates)
            m = folium.Map(location=[37.7749, -122.4194], zoom_start=16)

            # Add risk zone (simulated coordinates)
            risk_radius = 200 if risk_level == "High" else 100
            folium.Circle(
                location=[37.7749, -122.4194],
                radius=risk_radius,
                color='red' if risk_level == "High" else 'yellow',
                fill=True,
                popup=f"Risk Zone - {location}"
            ).add_to(m)

            # Add safe zones
            safe_zones = [
                {"name": "Assembly Point A", "coords": [37.7760, -122.4180]},
                {"name": "Emergency Shelter B", "coords": [37.7740, -122.4210]},
                {"name": "Medical Station", "coords": [37.7755, -122.4165]}
            ]

            for zone in safe_zones:
                folium.Marker(
                    zone["coords"],
                    popup=zone["name"],
                    icon=folium.Icon(color='green', icon='info-sign')
                ).add_to(m)

            # Add evacuation routes
            routes = [
                {"points": [[37.7749, -122.4194], [37.7760, -122.4180]], "name": "Route A"},
                {"points": [[37.7749, -122.4194], [37.7740, -122.4210]], "name": "Route B"}
            ]

            # Highlight Route A for demo
            folium.PolyLine(
                routes[0]["points"],
                weight=7,
                color='blue',
                popup="Demo Evacuation Route (Core Area)"
            ).add_to(m)

            for route in routes:
                folium.PolyLine(
                    route["points"],
                    weight=3,
                    color='green',
                    popup=route["name"]
                ).add_to(m)

            # Display map
            folium_static(m)

            # Add demo voice navigation button
            if st.button("Demo: Voice Navigation for Core Mining Area Evacuation", key="voice_nav_demo"):
                nav_text = (
                    "Attention: Rockfall detected in the core mining area. "
                    "Please evacuate immediately via the highlighted blue route (Route A) to Assembly Point A. "
                    "Follow the marked path and listen for further instructions."
                )
                self.speak_alert(nav_text)
                st.success("Voice navigation demo triggered. Listen for evacuation instructions.")

        except Exception as e:
            st.error(f"Error rendering map: {str(e)}")
    
    def show_analysis(self, probability, risk_level, location, time_to_impact):
        """Display risk analysis and alerts"""
        try:
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Level", risk_level)
            with col2:
                st.metric("Probability", f"{probability:.2%}")
            with col3:
                st.metric("Time to Impact", f"{time_to_impact} min")
            
            # Show alert based on risk level
            alert_colors = {
                "High": "red",
                "Medium": "orange",
                "Low": "green"
            }
            
            alert_box = st.empty()
            alert_color = alert_colors.get(risk_level, "blue")
            
            # Generate and display alert message
            alert_message = self.generate_voice_alert(risk_level, location, time_to_impact)
            alert_box.markdown(f"<div style='padding: 10px; background-color: {alert_color}; color: white; border-radius: 5px;'>{alert_message}</div>", unsafe_allow_html=True)
            
            # Trigger voice alert for high and medium risks
            if risk_level in ["High", "Medium"]:
                self.speak_alert(alert_message)
            
            # Show historical alerts
            if 'alerts' not in st.session_state:
                st.session_state.alerts = []
            
            # Add new alert to history
            new_alert = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'risk_level': risk_level,
                'location': location,
                'message': alert_message
            }
            st.session_state.alerts.insert(0, new_alert)
            
            # Keep last 10 alerts
            st.session_state.alerts = st.session_state.alerts[:10]
            
            # Display alert history
            with st.expander("Alert History"):
                for alert in st.session_state.alerts:
                    st.markdown(f"""
                    **{alert['timestamp']}** - {alert['risk_level']}  
                    Location: {alert['location']}  
                    Message: {alert['message']}  
                    ---
                    """)
                    
        except Exception as e:
            st.error(f"Error showing analysis: {str(e)}")
    
    def render_page(self):
        """Render the live monitoring page"""
        st.title("Live Camera & Alerts")

        # Show total alerts metric at the top
        total_alerts = 0
        if 'alerts' in st.session_state:
            total_alerts = len([a for a in st.session_state.alerts if a['risk_level'] in ['High', 'Medium']])
        st.metric("Total Alerts (High/Medium)", total_alerts)

        # Initialize camera
        self.setup_camera()

        # Initialize upload tracking in session state
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None

        # Camera controls
        st.sidebar.header("Camera Controls")
        camera_options = {
            "Live Camera": 0,
            "Drone Feed": 1
        }

        # Volume control
        volume = st.sidebar.slider("Alert Volume", 0.0, 1.0, 0.9, 0.1)
        if self.voice_engine:
            self.voice_engine.setProperty('volume', volume)

        # Language selection
        languages = {
            "English": "english",
            "Spanish": "spanish",
            "French": "french"
        }
        selected_language = st.sidebar.selectbox("Alert Language", list(languages.keys()))
        selected_source = st.sidebar.selectbox("Camera Source", list(camera_options.keys()))

        def handle_frame(frame, mode_caption):
            st.image(frame, channels="BGR", caption=mode_caption)
            processed_frame, probability, risk_level, location, time_to_impact = self.process_frame(frame)
            self.show_analysis(probability, risk_level, location, time_to_impact)
            st.subheader("Evacuation Map")
            self.render_evacuation_map(location, risk_level, time_to_impact)
            # Voice navigation button for high/medium risk
            if risk_level in ["High", "Medium"]:
                if st.button("Start Voice Navigation", key=f"voice_nav_{mode_caption}"):
                    nav_text = f"Please follow the green route on the map to the nearest safe zone. Evacuate immediately via Route A or B as shown."
                    self.speak_alert(nav_text)

            # Upload Image option and logic removed
        if st.sidebar.button("Toggle Camera"):
            st.session_state.camera_active = not st.session_state.camera_active
            st.session_state.camera_source = camera_options[selected_source]
        if st.session_state.camera_active:
            try:
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                handle_frame(frame, "Live Feed")
            except Exception as e:
                st.error(f"Error accessing camera: {str(e)}")
                st.session_state.camera_active = False
        else:
            st.info("Camera is currently inactive. Click 'Toggle Camera' to start.")