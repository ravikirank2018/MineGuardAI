import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFilter
import os
import random
from datetime import datetime, timedelta
from faker import Faker
import joblib
from config import *

fake = Faker()

class RockfallDataGenerator:
    """Generate synthetic data for rockfall prediction including DEM images and tabular data"""
    
    def __init__(self, dataset_size=5000, image_size=(64, 64)):
        self.dataset_size = dataset_size
        self.image_size = image_size
        self.images = []
        self.tabular_data = []
        self.labels = []
        
    def generate_dem_image(self, risk_level='Low'):
        """Generate synthetic Digital Elevation Model (DEM) image"""
        
        width, height = self.image_size
        
        # Base elevation map using noise
        base_elevation = np.random.rand(height, width) * 255
        
        # Apply Gaussian blur for realistic terrain
        base_elevation = cv2.GaussianBlur(base_elevation.astype(np.uint8), (5, 5), 0)
        
        # Add terrain features based on risk level
        if risk_level == 'High':
            # Add steep slopes and fractures
            for _ in range(random.randint(5, 10)):
                # Random fracture lines
                x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
                x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
                cv2.line(base_elevation, (x1, y1), (x2, y2), 0, thickness=random.randint(1, 3))
            
            # Add unstable rock formations
            for _ in range(random.randint(3, 6)):
                center = (random.randint(10, width-10), random.randint(10, height-10))
                radius = random.randint(3, 8)
                cv2.circle(base_elevation, center, radius, random.randint(200, 255), -1)
                
        elif risk_level == 'Medium':
            # Moderate terrain variation
            for _ in range(random.randint(2, 5)):
                x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
                x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
                cv2.line(base_elevation, (x1, y1), (x2, y2), random.randint(50, 150), thickness=1)
        
        else:  # Low risk
            # Stable, gradual terrain
            base_elevation = cv2.GaussianBlur(base_elevation, (7, 7), 0)
        
        # Convert to 3-channel image (RGB)
        dem_image = cv2.cvtColor(base_elevation, cv2.COLOR_GRAY2RGB)
        
        # Add some color variation to simulate different rock types
        if risk_level == 'High':
            # Reddish tint for unstable areas
            dem_image[:, :, 0] = np.minimum(dem_image[:, :, 0] + 30, 255)
        elif risk_level == 'Medium':
            # Yellowish tint
            dem_image[:, :, 0] = np.minimum(dem_image[:, :, 0] + 15, 255)
            dem_image[:, :, 1] = np.minimum(dem_image[:, :, 1] + 15, 255)
        
        return dem_image
    
    def generate_tabular_features(self, risk_level='Low'):
        """Generate synthetic tabular data for rockfall prediction with comprehensive parameters"""
        
        # 1. Slope Geometry Parameters
        if risk_level == 'High':
            slope_height = np.random.uniform(50, 100)        # Higher slopes (meters)
            slope_angle = np.random.uniform(60, 85)          # Steeper slopes (degrees)
            slope_length = np.random.uniform(100, 200)       # Longer slopes (meters)
            surface_roughness = np.random.uniform(0.1, 0.3)  # Low roughness (dimensionless)
            # 1=Convex, 2=Planar, 3=Concave, 4=Stepped
            slope_profile = np.random.choice([1, 4], p=[0.7, 0.3])  # Mostly convex or stepped
        elif risk_level == 'Medium':
            slope_height = np.random.uniform(30, 60)
            slope_angle = np.random.uniform(40, 65)
            slope_length = np.random.uniform(60, 120)
            surface_roughness = np.random.uniform(0.3, 0.6)
            slope_profile = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
        else:  # Low risk
            slope_height = np.random.uniform(10, 35)
            slope_angle = np.random.uniform(20, 45)
            slope_length = np.random.uniform(30, 70)
            surface_roughness = np.random.uniform(0.5, 0.8)
            slope_profile = np.random.choice([2, 3], p=[0.6, 0.4])  # Mostly planar or concave
        
        # 2. Rock/Block Parameters
        if risk_level == 'High':
            block_size = np.random.uniform(1.0, 5.0)         # Large blocks (m³)
            block_mass = block_size * np.random.uniform(2500, 3000)  # Mass (kg)
            # 1=Angular, 2=Round, 3=Irregular
            block_shape = np.random.choice([1, 3], p=[0.7, 0.3])  # Mostly angular
            material_strength = np.random.uniform(20, 50)     # Low strength (MPa)
            restitution_normal = np.random.uniform(0.5, 0.8)  # High bounce
            restitution_tangential = np.random.uniform(0.7, 0.9)
            friction_coefficient = np.random.uniform(0.1, 0.3)  # Low friction
        elif risk_level == 'Medium':
            block_size = np.random.uniform(0.5, 2.0)
            block_mass = block_size * np.random.uniform(2300, 2800)
            block_shape = np.random.choice([1, 2, 3], p=[0.4, 0.2, 0.4])
            material_strength = np.random.uniform(40, 80)
            restitution_normal = np.random.uniform(0.3, 0.6)
            restitution_tangential = np.random.uniform(0.5, 0.8)
            friction_coefficient = np.random.uniform(0.3, 0.5)
        else:  # Low risk
            block_size = np.random.uniform(0.1, 0.8)
            block_mass = block_size * np.random.uniform(2200, 2600)
            block_shape = np.random.choice([2, 3], p=[0.6, 0.4])  # Mostly round
            material_strength = np.random.uniform(70, 120)
            restitution_normal = np.random.uniform(0.1, 0.4)
            restitution_tangential = np.random.uniform(0.3, 0.6)
            friction_coefficient = np.random.uniform(0.5, 0.8)  # High friction
        
        # 3. Environmental & Triggering Parameters
        if risk_level == 'High':
            moisture_content = np.random.uniform(20, 40)      # High moisture (%)
            freeze_thaw_cycles = np.random.randint(15, 30)    # Many cycles per year
            vegetation_cover = np.random.uniform(0, 20)       # Low vegetation (%)
            temperature_variation = np.random.uniform(15, 30) # High variation (°C)
            seismic_activity = np.random.uniform(0.5, 1.0)    # High activity (scaled 0-1)
        elif risk_level == 'Medium':
            moisture_content = np.random.uniform(10, 25)
            freeze_thaw_cycles = np.random.randint(5, 20)
            vegetation_cover = np.random.uniform(20, 50)
            temperature_variation = np.random.uniform(8, 18)
            seismic_activity = np.random.uniform(0.2, 0.6)
        else:  # Low risk
            moisture_content = np.random.uniform(0, 15)
            freeze_thaw_cycles = np.random.randint(0, 10)
            vegetation_cover = np.random.uniform(50, 90)
            temperature_variation = np.random.uniform(3, 10)
            seismic_activity = np.random.uniform(0, 0.3)
        
        # 4. Initial Conditions
        if risk_level == 'High':
            release_point_height = np.random.uniform(0.8, 1.0)  # Relative to slope height (0-1)
            initial_velocity = np.random.uniform(2.0, 5.0)      # m/s
            initial_angle = np.random.uniform(10, 30)           # degrees from horizontal
        elif risk_level == 'Medium':
            release_point_height = np.random.uniform(0.5, 0.8)
            initial_velocity = np.random.uniform(1.0, 3.0)
            initial_angle = np.random.uniform(5, 15)
        else:  # Low risk
            release_point_height = np.random.uniform(0.2, 0.5)
            initial_velocity = np.random.uniform(0.0, 1.0)
            initial_angle = np.random.uniform(0, 10)
        
        # 5. Simulation & Output Parameters
        if risk_level == 'High':
            runout_distance = np.random.uniform(100, 200)     # meters
            impact_velocity = np.random.uniform(15, 30)       # m/s
            kinetic_energy = 0.5 * block_mass * (impact_velocity ** 2)  # Joules
            bounce_height = np.random.uniform(5, 15)          # meters
            lateral_dispersion = np.random.uniform(10, 30)    # meters
            num_bounces = np.random.randint(3, 8)
        elif risk_level == 'Medium':
            runout_distance = np.random.uniform(50, 120)
            impact_velocity = np.random.uniform(8, 18)
            kinetic_energy = 0.5 * block_mass * (impact_velocity ** 2)
            bounce_height = np.random.uniform(2, 8)
            lateral_dispersion = np.random.uniform(5, 15)
            num_bounces = np.random.randint(5, 12)
        else:  # Low risk
            runout_distance = np.random.uniform(10, 60)
            impact_velocity = np.random.uniform(3, 10)
            kinetic_energy = 0.5 * block_mass * (impact_velocity ** 2)
            bounce_height = np.random.uniform(1, 4)
            lateral_dispersion = np.random.uniform(2, 8)
            num_bounces = np.random.randint(8, 15)
        
        # 6. Historical & Statistical Data
        if risk_level == 'High':
            rockfall_frequency = np.random.uniform(5, 10)     # events per year
            block_volume_distribution = np.random.uniform(0.7, 1.0)  # skewness toward larger blocks
            historical_runout = np.random.uniform(0.8, 1.0)   # relative to predicted runout
        elif risk_level == 'Medium':
            rockfall_frequency = np.random.uniform(2, 6)
            block_volume_distribution = np.random.uniform(0.4, 0.7)
            historical_runout = np.random.uniform(0.6, 0.9)
        else:  # Low risk
            rockfall_frequency = np.random.uniform(0.5, 3)
            block_volume_distribution = np.random.uniform(0.1, 0.5)
            historical_runout = np.random.uniform(0.3, 0.7)
        
        return [
            # 1. Slope Geometry Parameters
            slope_height, slope_angle, slope_length, surface_roughness, slope_profile,
            # 2. Rock/Block Parameters
            block_size, block_mass, block_shape, material_strength, 
            restitution_normal, restitution_tangential, friction_coefficient,
            # 3. Environmental & Triggering Parameters
            moisture_content, freeze_thaw_cycles, vegetation_cover, 
            temperature_variation, seismic_activity,
            # 4. Initial Conditions
            release_point_height, initial_velocity, initial_angle,
            # 5. Simulation & Output Parameters
            runout_distance, impact_velocity, kinetic_energy, 
            bounce_height, lateral_dispersion, num_bounces,
            # 6. Historical & Statistical Data
            rockfall_frequency, block_volume_distribution, historical_runout
        ]
    
    def generate_single_sample(self):
        """Generate a single sample of image and tabular data"""
        
        # Determine risk level based on distribution
        risk_prob = np.random.random()
        if risk_prob < 0.7:
            risk_level = 'Low'
        elif risk_prob < 0.9:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Generate image and tabular data
        image = self.generate_dem_image(risk_level)
        tabular_features = self.generate_tabular_features(risk_level)
        
        return image, tabular_features, risk_level
    
    def generate_complete_dataset(self, progress_callback=None):
        """Generate complete dataset with images and tabular data"""
        
        self.images = []
        self.tabular_data = []
        self.labels = []
        
        feature_names = [
            # 1. Slope Geometry Parameters
            'slope_height', 'slope_angle', 'slope_length', 'surface_roughness', 'slope_profile',
            # 2. Rock/Block Parameters
            'block_size', 'block_mass', 'block_shape', 'material_strength',
            'restitution_normal', 'restitution_tangential', 'friction_coefficient',
            # 3. Environmental & Triggering Parameters
            'moisture_content', 'freeze_thaw_cycles', 'vegetation_cover',
            'temperature_variation', 'seismic_activity',
            # 4. Initial Conditions
            'release_point_height', 'initial_velocity', 'initial_angle',
            # 5. Simulation & Output Parameters
            'runout_distance', 'impact_velocity', 'kinetic_energy',
            'bounce_height', 'lateral_dispersion', 'num_bounces',
            # 6. Historical & Statistical Data
            'rockfall_frequency', 'block_volume_distribution', 'historical_runout'
        ]
        
        for i in range(self.dataset_size):
            if progress_callback:
                progress_callback(i + 1, self.dataset_size, "Generating samples")
            
            image, tabular_features, risk_level = self.generate_single_sample()
            
            self.images.append(image)
            self.tabular_data.append(tabular_features)
            self.labels.append(risk_level)
        
        # Convert to numpy arrays
        self.images = np.array(self.images)
        self.tabular_data = np.array(self.tabular_data)
        
        # Save data to files
        self.save_data(feature_names)
        
        return self.images, self.tabular_data, self.labels
    
    def save_data(self, feature_names):
        """Save generated data to files"""
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Save images
        np.save(IMAGE_DATA_PATH, self.images)
        
        # Save tabular data as CSV
        df = pd.DataFrame(self.tabular_data, columns=feature_names)
        df['risk_level'] = self.labels
        
        # Add probability scores based on risk level
        risk_to_prob = {'Low': 0.1, 'Medium': 0.5, 'High': 0.8}
        df['probability'] = df['risk_level'].map(risk_to_prob)
        df['probability'] += np.random.normal(0, 0.1, len(df))  # Add noise
        df['probability'] = np.clip(df['probability'], 0, 1)
        
        df.to_csv(TABULAR_DATA_PATH, index=False)
        
        print(f"Data saved: {len(self.images)} samples")
        print(f"Images saved to: {IMAGE_DATA_PATH}")
        print(f"Tabular data saved to: {TABULAR_DATA_PATH}")
    
    def load_data(self):
        """Load previously generated data"""
        
        if os.path.exists(IMAGE_DATA_PATH) and os.path.exists(TABULAR_DATA_PATH):
            self.images = np.load(IMAGE_DATA_PATH)
            df = pd.read_csv(TABULAR_DATA_PATH)
            
            # Extract features and labels
            feature_cols = [col for col in df.columns if col not in ['risk_level', 'probability']]
            self.tabular_data = df[feature_cols].values
            self.labels = df['risk_level'].tolist()
            
            return True
        return False
    
    def get_data_statistics(self):
        """Get statistics about the generated data"""
        
        if not self.labels:
            return None
        
        stats = {
            'total_samples': len(self.labels),
            'risk_distribution': pd.Series(self.labels).value_counts().to_dict(),
            'image_shape': self.images.shape if len(self.images) > 0 else None,
            'feature_count': len(self.tabular_data[0]) if len(self.tabular_data) > 0 else 0
        }
        
        return stats

# Utility functions for data generation
def create_synthetic_weather_pattern(days=30):
    """Create realistic weather pattern over time"""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Base seasonal pattern
    base_temp = 20 + 10 * np.sin(np.arange(days) * 2 * np.pi / 365)
    
    weather_data = []
    for i, date in enumerate(dates):
        temp_variation = np.random.normal(0, 3)
        rain_prob = 0.3 + 0.2 * np.sin(i * 2 * np.pi / 7)  # Weekly pattern
        
        weather = {
            'date': date,
            'temperature': base_temp[i] + temp_variation,
            'humidity': np.random.uniform(40, 90),
            'rainfall': np.random.exponential(5) if np.random.random() < rain_prob else 0,
            'wind_speed': np.random.gamma(2, 5),
            'pressure': np.random.normal(1013, 10)
        }
        weather_data.append(weather)
    
    return pd.DataFrame(weather_data)

def simulate_geological_survey_data():
    """Simulate geological survey data for the mine site"""
    
    # Grid points for the mine site
    x_coords = np.linspace(0, 1000, 20)  # 1km x 1km area
    y_coords = np.linspace(0, 1000, 20)
    
    survey_data = []
    
    for x in x_coords:
        for y in y_coords:
            # Simulate geological properties at each point
            survey_point = {
                'x_coord': x,
                'y_coord': y,
                'elevation': 100 + 50 * np.sin(x/200) + 30 * np.cos(y/150) + np.random.normal(0, 5),
                'rock_type': np.random.choice(['granite', 'sandstone', 'limestone', 'shale'], 
                                            p=[0.4, 0.3, 0.2, 0.1]),
                'fracture_orientation': np.random.uniform(0, 360),
                'joint_spacing': np.random.lognormal(0, 1),
                'weathering_grade': np.random.randint(1, 6),
                'discontinuity_roughness': np.random.uniform(0, 20)
            }
            survey_data.append(survey_point)
    
    return pd.DataFrame(survey_data)

if __name__ == "__main__":
    # Test data generation
    generator = RockfallDataGenerator(dataset_size=100, image_size=(64, 64))
    
    print("Generating test dataset...")
    images, tabular_data, labels = generator.generate_complete_dataset()
    
    print(f"Generated {len(images)} samples")
    print(f"Image shape: {images.shape}")
    print(f"Tabular data shape: {tabular_data.shape}")
    print(f"Risk distribution: {pd.Series(labels).value_counts()}")
    
    # Show sample
    print("\nSample tabular features:")
    print(tabular_data[0])
    print(f"Sample label: {labels[0]}")
