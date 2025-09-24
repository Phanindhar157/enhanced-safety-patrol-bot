#!/usr/bin/env python3

"""
Dataset Setup and Organization Script
Organizes your existing datasets and creates synthetic data for missing components
"""

import os
import shutil
import numpy as np
import pandas as pd
import json
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
import random

def create_dataset_structure():
    """Create the complete dataset directory structure"""
    directories = [
        # Fire detection datasets
        'datasets/fire_detection/fire',
        'datasets/fire_detection/no_fire',
        'datasets/fire_detection/thermal',
        
        # Gas detection datasets
        'datasets/gas_detection/raw_data',
        'datasets/gas_detection/processed_data',
        
        # PPE compliance datasets (from PASCAL VOC)
        'datasets/ppe_compliance/helmet',
        'datasets/ppe_compliance/no_helmet',
        'datasets/ppe_compliance/glasses',
        'datasets/ppe_compliance/no_glasses',
        'datasets/ppe_compliance/vest',
        'datasets/ppe_compliance/no_vest',
        'datasets/ppe_compliance/person',
        
        # Structural health datasets (synthetic)
        'datasets/structural_health/lidar_data',
        'datasets/structural_health/vibration_data',
        'datasets/structural_health/combined_data',
        
        # Object detection datasets (from PASCAL VOC)
        'datasets/object_detection/obstacles',
        'datasets/object_detection/equipment',
        'datasets/object_detection/people',
        
        # Synthetic datasets
        'datasets/synthetic/line_following',
        'datasets/synthetic/navigation',
        'datasets/synthetic/emergency_scenarios'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def setup_fire_dataset():
    """Setup fire detection dataset"""
    print("\n🔥 Setting up Fire Detection Dataset...")
    
    fire_dir = 'datasets/fire_detection'
    
    # Instructions for user
    instructions = """
    FIRE DATASET SETUP INSTRUCTIONS:
    
    1. Place your fire images in: datasets/fire_detection/fire/
    2. Place your non-fire images in: datasets/fire_detection/no_fire/
    3. If you have thermal images, place them in: datasets/fire_detection/thermal/
    
    Expected structure:
    datasets/fire_detection/
    ├── fire/              # Fire images (JPG, PNG)
    ├── no_fire/           # Non-fire images (JPG, PNG)
    └── thermal/           # Thermal images (optional)
    
    The system will automatically:
    - Validate image formats
    - Create train/validation/test splits
    - Generate thermal data if not available
    """
    
    with open(f'{fire_dir}/SETUP_INSTRUCTIONS.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Fire dataset structure created")
    print("📋 Please follow the instructions in datasets/fire_detection/SETUP_INSTRUCTIONS.txt")

def setup_gas_dataset():
    """Setup gas detection dataset"""
    print("\n⛽ Setting up Gas Detection Dataset...")
    
    gas_dir = 'datasets/gas_detection'
    
    # Instructions for user
    instructions = """
    GAS SENSOR ARRAY DRIFT DATASET SETUP INSTRUCTIONS:
    
    1. Place your Gas Sensor Array Drift Dataset files in: datasets/gas_detection/raw_data/
    
    Expected files:
    - Gas sensor data files (CSV, TXT, or other formats)
    - Any documentation or metadata files
    
    The system will automatically:
    - Parse the gas sensor data
    - Create standardized CSV format
    - Generate synthetic data for missing gas types
    - Create train/validation/test splits
    """
    
    with open(f'{gas_dir}/SETUP_INSTRUCTIONS.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ Gas dataset structure created")
    print("📋 Please follow the instructions in datasets/gas_detection/SETUP_INSTRUCTIONS.txt")

def setup_pascal_voc_dataset():
    """Setup PASCAL VOC dataset for PPE compliance and object detection"""
    print("\n🦺 Setting up PASCAL VOC Dataset for PPE Compliance...")
    
    ppe_dir = 'datasets/ppe_compliance'
    obj_dir = 'datasets/object_detection'
    
    # Instructions for user
    instructions = """
    PASCAL VOC 2012 DATASET SETUP INSTRUCTIONS:
    
    IMPORTANT: PASCAL VOC 2012 has a specific structure that needs to be processed!
    
    Dataset Structure:
    pascal_voc_2012/
    ├── JPEGImages/          # All images (.jpg)
    ├── Annotations/         # XML annotation files
    ├── ImageSets/Main/      # Train/val/test splits
    └── SegmentationClass/   # Segmentation masks
    
    AUTOMATIC PROCESSING (RECOMMENDED):
    1. Download PASCAL VOC 2012 from: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset
    2. Extract the dataset to a folder (e.g., "pascal_voc_2012")
    3. Run: python datasets/process_pascal_voc.py
    4. Follow the prompts to specify paths
    
    MANUAL PROCESSING (Alternative):
    1. Extract PASCAL VOC 2012 dataset
    2. Use the person annotations to identify person images
    3. Manually categorize images for PPE compliance:
       - Person images with helmets → datasets/ppe_compliance/helmet/
       - Person images without helmets → datasets/ppe_compliance/no_helmet/
       - Person images with glasses → datasets/ppe_compliance/glasses/
       - Person images without glasses → datasets/ppe_compliance/no_glasses/
       - Person images with safety vests → datasets/ppe_compliance/vest/
       - Person images without safety vests → datasets/ppe_compliance/no_vest/
       - All person images → datasets/ppe_compliance/person/
    
    For Object Detection:
    - Vehicle images → datasets/object_detection/vehicles/
    - Equipment images → datasets/object_detection/equipment/
    - Person images → datasets/object_detection/people/
    - Animal images → datasets/object_detection/animals/
    - Obstacle images → datasets/object_detection/obstacles/
    
    NOTE: PPE detection from PASCAL VOC is limited. For better results, 
    consider using specialized safety equipment datasets or manual annotation.
    """
    
    with open(f'{ppe_dir}/SETUP_INSTRUCTIONS.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    with open(f'{obj_dir}/SETUP_INSTRUCTIONS.txt', 'w', encoding='utf-8') as f:
        f.write(instructions)
    
    print("✅ PASCAL VOC dataset structure created")
    print("📋 Please follow the instructions in datasets/ppe_compliance/SETUP_INSTRUCTIONS.txt")
    print("🔧 For automatic processing, run: python datasets/process_pascal_voc.py")

def create_synthetic_structural_health_data():
    """Create synthetic structural health monitoring data"""
    print("\n🏗️ Creating Synthetic Structural Health Data...")
    
    # Create synthetic LiDAR data
    create_synthetic_lidar_data()
    
    # Create synthetic vibration data
    create_synthetic_vibration_data()
    
    # Create combined structural health data
    create_combined_structural_data()
    
    print("✅ Synthetic structural health data created")

def create_synthetic_lidar_data():
    """Create synthetic LiDAR point cloud data for structural analysis"""
    lidar_dir = 'datasets/structural_health/lidar_data'
    
    # Generate normal structural point clouds
    for i in range(100):
        # Normal structure - smooth surfaces
        points = generate_normal_structure_points()
        np.save(f'{lidar_dir}/normal_structure_{i:03d}.npy', points)
    
    # Generate damaged structural point clouds
    for i in range(100):
        # Damaged structure - irregular surfaces, cracks
        points = generate_damaged_structure_points()
        np.save(f'{lidar_dir}/damaged_structure_{i:03d}.npy', points)
    
    # Create metadata
    metadata = {
        "description": "Synthetic LiDAR point cloud data for structural health monitoring",
        "normal_samples": 100,
        "damaged_samples": 100,
        "point_cloud_size": 1024,
        "features": ["x", "y", "z", "intensity", "normal_x", "normal_y", "normal_z"],
        "damage_types": ["cracks", "corrosion", "deformation", "missing_parts"]
    }
    
    with open(f'{lidar_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created 200 synthetic LiDAR point clouds in {lidar_dir}")

def generate_normal_structure_points():
    """Generate normal structure point cloud"""
    n_points = 1024
    
    # Generate points on smooth surfaces
    points = []
    for _ in range(n_points):
        # Random point on a smooth surface
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z = np.random.uniform(0, 10)
        
        # Add small random noise for realism
        noise = np.random.normal(0, 0.01, 3)
        point = [x + noise[0], y + noise[1], z + noise[2]]
        
        # Add intensity and normal vectors
        intensity = np.random.uniform(0.3, 0.8)
        normal = [0, 0, 1]  # Mostly vertical normals for smooth surfaces
        
        points.append(point + [intensity] + normal)
    
    return np.array(points)

def generate_damaged_structure_points():
    """Generate damaged structure point cloud"""
    n_points = 1024
    
    points = []
    for _ in range(n_points):
        # Random point with potential damage
        x = np.random.uniform(-5, 5)
        y = np.random.uniform(-5, 5)
        z = np.random.uniform(0, 10)
        
        # Add damage patterns
        damage_type = np.random.choice(['crack', 'corrosion', 'deformation', 'normal'])
        
        if damage_type == 'crack':
            # Crack pattern - irregular z values
            z += np.random.normal(0, 0.5)
        elif damage_type == 'corrosion':
            # Corrosion - lower intensity
            intensity = np.random.uniform(0.1, 0.4)
        elif damage_type == 'deformation':
            # Deformation - irregular surface
            z += np.random.normal(0, 0.3)
            x += np.random.normal(0, 0.2)
            y += np.random.normal(0, 0.2)
        
        # Add noise
        noise = np.random.normal(0, 0.05, 3)
        point = [x + noise[0], y + noise[1], z + noise[2]]
        
        # Add intensity and normal vectors
        if damage_type == 'corrosion':
            intensity = np.random.uniform(0.1, 0.4)
        else:
            intensity = np.random.uniform(0.3, 0.8)
        
        # Irregular normals for damaged surfaces
        normal = [np.random.normal(0, 0.3), np.random.normal(0, 0.3), np.random.uniform(0.5, 1.0)]
        normal = np.array(normal) / np.linalg.norm(normal)  # Normalize
        
        points.append(point + [intensity] + list(normal))
    
    return np.array(points)

def create_synthetic_vibration_data():
    """Create synthetic vibration data for structural health monitoring"""
    vibration_dir = 'datasets/structural_health/vibration_data'
    
    # Generate normal vibration patterns
    for i in range(100):
        vibration_data = generate_normal_vibration()
        np.save(f'{vibration_dir}/normal_vibration_{i:03d}.npy', vibration_data)
    
    # Generate damaged vibration patterns
    for i in range(100):
        vibration_data = generate_damaged_vibration()
        np.save(f'{vibration_dir}/damaged_vibration_{i:03d}.npy', vibration_data)
    
    # Create metadata
    metadata = {
        "description": "Synthetic vibration data for structural health monitoring",
        "normal_samples": 100,
        "damaged_samples": 100,
        "sampling_rate": 1000,  # Hz
        "duration": 10,  # seconds
        "features": ["x_acceleration", "y_acceleration", "z_acceleration", "magnitude", "frequency_domain"],
        "damage_indicators": ["increased_amplitude", "frequency_shift", "harmonic_distortion"]
    }
    
    with open(f'{vibration_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created 200 synthetic vibration datasets in {vibration_dir}")

def generate_normal_vibration():
    """Generate normal vibration pattern"""
    sampling_rate = 1000  # Hz
    duration = 10  # seconds
    n_samples = sampling_rate * duration
    
    t = np.linspace(0, duration, n_samples)
    
    # Normal vibration - low amplitude, stable frequency
    frequency = 10  # Hz
    amplitude = 0.1
    
    x_acc = amplitude * np.sin(2 * np.pi * frequency * t) + np.random.normal(0, 0.01, n_samples)
    y_acc = amplitude * np.sin(2 * np.pi * frequency * t + np.pi/4) + np.random.normal(0, 0.01, n_samples)
    z_acc = amplitude * np.sin(2 * np.pi * frequency * t + np.pi/2) + np.random.normal(0, 0.01, n_samples)
    
    # Calculate magnitude
    magnitude = np.sqrt(x_acc**2 + y_acc**2 + z_acc**2)
    
    return np.column_stack([x_acc, y_acc, z_acc, magnitude])

def generate_damaged_vibration():
    """Generate damaged vibration pattern"""
    sampling_rate = 1000  # Hz
    duration = 10  # seconds
    n_samples = sampling_rate * duration
    
    t = np.linspace(0, duration, n_samples)
    
    # Damaged vibration - higher amplitude, frequency shift, harmonics
    base_frequency = 10  # Hz
    amplitude = 0.3  # Higher amplitude
    
    # Primary frequency with some shift
    freq_shift = np.random.uniform(0.8, 1.2)
    primary_freq = base_frequency * freq_shift
    
    x_acc = amplitude * np.sin(2 * np.pi * primary_freq * t) + np.random.normal(0, 0.05, n_samples)
    y_acc = amplitude * np.sin(2 * np.pi * primary_freq * t + np.pi/4) + np.random.normal(0, 0.05, n_samples)
    z_acc = amplitude * np.sin(2 * np.pi * primary_freq * t + np.pi/2) + np.random.normal(0, 0.05, n_samples)
    
    # Add harmonics (damage indicator)
    harmonic_amplitude = amplitude * 0.3
    x_acc += harmonic_amplitude * np.sin(2 * np.pi * primary_freq * 2 * t)
    y_acc += harmonic_amplitude * np.sin(2 * np.pi * primary_freq * 2 * t + np.pi/4)
    z_acc += harmonic_amplitude * np.sin(2 * np.pi * primary_freq * 2 * t + np.pi/2)
    
    # Calculate magnitude
    magnitude = np.sqrt(x_acc**2 + y_acc**2 + z_acc**2)
    
    return np.column_stack([x_acc, y_acc, z_acc, magnitude])

def create_combined_structural_data():
    """Create combined LiDAR and vibration data for structural health monitoring"""
    combined_dir = 'datasets/structural_health/combined_data'
    
    # Load LiDAR and vibration data
    lidar_dir = 'datasets/structural_health/lidar_data'
    vibration_dir = 'datasets/structural_health/vibration_data'
    
    combined_data = []
    
    # Combine normal data
    for i in range(100):
        lidar_data = np.load(f'{lidar_dir}/normal_structure_{i:03d}.npy')
        vibration_data = np.load(f'{vibration_dir}/normal_vibration_{i:03d}.npy')
        
        # Extract features
        lidar_features = extract_lidar_features(lidar_data)
        vibration_features = extract_vibration_features(vibration_data)
        
        combined_features = np.concatenate([lidar_features, vibration_features])
        combined_data.append({
            'features': combined_features.tolist(),
            'label': 0,  # Normal
            'lidar_file': f'normal_structure_{i:03d}.npy',
            'vibration_file': f'normal_vibration_{i:03d}.npy'
        })
    
    # Combine damaged data
    for i in range(100):
        lidar_data = np.load(f'{lidar_dir}/damaged_structure_{i:03d}.npy')
        vibration_data = np.load(f'{vibration_dir}/damaged_vibration_{i:03d}.npy')
        
        # Extract features
        lidar_features = extract_lidar_features(lidar_data)
        vibration_features = extract_vibration_features(vibration_data)
        
        combined_features = np.concatenate([lidar_features, vibration_features])
        combined_data.append({
            'features': combined_features.tolist(),
            'label': 1,  # Damaged
            'lidar_file': f'damaged_structure_{i:03d}.npy',
            'vibration_file': f'damaged_vibration_{i:03d}.npy'
        })
    
    # Save combined data
    np.save(f'{combined_dir}/combined_features.npy', [item['features'] for item in combined_data])
    np.save(f'{combined_dir}/combined_labels.npy', [item['label'] for item in combined_data])
    
    # Save metadata
    with open(f'{combined_dir}/combined_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"✅ Created combined structural health data in {combined_dir}")

def extract_lidar_features(lidar_data):
    """Extract features from LiDAR point cloud"""
    # Statistical features
    mean_intensity = np.mean(lidar_data[:, 3])
    std_intensity = np.std(lidar_data[:, 3])
    mean_z = np.mean(lidar_data[:, 2])
    std_z = np.std(lidar_data[:, 2])
    
    # Surface roughness (standard deviation of z values)
    surface_roughness = np.std(lidar_data[:, 2])
    
    # Normal vector statistics
    normal_std = np.std(lidar_data[:, 4:7], axis=0)
    
    return np.concatenate([
        [mean_intensity, std_intensity, mean_z, std_z, surface_roughness],
        normal_std
    ])

def extract_vibration_features(vibration_data):
    """Extract features from vibration data"""
    x_acc, y_acc, z_acc, magnitude = vibration_data.T
    
    # Statistical features
    mean_magnitude = np.mean(magnitude)
    std_magnitude = np.std(magnitude)
    max_magnitude = np.max(magnitude)
    
    # Frequency domain features (simplified)
    fft_magnitude = np.fft.fft(magnitude)
    dominant_frequency = np.argmax(np.abs(fft_magnitude[:len(fft_magnitude)//2]))
    
    return np.array([mean_magnitude, std_magnitude, max_magnitude, dominant_frequency])

def create_synthetic_line_following_data():
    """Create synthetic line following data"""
    print("\n🛤️ Creating Synthetic Line Following Data...")
    
    line_dir = 'datasets/synthetic/line_following'
    
    # Generate line following scenarios
    scenarios = [
        'straight_line',
        'curved_line',
        'intersection',
        'broken_line',
        'multiple_lines'
    ]
    
    for scenario in scenarios:
        scenario_dir = f'{line_dir}/{scenario}'
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Generate 50 samples for each scenario
        for i in range(50):
            image = generate_line_following_image(scenario)
            image.save(f'{scenario_dir}/line_{i:03d}.png')
    
    print(f"✅ Created synthetic line following data in {line_dir}")

def generate_line_following_image(scenario):
    """Generate synthetic line following image"""
    width, height = 64, 64
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    if scenario == 'straight_line':
        # Draw straight line
        draw.line([(10, 32), (54, 32)], fill='black', width=3)
    elif scenario == 'curved_line':
        # Draw curved line
        points = [(10, 32), (20, 20), (30, 15), (40, 20), (50, 32), (54, 45)]
        draw.line(points, fill='black', width=3)
    elif scenario == 'intersection':
        # Draw intersection
        draw.line([(10, 32), (54, 32)], fill='black', width=3)
        draw.line([(32, 10), (32, 54)], fill='black', width=3)
    elif scenario == 'broken_line':
        # Draw broken line
        for i in range(0, 44, 8):
            draw.line([(10+i, 32), (14+i, 32)], fill='black', width=3)
    elif scenario == 'multiple_lines':
        # Draw multiple lines
        draw.line([(10, 20), (54, 20)], fill='black', width=2)
        draw.line([(10, 32), (54, 32)], fill='black', width=2)
        draw.line([(10, 44), (54, 44)], fill='black', width=2)
    
    return image

def create_dataset_info():
    """Create comprehensive dataset information file"""
    info = {
        "dataset_overview": {
            "fire_detection": {
                "description": "Fire and non-fire images for fire detection training",
                "location": "datasets/fire_detection/",
                "format": "Images (JPG, PNG)",
                "status": "Ready for user data"
            },
            "gas_detection": {
                "description": "Gas sensor array drift dataset for gas detection",
                "location": "datasets/gas_detection/",
                "format": "CSV, TXT files",
                "status": "Ready for user data"
            },
            "ppe_compliance": {
                "description": "PPE compliance detection from PASCAL VOC dataset",
                "location": "datasets/ppe_compliance/",
                "format": "Images with annotations",
                "status": "Ready for user data"
            },
            "structural_health": {
                "description": "Synthetic LiDAR and vibration data for structural health monitoring",
                "location": "datasets/structural_health/",
                "format": "NumPy arrays, JSON metadata",
                "status": "Generated"
            },
            "object_detection": {
                "description": "Object detection from PASCAL VOC dataset",
                "location": "datasets/object_detection/",
                "format": "Images with annotations",
                "status": "Ready for user data"
            },
            "line_following": {
                "description": "Synthetic line following scenarios",
                "location": "datasets/synthetic/line_following/",
                "format": "Images",
                "status": "Generated"
            }
        },
        "training_instructions": {
            "step_1": "Place your datasets in the appropriate directories",
            "step_2": "Run: python ml_models/train_advanced_models.py",
            "step_3": "Models will be saved in models/ directory",
            "step_4": "Use trained models in Webots simulation"
        },
        "dataset_sizes": {
            "fire_detection": "User provided",
            "gas_detection": "User provided", 
            "ppe_compliance": "User provided",
            "structural_health": "200 samples (100 normal, 100 damaged)",
            "object_detection": "User provided",
            "line_following": "250 samples (50 per scenario)"
        }
    }
    
    with open('datasets/DATASET_INFO.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2)
    
    print("📋 Dataset information saved to datasets/DATASET_INFO.json")

def main():
    """Main setup function"""
    print("🚀 Enhanced Safety Patrol Bot - Dataset Setup")
    print("=" * 60)
    print("Setting up your datasets and creating synthetic data...")
    print()
    
    # Create directory structure
    create_dataset_structure()
    
    # Setup your existing datasets
    setup_fire_dataset()
    setup_gas_dataset()
    setup_pascal_voc_dataset()
    
    # Create synthetic data
    create_synthetic_structural_health_data()
    create_synthetic_line_following_data()
    
    # Create dataset information
    create_dataset_info()
    
    print("\n✅ Dataset setup completed successfully!")
    print("=" * 60)
    print("📋 Next Steps:")
    print("1. Place your fire dataset images in: datasets/fire_detection/fire/ and datasets/fire_detection/no_fire/")
    print("2. Place your gas sensor data in: datasets/gas_detection/raw_data/")
    print("3. Extract PASCAL VOC dataset and organize PPE images in: datasets/ppe_compliance/")
    print("4. Run training: python ml_models/train_advanced_models.py")
    print("5. Start Webots simulation with trained models")
    print()
    print("📊 Synthetic data created:")
    print("   ✅ 200 structural health samples (LiDAR + vibration)")
    print("   ✅ 250 line following scenarios")
    print("   ✅ Combined structural health features")
    print()
    print("📖 See datasets/DATASET_INFO.json for complete information")

if __name__ == "__main__":
    main()
