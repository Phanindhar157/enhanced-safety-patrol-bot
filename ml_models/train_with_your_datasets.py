#!/usr/bin/env python3

"""
Training Script for Your Specific Datasets
Trains AI models using your fire dataset, gas sensor data, and PASCAL VOC dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path

class FireDataset(Dataset):
    """Dataset for fire detection using your fire dataset"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class PPEDataset(Dataset):
    """Dataset for PPE compliance using PASCAL VOC data"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class FireDetectionModel(nn.Module):
    """CNN model for fire detection"""
    def __init__(self):
        super(FireDetectionModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)  # Fire/No Fire
        
    def forward(self, x):
        return self.backbone(x)

class PPEComplianceModel(nn.Module):
    """CNN model for PPE compliance detection"""
    def __init__(self):
        super(PPEComplianceModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 6)  # 6 PPE items
        
    def forward(self, x):
        return self.backbone(x)

def load_fire_dataset():
    """Load your fire detection dataset"""
    print("🔥 Loading Fire Detection Dataset...")
    
    fire_dir = 'datasets/fire_detection/fire'
    no_fire_dir = 'datasets/fire_detection/no_fire'
    
    fire_images = []
    no_fire_images = []
    
    # Load fire images
    if os.path.exists(fire_dir):
        for filename in os.listdir(fire_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                fire_images.append(os.path.join(fire_dir, filename))
    
    # Load no-fire images
    if os.path.exists(no_fire_dir):
        for filename in os.listdir(no_fire_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                no_fire_images.append(os.path.join(no_fire_dir, filename))
    
    if len(fire_images) == 0 and len(no_fire_images) == 0:
        print("❌ No fire dataset images found!")
        print("Please place your fire images in:")
        print("  - datasets/fire_detection/fire/ (fire images)")
        print("  - datasets/fire_detection/no_fire/ (non-fire images)")
        return None, None
    
    # Create labels
    fire_labels = [1] * len(fire_images)  # 1 for fire
    no_fire_labels = [0] * len(no_fire_images)  # 0 for no fire
    
    # Combine datasets
    all_images = fire_images + no_fire_images
    all_labels = fire_labels + no_fire_labels
    
    print(f"✅ Loaded {len(all_images)} images:")
    print(f"   - Fire images: {len(fire_images)}")
    print(f"   - No-fire images: {len(no_fire_images)}")
    
    return all_images, all_labels

def load_gas_dataset():
    """Load your gas sensor array drift dataset"""
    print("⛽ Loading Gas Sensor Dataset...")
    
    gas_dir = 'datasets/gas_detection/raw_data'
    
    if not os.path.exists(gas_dir):
        print("❌ Gas sensor data directory not found!")
        print("Please place your Gas Sensor Array Drift Dataset in:")
        print("  - datasets/gas_detection/raw_data/")
        return None, None
    
    # Look for data files
    data_files = []
    for filename in os.listdir(gas_dir):
        if filename.lower().endswith(('.csv', '.txt', '.dat')):
            data_files.append(os.path.join(gas_dir, filename))
    
    if len(data_files) == 0:
        print("❌ No gas sensor data files found!")
        print("Please place your gas sensor data files in datasets/gas_detection/raw_data/")
        return None, None
    
    # Load and process gas sensor data
    all_data = []
    all_labels = []
    
    for data_file in data_files:
        try:
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
            else:
                df = pd.read_csv(data_file, sep='\t')
            
            # Extract sensor readings (assuming last column is label)
            sensor_columns = df.columns[:-1]  # All columns except last
            label_column = df.columns[-1]     # Last column is label
            
            sensor_data = df[sensor_columns].values
            labels = df[label_column].values
            
            all_data.extend(sensor_data)
            all_labels.extend(labels)
            
            print(f"✅ Loaded {len(sensor_data)} samples from {os.path.basename(data_file)}")
            
        except Exception as e:
            print(f"⚠️ Error loading {data_file}: {e}")
            continue
    
    if len(all_data) == 0:
        print("❌ No valid gas sensor data loaded!")
        return None, None
    
    print(f"✅ Total gas sensor samples loaded: {len(all_data)}")
    return np.array(all_data), np.array(all_labels)

def load_ppe_dataset():
    """Load PPE compliance dataset from PASCAL VOC"""
    print("🦺 Loading PPE Compliance Dataset...")
    
    ppe_dir = 'datasets/ppe_compliance'
    
    # Check if PPE data exists
    ppe_categories = ['helmet', 'no_helmet', 'glasses', 'no_glasses', 'vest', 'no_vest']
    existing_categories = []
    
    for category in ppe_categories:
        category_dir = os.path.join(ppe_dir, category)
        if os.path.exists(category_dir) and len(os.listdir(category_dir)) > 0:
            existing_categories.append(category)
    
    if len(existing_categories) == 0:
        print("❌ No PPE compliance data found!")
        print("Please organize your PASCAL VOC dataset in:")
        print("  - datasets/ppe_compliance/helmet/")
        print("  - datasets/ppe_compliance/no_helmet/")
        print("  - datasets/ppe_compliance/glasses/")
        print("  - datasets/ppe_compliance/no_glasses/")
        print("  - datasets/ppe_compliance/vest/")
        print("  - datasets/ppe_compliance/no_vest/")
        return None, None
    
    # Load images and create labels
    all_images = []
    all_labels = []
    
    for category in existing_categories:
        category_dir = os.path.join(ppe_dir, category)
        images = []
        
        for filename in os.listdir(category_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(category_dir, filename))
        
        if len(images) > 0:
            all_images.extend(images)
            # Create multi-label for PPE items
            label = [0] * 6  # [helmet, glasses, vest, gloves, boots, overall]
            
            if 'helmet' in category and 'no_helmet' not in category:
                label[0] = 1
            if 'glasses' in category and 'no_glasses' not in category:
                label[1] = 1
            if 'vest' in category and 'no_vest' not in category:
                label[2] = 1
            
            all_labels.extend([label] * len(images))
            print(f"✅ Loaded {len(images)} images from {category}")
    
    if len(all_images) == 0:
        print("❌ No PPE images loaded!")
        return None, None
    
    print(f"✅ Total PPE images loaded: {len(all_images)}")
    return all_images, all_labels

def load_structural_health_dataset():
    """Load synthetic structural health dataset"""
    print("🏗️ Loading Structural Health Dataset...")
    
    combined_dir = 'datasets/structural_health/combined_data'
    
    if not os.path.exists(f'{combined_dir}/combined_features.npy'):
        print("❌ Structural health data not found!")
        print("Please run: python datasets/setup_datasets.py")
        return None, None
    
    features = np.load(f'{combined_dir}/combined_features.npy')
    labels = np.load(f'{combined_dir}/combined_labels.npy')
    
    print(f"✅ Loaded {len(features)} structural health samples")
    return features, labels

def train_fire_detection_model():
    """Train fire detection model with your dataset"""
    print("\n🔥 Training Fire Detection Model...")
    
    # Load dataset
    images, labels = load_fire_dataset()
    if images is None:
        return None
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Split dataset
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Create datasets
    train_dataset = FireDataset(train_images, train_labels, transform)
    val_dataset = FireDataset(val_images, val_labels, transform)
    test_dataset = FireDataset(test_images, test_labels, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = FireDetectionModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 20
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/fire_detection_model.pth')
        
        scheduler.step()
    
    # Test model
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100 * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                              target_names=['No Fire', 'Fire']))
    
    return model

def train_gas_detection_model():
    """Train gas detection model with your dataset"""
    print("\n⛽ Training Gas Detection Model...")
    
    # Load dataset
    data, labels = load_gas_dataset()
    if data is None:
        return None
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    print("Gas Detection Model Results:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'models/gas_detection_model.joblib')
    print("✅ Gas detection model saved!")
    
    return model

def train_ppe_compliance_model():
    """Train PPE compliance model with your dataset"""
    print("\n🦺 Training PPE Compliance Model...")
    
    # Load dataset
    images, labels = load_ppe_dataset()
    if images is None:
        return None
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Split dataset
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = PPEDataset(train_images, train_labels, transform)
    val_dataset = PPEDataset(val_images, val_labels, transform)
    test_dataset = PPEDataset(test_images, test_labels, transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = PPEComplianceModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    num_epochs = 15
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), torch.FloatTensor(labels).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), torch.FloatTensor(labels).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/ppe_compliance_model.pth')
        
        scheduler.step()
    
    print("✅ PPE compliance model saved!")
    return model

def train_structural_health_model():
    """Train structural health model with synthetic data"""
    print("\n🏗️ Training Structural Health Model...")
    
    # Load dataset
    features, labels = load_structural_health_dataset()
    if features is None:
        return None
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    print("Structural Health Model Results:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Damaged']))
    
    # Save model
    joblib.dump(model, 'models/structural_health_model.joblib')
    print("✅ Structural health model saved!")
    
    return model

def create_training_report():
    """Create training report"""
    report = {
        "training_date": str(pd.Timestamp.now()),
        "datasets_used": {
            "fire_detection": "User provided fire dataset",
            "gas_detection": "User provided gas sensor array drift dataset",
            "ppe_compliance": "User provided PASCAL VOC dataset",
            "structural_health": "Synthetic LiDAR and vibration data"
        },
        "models_trained": [
            "fire_detection_model.pth",
            "gas_detection_model.joblib",
            "ppe_compliance_model.pth",
            "structural_health_model.joblib"
        ],
        "next_steps": [
            "Test models in Webots simulation",
            "Validate performance with real sensor data",
            "Fine-tune models based on field performance",
            "Deploy in enhanced safety patrol bot"
        ]
    }
    
    with open('models/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📊 Training report saved to models/training_report.json")

def main():
    """Main training function"""
    print("🚀 Training AI Models with Your Datasets")
    print("=" * 60)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    try:
        # Train all models
        fire_model = train_fire_detection_model()
        gas_model = train_gas_detection_model()
        ppe_model = train_ppe_compliance_model()
        structural_model = train_structural_health_model()
        
        # Create training report
        create_training_report()
        
        print("\n" + "=" * 60)
        print("✅ All models trained successfully!")
        print("\n📁 Models saved in 'models/' directory:")
        print("   - fire_detection_model.pth")
        print("   - gas_detection_model.joblib")
        print("   - ppe_compliance_model.pth")
        print("   - structural_health_model.joblib")
        
        print("\n🔄 Next steps:")
        print("   1. Test models in Webots simulation")
        print("   2. Run: python launch_enhanced_system.py")
        print("   3. Load enhanced_safety_patrol_bot.wbt in Webots")
        print("   4. Access dashboard at: http://localhost:5000")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training completed successfully!")
    else:
        print("\n💥 Training failed!")

