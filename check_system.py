#!/usr/bin/env python3

"""
System Status Checker for Enhanced Safety Patrol Bot
Verifies all components are working correctly
"""

import os
import sys
import requests
import json
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("🔍 Checking Required Files...")
    
    required_files = [
        "enhanced_safety_patrol_bot.wbt",
        "launch_enhanced_system.py",
        "web_interface/simple_app.py",
        "controllers/enhanced_patrol_controller/enhanced_patrol_controller.py",
        "emergency_response/emergency_controller.py",
        "ml_models/train_with_your_datasets.py",
        "models/fire_detection_model.pth",
        "models/gas_detection_model.joblib",
        "models/ppe_compliance_model.pth",
        "models/structural_health_model.joblib"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_dependencies():
    """Check if all Python dependencies are installed"""
    print("\n📦 Checking Python Dependencies...")
    
    required_packages = [
        "flask", "flask_cors", "torch", "torchvision", 
        "sklearn", "joblib", "numpy", "pandas", 
        "PIL", "cv2", "matplotlib", "requests"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "cv2":
                import cv2
            elif package == "sklearn":
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_web_interface():
    """Check if web interface is running"""
    print("\n🌐 Checking Web Interface...")
    
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        if response.status_code == 200:
            print("✅ Web interface is running at http://localhost:5000")
            return True
        else:
            print(f"❌ Web interface returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Web interface is not running: {e}")
        print("💡 To start web interface: python web_interface/simple_app.py")
        return False

def check_datasets():
    """Check if datasets are available"""
    print("\n📊 Checking Datasets...")
    
    dataset_dirs = [
        "datasets/fire_detection",
        "datasets/gas_detection", 
        "datasets/ppe_compliance",
        "datasets/structural_health",
        "datasets/object_detection"
    ]
    
    all_available = True
    for dataset_dir in dataset_dirs:
        if os.path.exists(dataset_dir):
            # Count files in directory
            file_count = 0
            for root, dirs, files in os.walk(dataset_dir):
                file_count += len(files)
            
            if file_count > 0:
                print(f"✅ {dataset_dir} ({file_count} files)")
            else:
                print(f"⚠️ {dataset_dir} (empty)")
                all_available = False
        else:
            print(f"❌ {dataset_dir} (not found)")
            all_available = False
    
    return all_available

def main():
    """Main system check"""
    print("🚀 Enhanced Safety Patrol Bot - System Status Check")
    print("=" * 60)
    
    # Run all checks
    files_ok = check_files()
    deps_ok = check_dependencies()
    web_ok = check_web_interface()
    datasets_ok = check_datasets()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 System Status Summary:")
    print(f"   Files: {'✅ OK' if files_ok else '❌ Issues'}")
    print(f"   Dependencies: {'✅ OK' if deps_ok else '❌ Issues'}")
    print(f"   Web Interface: {'✅ OK' if web_ok else '❌ Issues'}")
    print(f"   Datasets: {'✅ OK' if datasets_ok else '❌ Issues'}")
    
    if all([files_ok, deps_ok, web_ok, datasets_ok]):
        print("\n🎉 All systems are operational!")
        print("🚀 Your enhanced safety patrol bot is ready!")
        print("\n📋 Next steps:")
        print("   1. Open Webots and load enhanced_safety_patrol_bot.wbt")
        print("   2. Set controller to enhanced_patrol_controller")
        print("   3. Monitor via http://localhost:5000")
    else:
        print("\n⚠️ Some issues detected. Please fix them before proceeding.")
        
        if not deps_ok:
            print("\n💡 To install missing dependencies:")
            print("   pip install -r requirements.txt")
        
        if not web_ok:
            print("\n💡 To start web interface:")
            print("   python web_interface/simple_app.py")

if __name__ == "__main__":
    main()

