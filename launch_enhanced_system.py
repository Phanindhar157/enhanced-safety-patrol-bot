#!/usr/bin/env python3

"""
Launch Enhanced Safety Patrol Bot System
Starts the complete system with Webots simulation and web interface
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_system_requirements():
    """Check if all system requirements are met"""
    print("🔍 Checking System Requirements...")
    
    requirements_met = True
    
    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ Models directory not found!")
        requirements_met = False
    else:
        model_files = [
            "fire_detection_model.pth",
            "gas_detection_model.joblib", 
            "ppe_compliance_model.pth",
            "structural_health_model.joblib"
        ]
        
        for model_file in model_files:
            model_path = models_dir / model_file
            if model_path.exists():
                print(f"✅ {model_file} found")
            else:
                print(f"❌ {model_file} not found")
                requirements_met = False
    
    # Check if Webots world exists
    world_file = Path("enhanced_safety_patrol_bot.wbt")
    if world_file.exists():
        print("✅ Enhanced Webots world found")
    else:
        print("❌ Enhanced Webots world not found")
        requirements_met = False
    
    # Check if web interface exists
    web_app = Path("web_interface/simple_app.py")
    if web_app.exists():
        print("✅ Web interface found")
    else:
        print("❌ Web interface not found")
        requirements_met = False
    
    return requirements_met

def start_web_interface():
    """Start the enhanced web interface"""
    print("\n🌐 Starting Enhanced Web Interface...")
    
    try:
        # Start Flask app in background
        web_app_path = "web_interface/simple_app.py"
        if os.path.exists(web_app_path):
            print("🚀 Launching web interface at http://localhost:5000")
            print("📊 Dashboard will open in your browser...")
            
            # Start the web server
            subprocess.Popen([sys.executable, web_app_path], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Open browser
            try:
                webbrowser.open("http://localhost:5000")
                print("✅ Web interface started successfully!")
                return True
            except Exception as e:
                print(f"⚠️ Could not open browser automatically: {e}")
                print("📱 Please manually open: http://localhost:5000")
                return True
        else:
            print("❌ Web interface not found!")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start web interface: {e}")
        return False

def show_webots_instructions():
    """Show instructions for starting Webots simulation"""
    print("\n🤖 Webots Simulation Instructions:")
    print("=" * 50)
    print("1. Open Webots R2023b or later")
    print("2. Load the world file: enhanced_safety_patrol_bot.wbt")
    print("3. Set the controller to: enhanced_patrol_controller")
    print("4. Click the 'Run' button to start simulation")
    print("5. The robot will begin autonomous patrol with AI models")
    print("\n📋 Controller Settings:")
    print("   - Controller: enhanced_patrol_controller")
    print("   - World: enhanced_safety_patrol_bot.wbt")
    print("   - AI Models: All 4 models loaded and ready")

def show_system_status():
    """Show current system status"""
    print("\n📊 Enhanced Safety Patrol Bot System Status:")
    print("=" * 60)
    
    # Check datasets
    print("📁 Datasets:")
    datasets = {
        "Fire Detection": "datasets/fire_detection/",
        "Gas Detection": "datasets/gas_detection/", 
        "PPE Compliance": "datasets/ppe_compliance/",
        "Structural Health": "datasets/structural_health/",
        "Object Detection": "datasets/object_detection/"
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: Ready")
        else:
            print(f"   ❌ {name}: Not found")
    
    # Check models
    print("\n🤖 AI Models:")
    models = {
        "Fire Detection": "models/fire_detection_model.pth",
        "Gas Detection": "models/gas_detection_model.joblib",
        "PPE Compliance": "models/ppe_compliance_model.pth", 
        "Structural Health": "models/structural_health_model.joblib"
    }
    
    for name, path in models.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: Trained and ready")
        else:
            print(f"   ❌ {name}: Not found")
    
    # Check components
    print("\n🔧 System Components:")
    components = {
        "Webots World": "enhanced_safety_patrol_bot.wbt",
        "Enhanced Controller": "controllers/enhanced_patrol_controller/",
        "Web Interface": "web_interface/enhanced_app.py",
        "Emergency Response": "emergency_response/emergency_controller.py"
    }
    
    for name, path in components.items():
        if os.path.exists(path):
            print(f"   ✅ {name}: Ready")
        else:
            print(f"   ❌ {name}: Not found")

def main():
    """Main launch function"""
    print("🚀 Enhanced Safety Patrol Bot System Launcher")
    print("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements not met!")
        print("Please ensure all components are properly set up.")
        return False
    
    print("\n✅ All system requirements met!")
    
    # Show system status
    show_system_status()
    
    # Start web interface
    web_started = start_web_interface()
    
    # Show Webots instructions
    show_webots_instructions()
    
    print("\n" + "=" * 60)
    print("🎉 Enhanced Safety Patrol Bot System Ready!")
    print("\n📋 Quick Start:")
    print("   1. Web Interface: http://localhost:5000 (should open automatically)")
    print("   2. Webots: Load enhanced_safety_patrol_bot.wbt")
    print("   3. Controller: Set to enhanced_patrol_controller")
    print("   4. Run simulation and monitor via web dashboard")
    
    print("\n🔧 Features Available:")
    print("   ✅ Fire & Heat Detection")
    print("   ✅ Gas & Air Quality Monitoring") 
    print("   ✅ PPE Compliance Monitoring")
    print("   ✅ Structural Health Monitoring")
    print("   ✅ Object Detection & Navigation")
    print("   ✅ Emergency Response System")
    print("   ✅ Real-time Web Dashboard")
    
    print("\n📱 Web Dashboard Features:")
    print("   - Real-time sensor data visualization")
    print("   - AI model predictions and alerts")
    print("   - Emergency response management")
    print("   - System performance monitoring")
    print("   - Historical data analysis")
    
    if web_started:
        print("\n🌐 Web interface is running at: http://localhost:5000")
        print("📊 Monitor your safety patrol bot in real-time!")
    
    print("\n🎯 Your enhanced safety patrol bot is ready for deployment!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ System launch completed successfully!")
        print("🤖 Start your Webots simulation to begin autonomous patrol!")
    else:
        print("\n❌ System launch failed!")
        print("Please check the requirements and try again.")
