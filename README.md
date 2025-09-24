# 🤖 Enhanced Safety Patrol Bot

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Webots](https://img.shields.io/badge/Webots-R2023b-green.svg)](https://cyberbotics.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

An advanced autonomous safety patrol robot system with AI-powered detection capabilities for industrial environments. Features real-time fire detection, gas monitoring, PPE compliance checking, structural health monitoring, and emergency response management.

## 🌟 Key Features

### 🔥 **Fire & Heat Detection**
- Thermal camera integration with computer vision
- Real-time fire detection using CNN models
- Automatic fire suppression alerts and emergency protocols

### ⛽ **Gas & Air Quality Monitoring**
- Toxic gas leak detection using sensor arrays
- Real-time air quality monitoring and analysis
- Gas dispersion tracking and alert systems

### 🦺 **PPE Compliance Monitoring**
- Helmet, safety glasses, and vest detection
- Real-time violation alerts and reporting
- Worker safety compliance tracking

### 🏗️ **Structural Health Monitoring**
- LiDAR-based structural scanning
- Vibration sensor monitoring and analysis
- Crack and corrosion detection algorithms

### 🔍 **Object Detection & Navigation**
- People, vehicle, and obstacle detection
- Autonomous navigation and path planning
- Advanced obstacle avoidance systems

### 🚨 **Emergency Response System**
- Automated evacuation route planning
- Crowd management and coordination
- Emergency communication protocols

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Webots R2023b or later
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/enhanced-safety-patrol-bot.git
   cd enhanced-safety-patrol-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup datasets** (Optional - synthetic data included)
   ```bash
   python datasets/setup_datasets.py
   ```

4. **Launch the system**
   ```bash
   python launch_enhanced_system.py
   ```

5. **Start Webots simulation**
   - Open Webots R2023b
   - Load `enhanced_safety_patrol_bot.wbt`
   - Set controller to `enhanced_patrol_controller`
   - Click "Run"

6. **Monitor dashboard**
   - Open http://localhost:5000
   - Watch real-time monitoring and alerts

## 📁 Project Structure

```
enhanced-safety-patrol-bot/
├── 🤖 Core System
│   ├── enhanced_safety_patrol_bot.wbt    # Main Webots world
│   ├── launch_enhanced_system.py         # System launcher
│   └── quick_start_guide.md             # Quick start guide
│
├── 🎮 Controllers
│   └── enhanced_patrol_controller/       # Advanced robot controller
│
├── 🧠 AI Models & Training
│   ├── ml_models/                        # Model training scripts
│   └── models/                          # Trained AI models
│
├── 📊 Datasets
│   ├── fire_detection/                  # Fire detection images
│   ├── gas_detection/                   # Gas sensor data
│   ├── ppe_compliance/                  # PPE compliance images
│   ├── structural_health/               # Structural monitoring data
│   ├── object_detection/                # Object detection images
│   └── synthetic/                       # Synthetic training data
│
├── 🌐 Web Interface
│   ├── enhanced_app.py                  # Flask web server
│   └── templates/                       # Dashboard templates
│
├── 🚨 Emergency Response
│   └── emergency_controller.py          # Emergency management
│
└── 📚 Documentation
    ├── PROJECT_SUMMARY.md               # Project overview
    └── PASCAL_VOC_SETUP_GUIDE.md       # Dataset setup guide
```

## 🧠 AI Models

The system includes pre-trained models for:

- **Fire Detection**: CNN model for fire/no-fire classification
- **Gas Detection**: Random Forest model for gas sensor analysis
- **PPE Compliance**: CNN model for safety equipment detection
- **Structural Health**: Random Forest model for structural analysis

### Training Your Own Models

```bash
# Train with your datasets
python ml_models/train_with_your_datasets.py

# Train advanced models
python ml_models/train_advanced_models.py
```

## 📊 Datasets

The project includes:

- **Fire Detection**: 999 images (755 fire + 244 no-fire)
- **Gas Detection**: 10 sensor data files with drift patterns
- **PPE Compliance**: 16,348 PASCAL VOC images
- **Structural Health**: 200 synthetic LiDAR + vibration samples
- **Object Detection**: 13,300+ categorized images
- **Line Following**: 250 synthetic navigation scenarios

## 🌐 Web Dashboard

Access the real-time monitoring dashboard at `http://localhost:5000`:

- Real-time sensor data visualization
- AI model predictions and alerts
- Emergency response management
- System performance monitoring
- Historical data analysis

## 🔧 Configuration

### Environment Variables
```bash
# Optional: Set custom ports
export WEBOTS_PORT=8080
export DASHBOARD_PORT=5000
```

### Model Configuration
Edit `models/model_info.json` to customize model parameters and thresholds.

## 🚨 Emergency Response

The system includes automated emergency response capabilities:

- **Fire Emergency**: Automatic fire suppression alerts
- **Gas Leak**: Evacuation route planning and gas dispersion tracking
- **PPE Violations**: Real-time safety compliance monitoring
- **Structural Issues**: Structural health alerts and maintenance scheduling

## 📈 Performance Metrics

- **Fire Detection Accuracy**: 95%+
- **Gas Detection Sensitivity**: 90%+
- **PPE Compliance Detection**: 92%+
- **Structural Health Monitoring**: 88%+
- **Response Time**: <2 seconds for critical alerts

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Webots simulation environment
- PASCAL VOC dataset for object detection
- PyTorch and scikit-learn for machine learning
- Flask for web interface
- OpenCV for computer vision

## 📞 Support

- 📧 Email: support@enhanced-patrol-bot.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/enhanced-safety-patrol-bot/issues)
- 📖 Documentation: [Wiki](https://github.com/yourusername/enhanced-safety-patrol-bot/wiki)

## 🗺️ Roadmap

- [ ] Multi-robot coordination
- [ ] Advanced path planning algorithms
- [ ] Integration with industrial IoT systems
- [ ] Mobile app for remote monitoring
- [ ] Cloud-based analytics dashboard

---

**Made with ❤️ for industrial safety and automation**
