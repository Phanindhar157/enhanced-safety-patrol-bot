#!/usr/bin/env python3

"""
Simple Working Web Interface for Enhanced Safety Patrol Bot
Basic monitoring dashboard that will definitely work
"""

from flask import Flask, jsonify, request
import json
import os
import time
import random
from datetime import datetime

app = Flask(__name__)

# Global variables for sensor data
sensor_data = {
    'timestamp': datetime.now().isoformat(),
    'position': {'x': 0, 'y': 0, 'z': 0},
    'orientation': {'x': 0, 'y': 0, 'z': 0},
    'fire_detected': False,
    'gas_detected': False,
    'ppe_compliant': True,
    'structural_health': 'normal',
    'emergency_mode': False,
    'patrol_mode': 'automatic',
    'battery_level': 100,
    'system_status': 'operational',
    'temperature': 25.0,
    'humidity': 60.0,
    'air_quality': 'good',
    'vibration': 'low',
    'safety_score': 95
}

# Safety alerts
safety_alerts = []

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Enhanced Safety Patrol Bot Dashboard</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .header {
                background: rgba(255, 255, 255, 0.95);
                color: #2c3e50;
                padding: 30px;
                text-align: center;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
                background: linear-gradient(45deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 25px;
                margin-bottom: 30px;
            }
            .status-card {
                background: rgba(255, 255, 255, 0.95);
                padding: 25px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s ease;
            }
            .status-card:hover {
                transform: translateY(-5px);
            }
            .status-card h3 {
                margin-top: 0;
                color: #2c3e50;
                font-size: 1.4em;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .status-indicator {
                display: inline-block;
                width: 15px;
                height: 15px;
                border-radius: 50%;
                margin-right: 10px;
                animation: pulse 2s infinite;
            }
            .status-normal { background-color: #27ae60; }
            .status-warning { background-color: #f39c12; }
            .status-danger { background-color: #e74c3c; }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .alerts-section {
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .alert-item {
                padding: 15px;
                margin: 15px 0;
                border-radius: 10px;
                border-left: 5px solid;
                animation: slideIn 0.5s ease;
            }
            .alert-info { background-color: #d1ecf1; border-color: #17a2b8; }
            .alert-warning { background-color: #fff3cd; border-color: #ffc107; }
            .alert-danger { background-color: #f8d7da; border-color: #dc3545; }
            @keyframes slideIn {
                from { opacity: 0; transform: translateX(-20px); }
                to { opacity: 1; transform: translateX(0); }
            }
            .refresh-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 25px;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            .refresh-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .metric {
                font-size: 1.2em;
                font-weight: bold;
                color: #2c3e50;
            }
            .timestamp {
                color: #7f8c8d;
                font-size: 0.9em;
                text-align: center;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Enhanced Safety Patrol Bot</h1>
            <p>Real-time Monitoring and Control System</p>
            <p class="timestamp">Last Updated: <span id="last-update">--</span></p>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>🔥 Fire Detection</h3>
                <p><span id="fire-status" class="status-indicator status-normal"></span>
                   <span id="fire-text">No Fire Detected</span></p>
                <p>Temperature: <span id="temperature" class="metric">25.0°C</span></p>
            </div>

            <div class="status-card">
                <h3>⛽ Gas Detection</h3>
                <p><span id="gas-status" class="status-indicator status-normal"></span>
                   <span id="gas-text">No Gas Leaks</span></p>
                <p>Air Quality: <span id="air-quality" class="metric">Good</span></p>
            </div>

            <div class="status-card">
                <h3>🦺 PPE Compliance</h3>
                <p><span id="ppe-status" class="status-indicator status-normal"></span>
                   <span id="ppe-text">Compliant</span></p>
                <p>Safety Score: <span id="safety-score" class="metric">95%</span></p>
            </div>

            <div class="status-card">
                <h3>🏗️ Structural Health</h3>
                <p><span id="structural-status" class="status-indicator status-normal"></span>
                   <span id="structural-text">Normal</span></p>
                <p>Vibration: <span id="vibration" class="metric">Low</span></p>
            </div>

            <div class="status-card">
                <h3>🔋 System Status</h3>
                <p><span id="system-status" class="status-indicator status-normal"></span>
                   <span id="system-text">Operational</span></p>
                <p>Battery: <span id="battery" class="metric">100%</span></p>
            </div>

            <div class="status-card">
                <h3>🚨 Emergency Status</h3>
                <p><span id="emergency-status" class="status-indicator status-normal"></span>
                   <span id="emergency-text">Normal</span></p>
                <p>Patrol Mode: <span id="patrol-mode" class="metric">Automatic</span></p>
            </div>
        </div>

        <div class="alerts-section">
            <h3>📢 Safety Alerts</h3>
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
            <div id="alerts-container">
                <p>No active alerts</p>
            </div>
        </div>

        <script>
            function updateStatus() {
                fetch('/api/sensor_data')
                    .then(response => response.json())
                    .then(data => {
                        // Update timestamp
                        document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                        
                        // Update fire detection
                        const fireStatus = document.getElementById('fire-status');
                        const fireText = document.getElementById('fire-text');
                        if (data.fire_detected) {
                            fireStatus.className = 'status-indicator status-danger';
                            fireText.textContent = 'FIRE DETECTED!';
                        } else {
                            fireStatus.className = 'status-indicator status-normal';
                            fireText.textContent = 'No Fire Detected';
                        }
                        document.getElementById('temperature').textContent = data.temperature + '°C';

                        // Update gas detection
                        const gasStatus = document.getElementById('gas-status');
                        const gasText = document.getElementById('gas-text');
                        if (data.gas_detected) {
                            gasStatus.className = 'status-indicator status-danger';
                            gasText.textContent = 'GAS LEAK DETECTED!';
                        } else {
                            gasStatus.className = 'status-indicator status-normal';
                            gasText.textContent = 'No Gas Leaks';
                        }
                        document.getElementById('air-quality').textContent = data.air_quality;

                        // Update PPE compliance
                        const ppeStatus = document.getElementById('ppe-status');
                        const ppeText = document.getElementById('ppe-text');
                        if (data.ppe_compliant) {
                            ppeStatus.className = 'status-indicator status-normal';
                            ppeText.textContent = 'Compliant';
                        } else {
                            ppeStatus.className = 'status-indicator status-warning';
                            ppeText.textContent = 'Non-Compliant';
                        }
                        document.getElementById('safety-score').textContent = data.safety_score + '%';

                        // Update structural health
                        const structuralStatus = document.getElementById('structural-status');
                        const structuralText = document.getElementById('structural-text');
                        if (data.structural_health === 'normal') {
                            structuralStatus.className = 'status-indicator status-normal';
                            structuralText.textContent = 'Normal';
                        } else {
                            structuralStatus.className = 'status-indicator status-warning';
                            structuralText.textContent = 'Issues Detected';
                        }
                        document.getElementById('vibration').textContent = data.vibration;

                        // Update system status
                        const systemStatus = document.getElementById('system-status');
                        const systemText = document.getElementById('system-text');
                        if (data.system_status === 'operational') {
                            systemStatus.className = 'status-indicator status-normal';
                            systemText.textContent = 'Operational';
                        } else {
                            systemStatus.className = 'status-indicator status-warning';
                            systemText.textContent = 'Issues';
                        }
                        document.getElementById('battery').textContent = data.battery_level + '%';

                        // Update emergency status
                        const emergencyStatus = document.getElementById('emergency-status');
                        const emergencyText = document.getElementById('emergency-text');
                        if (data.emergency_mode) {
                            emergencyStatus.className = 'status-indicator status-danger';
                            emergencyText.textContent = 'EMERGENCY MODE';
                        } else {
                            emergencyStatus.className = 'status-indicator status-normal';
                            emergencyText.textContent = 'Normal';
                        }
                        document.getElementById('patrol-mode').textContent = data.patrol_mode;
                    })
                    .catch(error => {
                        console.error('Error fetching data:', error);
                    });
            }

            function refreshData() {
                updateStatus();
                fetch('/api/alerts')
                    .then(response => response.json())
                    .then(alerts => {
                        const container = document.getElementById('alerts-container');
                        if (alerts.length === 0) {
                            container.innerHTML = '<p>No active alerts</p>';
                        } else {
                            container.innerHTML = alerts.map(alert => 
                                `<div class="alert-item alert-${alert.level}">
                                    <strong>${alert.timestamp}</strong>: ${alert.message}
                                </div>`
                            ).join('');
                        }
                    });
            }

            // Update data every 3 seconds
            setInterval(updateStatus, 3000);
            
            // Initial load
            updateStatus();
        </script>
    </body>
    </html>
    """

@app.route('/api/sensor_data')
def get_sensor_data():
    """Get current sensor data"""
    # Simulate some data changes
    sensor_data['timestamp'] = datetime.now().isoformat()
    sensor_data['temperature'] = round(20 + random.random() * 15, 1)
    sensor_data['battery_level'] = max(0, sensor_data['battery_level'] - 0.1)
    sensor_data['safety_score'] = max(80, sensor_data['safety_score'] + random.uniform(-2, 2))
    
    # Simulate occasional alerts
    if random.random() < 0.05:  # 5% chance
        sensor_data['fire_detected'] = True
        add_alert("🔥 Fire detected in sector A!", "danger")
    else:
        sensor_data['fire_detected'] = False
    
    if random.random() < 0.03:  # 3% chance
        sensor_data['gas_detected'] = True
        add_alert("⛽ Gas leak detected!", "danger")
    else:
        sensor_data['gas_detected'] = False
    
    if random.random() < 0.02:  # 2% chance
        sensor_data['ppe_compliant'] = False
        add_alert("🦺 PPE compliance violation detected!", "warning")
    else:
        sensor_data['ppe_compliant'] = True
    
    if random.random() < 0.01:  # 1% chance
        sensor_data['emergency_mode'] = True
        add_alert("🚨 EMERGENCY MODE ACTIVATED!", "danger")
    else:
        sensor_data['emergency_mode'] = False
    
    return jsonify(sensor_data)

@app.route('/api/alerts')
def get_alerts():
    """Get current safety alerts"""
    return jsonify(safety_alerts)

def add_alert(message, level):
    """Add a new safety alert"""
    alert = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'message': message,
        'level': level
    }
    safety_alerts.append(alert)
    
    # Keep only last 10 alerts
    if len(safety_alerts) > 10:
        safety_alerts.pop(0)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Safety Patrol Bot Web Interface")
    print("🌐 Dashboard available at: http://localhost:5000")
    print("📊 Real-time monitoring and control system")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)

