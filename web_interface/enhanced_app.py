#!/usr/bin/env python3

"""
Enhanced Web Interface for Multi-Sensor Safety Patrol Bot
Comprehensive monitoring and control dashboard with advanced features
"""

from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
import json
import os
import threading
import time
import queue
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

app = Flask(__name__)
CORS(app)

# Global variables for communication with Webots controller
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
    'distance_sensors': {},
    'line_sensors': {},
    'gas_sensors': {},
    'air_quality': {},
    'vibration': {},
    'thermal_level': 0,
    'battery_level': 100,
    'system_status': 'operational'
}

# Safety alerts and emergency data
safety_alerts = []
emergency_status = {
    'active_emergencies': 0,
    'evacuation_status': 'normal',
    'emergency_type': None,
    'severity': 0,
    'location': None,
    'response_plan': None
}

# Historical data for analytics
historical_data = {
    'sensor_readings': [],
    'alerts': [],
    'patrol_routes': [],
    'performance_metrics': []
}

# Command queue for sending commands to robot
command_queue = queue.Queue()

# Performance metrics
performance_metrics = {
    'total_patrol_time': 0,
    'emergencies_handled': 0,
    'false_alarms': 0,
    'response_time_avg': 0,
    'system_uptime': 0,
    'battery_usage': 0
}

@app.route('/')
def index():
    """Main enhanced dashboard page"""
    return render_template('enhanced_dashboard.html')

@app.route('/api/sensor_data')
def get_sensor_data():
    """Get current comprehensive sensor data"""
    return jsonify(sensor_data)

@app.route('/api/safety_alerts')
def get_safety_alerts():
    """Get safety alerts and emergency status"""
    return jsonify({
        'alerts': safety_alerts[-50:],  # Last 50 alerts
        'emergency_status': emergency_status,
        'active_count': len(safety_alerts)
    })

@app.route('/api/historical_data')
def get_historical_data():
    """Get historical data for analytics"""
    hours = request.args.get('hours', 24, type=int)
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    filtered_data = {
        'sensor_readings': [d for d in historical_data['sensor_readings'] 
                           if datetime.fromisoformat(d['timestamp']) > cutoff_time],
        'alerts': [d for d in historical_data['alerts'] 
                  if datetime.fromisoformat(d['timestamp']) > cutoff_time],
        'performance_metrics': historical_data['performance_metrics']
    }
    
    return jsonify(filtered_data)

@app.route('/api/performance_metrics')
def get_performance_metrics():
    """Get performance metrics and analytics"""
    return jsonify(performance_metrics)

@app.route('/api/emergency_status')
def get_emergency_status():
    """Get detailed emergency status"""
    return jsonify(emergency_status)

@app.route('/api/patrol_control', methods=['POST'])
def patrol_control():
    """Control patrol operations"""
    try:
        data = request.json
        command = {
            'type': 'patrol_control',
            'action': data.get('action'),  # start, stop, pause, resume
            'mode': data.get('mode', 'automatic'),  # automatic, manual, emergency
            'route': data.get('route'),
            'timestamp': datetime.now().isoformat()
        }
        command_queue.put(command)
        return jsonify({'status': 'success', 'message': f'Patrol {command["action"]} command sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/emergency_response', methods=['POST'])
def emergency_response():
    """Handle emergency response commands"""
    try:
        data = request.json
        command = {
            'type': 'emergency_response',
            'action': data.get('action'),  # initiate_evacuation, clear_emergency, etc.
            'emergency_id': data.get('emergency_id'),
            'severity': data.get('severity'),
            'location': data.get('location'),
            'timestamp': datetime.now().isoformat()
        }
        command_queue.put(command)
        return jsonify({'status': 'success', 'message': 'Emergency response command sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/manual_control', methods=['POST'])
def manual_control():
    """Manual robot control with advanced options"""
    try:
        data = request.json
        command = {
            'type': 'manual_control',
            'left_speed': data.get('left_speed', 0),
            'right_speed': data.get('right_speed', 0),
            'camera_angle': data.get('camera_angle', 0),
            'sensor_sensitivity': data.get('sensor_sensitivity', 1.0),
            'timestamp': datetime.now().isoformat()
        }
        command_queue.put(command)
        return jsonify({'status': 'success', 'message': 'Manual control command sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/system_config', methods=['POST'])
def system_config():
    """Configure system parameters"""
    try:
        data = request.json
        command = {
            'type': 'system_config',
            'config': data.get('config', {}),
            'timestamp': datetime.now().isoformat()
        }
        command_queue.put(command)
        return jsonify({'status': 'success', 'message': 'System configuration updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/analytics/chart/<chart_type>')
def get_analytics_chart(chart_type):
    """Generate analytics charts"""
    try:
        hours = request.args.get('hours', 24, type=int)
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if chart_type == 'sensor_trends':
            return generate_sensor_trends_chart(cutoff_time)
        elif chart_type == 'alert_frequency':
            return generate_alert_frequency_chart(cutoff_time)
        elif chart_type == 'performance_metrics':
            return generate_performance_chart()
        elif chart_type == 'patrol_coverage':
            return generate_patrol_coverage_chart(cutoff_time)
        else:
            return jsonify({'error': 'Invalid chart type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_sensor_trends_chart(cutoff_time):
    """Generate sensor trends chart"""
    try:
        # Filter data
        filtered_data = [d for d in historical_data['sensor_readings'] 
                        if datetime.fromisoformat(d['timestamp']) > cutoff_time]
        
        if not filtered_data:
            return jsonify({'error': 'No data available'})
        
        # Extract data for plotting
        timestamps = [datetime.fromisoformat(d['timestamp']) for d in filtered_data]
        fire_detections = [1 if d.get('fire_detected', False) else 0 for d in filtered_data]
        gas_detections = [1 if d.get('gas_detected', False) else 0 for d in filtered_data]
        thermal_levels = [d.get('thermal_level', 0) for d in filtered_data]
        
        # Create plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        # Fire detections
        ax1.plot(timestamps, fire_detections, 'r-', label='Fire Detected')
        ax1.set_ylabel('Fire Detection')
        ax1.set_title('Fire Detection Over Time')
        ax1.grid(True)
        
        # Gas detections
        ax2.plot(timestamps, gas_detections, 'g-', label='Gas Detected')
        ax2.set_ylabel('Gas Detection')
        ax2.set_title('Gas Detection Over Time')
        ax2.grid(True)
        
        # Thermal levels
        ax3.plot(timestamps, thermal_levels, 'orange', label='Thermal Level')
        ax3.set_ylabel('Thermal Level')
        ax3.set_xlabel('Time')
        ax3.set_title('Thermal Levels Over Time')
        ax3.grid(True)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'chart': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_alert_frequency_chart(cutoff_time):
    """Generate alert frequency chart"""
    try:
        # Filter alerts
        filtered_alerts = [d for d in historical_data['alerts'] 
                          if datetime.fromisoformat(d['timestamp']) > cutoff_time]
        
        if not filtered_alerts:
            return jsonify({'error': 'No alert data available'})
        
        # Count alerts by type
        alert_types = {}
        for alert in filtered_alerts:
            alert_type = alert.get('category', 'unknown')
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = list(alert_types.keys())
        sizes = list(alert_types.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title('Alert Frequency by Type')
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'chart': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_performance_chart():
    """Generate performance metrics chart"""
    try:
        metrics = performance_metrics
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metric_names = ['Patrol Time (hrs)', 'Emergencies Handled', 'False Alarms', 
                       'Response Time (s)', 'System Uptime (hrs)', 'Battery Usage (%)']
        metric_values = [
            metrics['total_patrol_time'],
            metrics['emergencies_handled'],
            metrics['false_alarms'],
            metrics['response_time_avg'],
            metrics['system_uptime'],
            metrics['battery_usage']
        ]
        
        bars = ax.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange', 'purple', 'brown'])
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'chart': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_patrol_coverage_chart(cutoff_time):
    """Generate patrol coverage map"""
    try:
        # Filter patrol data
        filtered_routes = [d for d in historical_data['patrol_routes'] 
                          if datetime.fromisoformat(d['timestamp']) > cutoff_time]
        
        if not filtered_routes:
            return jsonify({'error': 'No patrol data available'})
        
        # Extract coordinates
        x_coords = [d.get('position', {}).get('x', 0) for d in filtered_routes]
        y_coords = [d.get('position', {}).get('z', 0) for d in filtered_routes]  # Using z as y for 2D map
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap of patrol coverage
        ax.scatter(x_coords, y_coords, c='blue', alpha=0.6, s=50)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Z Position')
        ax.set_title('Patrol Coverage Map')
        ax.grid(True, alpha=0.3)
        
        # Add coverage density
        if len(x_coords) > 10:
            from scipy.stats import gaussian_kde
            try:
                xy = np.vstack([x_coords, y_coords])
                density = gaussian_kde(xy)
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                x_grid = np.linspace(x_min, x_max, 50)
                y_grid = np.linspace(y_min, y_max, 50)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(density(positions).T, X.shape)
                ax.contour(X, Y, Z, levels=10, alpha=0.5, colors='red')
            except ImportError:
                pass  # scipy not available
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({'chart': f'data:image/png;base64,{img_base64}'})
        
    except Exception as e:
        return jsonify({'error': str(e)})

def update_sensor_data():
    """Update sensor data from log files"""
    global sensor_data, safety_alerts, emergency_status, historical_data, performance_metrics
    
    while True:
        try:
            # Update from enhanced sensor log
            if os.path.exists('enhanced_sensor_log.json'):
                with open('enhanced_sensor_log.json', 'r') as f:
                    new_sensor_data = json.load(f)
                
                if new_sensor_data:
                    latest = new_sensor_data[-1]
                    sensor_data.update(latest)
                    
                    # Add to historical data
                    historical_data['sensor_readings'].append(latest)
                    
                    # Keep only last 1000 entries
                    if len(historical_data['sensor_readings']) > 1000:
                        historical_data['sensor_readings'] = historical_data['sensor_readings'][-1000:]
            
            # Update safety alerts
            if os.path.exists('safety_alerts.json'):
                with open('safety_alerts.json', 'r') as f:
                    new_alerts = json.load(f)
                
                if new_alerts:
                    safety_alerts = new_alerts
                    historical_data['alerts'].extend(new_alerts[-10:])  # Add recent alerts
            
            # Update emergency status
            if os.path.exists('emergency_status.json'):
                with open('emergency_status.json', 'r') as f:
                    emergency_status.update(json.load(f))
            
            # Update performance metrics
            if os.path.exists('performance_metrics.json'):
                with open('performance_metrics.json', 'r') as f:
                    performance_metrics.update(json.load(f))
            
            time.sleep(0.5)  # Update every 500ms
            
        except Exception as e:
            print(f"Error updating data: {e}")
            time.sleep(1)

def get_command():
    """Get command from queue (for Webots controller)"""
    try:
        return command_queue.get_nowait()
    except queue.Empty:
        return None

def main():
    """Main function"""
    # Start data update thread
    data_thread = threading.Thread(target=update_sensor_data, daemon=True)
    data_thread.start()
    
    # Run Flask app
    print("🚀 Starting Enhanced Web Interface...")
    print("📊 Access the enhanced dashboard at: http://localhost:5000")
    print("🔧 Features:")
    print("   - Real-time sensor monitoring")
    print("   - Advanced analytics and charts")
    print("   - Emergency response management")
    print("   - Performance metrics")
    print("   - Patrol route visualization")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()

