#!/usr/bin/env python3

"""
Simple HTTP Server for Enhanced Safety Patrol Bot Dashboard
Works without Flask - uses Python's built-in HTTP server
"""

import http.server
import socketserver
import webbrowser
import os
import threading
import time

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def start_server():
    """Start the HTTP server"""
    PORT = 8000
    
    # Change to the directory containing dashboard.html
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"🚀 Enhanced Safety Patrol Bot Dashboard Server")
        print(f"🌐 Server running at: http://localhost:{PORT}")
        print(f"📊 Dashboard available at: http://localhost:{PORT}/dashboard.html")
        print("=" * 60)
        print("🎯 Your enhanced safety patrol bot dashboard is ready!")
        print("📱 Open your browser and go to: http://localhost:8000/dashboard.html")
        print("=" * 60)
        
        # Open browser automatically
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}/dashboard.html')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped by user")
            httpd.shutdown()

if __name__ == "__main__":
    start_server()

