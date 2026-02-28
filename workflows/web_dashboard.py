import http.server
import socketserver
import json
import sqlite3
import urllib.parse
import os
import webbrowser
import webbrowser

PORT = 8001
DB_FILE = os.path.join(os.path.dirname(__file__), 'trading_database.sqlite')

class TradingDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        # ── Serve the Front-end HTML ──
        if parsed_path.path == '/' or parsed_path.path == '/index.html':
            self.path = '/dashboard.html'
            
        return super().do_GET()

if __name__ == "__main__":
    # Change dir to workflows so it can serve dashboard.html from there
    os.chdir(os.path.dirname(__file__))
    # Force allows address reuse to prevent "Address already in use" errors during quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), TradingDashboardHandler) as httpd:
        print("="*60)
        print(f" 🚀 AI Trading Dashboard Online!")
        print(f" 🌐 Open your browser and go to: http://localhost:{PORT}")
        print("="*60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
