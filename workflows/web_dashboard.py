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
        
        # ── API Endpoint: Fetch Market Data from SQLite ──
        if parsed_path.path == '/api/data':
            query_components = urllib.parse.parse_qs(parsed_path.query)
            market = query_components.get('market', ['BTC'])[0]
            
            try:
                conn = sqlite3.connect(DB_FILE)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Fetch Overview Stats
                cursor.execute("SELECT * FROM strategy_performance WHERE market = ?", (market,))
                stats_row = cursor.fetchone()
                
                # Fetch Daily Signals
                table_name = f"signals_history_{market}"
                cursor.execute(f'SELECT * FROM "{table_name}" ORDER BY date ASC')
                history_rows = cursor.fetchall()
                conn.close()
                
                response_data = {
                    "market": market,
                    "stats": dict(stats_row) if stats_row else None,
                    "history": [dict(r) for r in history_rows]
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
            return
            
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
