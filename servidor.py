import http.server
import socketserver
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = 8000

class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, format, *args):
        try:
            super().log_message(format, *args)
            sys.stdout.flush()
        except Exception:
            pass

class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    while True:
        try:
            with ThreadingTCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
                print(f"Servidor activo en http://localhost:{PORT}")
                sys.stdout.flush()
                httpd.serve_forever()
        except Exception as e:
            print(f"Reanudando servidor tras excepción: {e}")
            sys.stdout.flush()
            time.sleep(1)
