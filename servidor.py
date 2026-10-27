"""
Servidor local para la app de préstamo.
Ejecuta FastAPI + Uvicorn con soporte de API REST y archivos estáticos.

Uso:
    cd prestamo
    python servidor.py
    # abrir http://localhost:8001
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
PORT = int(os.environ.get("PORT", "8001"))

try:
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from api.index import app

    # Montar archivos estáticos para el frontend web al final
    app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static")

    if __name__ == "__main__":
        print(f"Servidor Préstamo Inteligente (FastAPI + Web) activo en http://localhost:{PORT}")
        sys.stdout.flush()
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

except Exception as error:
    # Respaldo con servidor HTTP estándar en caso de fallo
    print(f"[AVISO] Iniciando servidor estático básico ({error}):")
    import http.server
    import socketserver

    class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

    if __name__ == "__main__":
        with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
            print(f"Servidor estático activo en http://localhost:{PORT}")
            sys.stdout.flush()
            httpd.serve_forever()
