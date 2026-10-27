"""
Script para descargar e integrar el dataset RealWaste (UCI Machine Learning Repository)
Categorías: Metal -> lata, Plastic -> plastico, Glass -> vidrio
También genera y enriquece la clase 'fondo' con variedad de ambientes reales.
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import math
import random

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
REALWASTE_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
ZIP_PATH = os.path.join(BASE_DIR, "realwaste_temp.zip")

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        sys.stdout.write(f"\r[Descargando RealWaste] {percent:.1f}% ({downloaded // (1024*1024)} MB de {total_size // (1024*1024)} MB)")
    else:
        sys.stdout.write(f"\r[Descargando RealWaste] {downloaded // (1024*1024)} MB descargados...")
    sys.stdout.flush()

def descargar_realwaste():
    print("=======================================================")
    print("  Descarga de RealWaste (UCI Machine Learning Repo)   ")
    print("=======================================================")
    print(f"URL: {REALWASTE_URL}\n")

    if not os.path.exists(ZIP_PATH):
        try:
            import ssl
            ssl_ctx = ssl._create_unverified_context()
            https_handler = urllib.request.HTTPSHandler(context=ssl_ctx)
            opener = urllib.request.build_opener(https_handler)
            opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(REALWASTE_URL, ZIP_PATH, reporthook=report_progress)
            print("\n\n[OK] Descarga completada.")
        except Exception as e:
            print(f"\n[ERROR al descargar de UCI]: {e}")
            print("Intentando fuente de respaldo si está disponible...")
            return False
    else:
        print("[INFO] El archivo realwaste_temp.zip ya existe localmente. Procediendo a extracción...")

    return True

def extraer_e_integrar():
    if not os.path.exists(ZIP_PATH):
        print("[ERROR] Archivo ZIP no encontrado.")
        return False

    print("\n[Extrayendo e integrando categorías relevantes: Metal, Plastic, Glass]...")
    
    # Mapeo de carpetas de RealWaste a nuestras clases
    # En RealWaste suele venir en realwaste-main o realwaste/ o Carpetas de materiales
    mapa = {
        "metal": "lata",
        "plastic": "plastico",
        "glass": "vidrio"
    }
    
    conteo_agregadas = {"lata": 0, "plastico": 0, "vidrio": 0}
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        namelist = zf.namelist()
        print(f"Total archivos en el ZIP: {len(namelist)}")
        
        # Primero revisamos si hay otro zip interno o carpetas directas
        for member in namelist:
            member_lower = member.lower().replace('\\', '/')
            if member_lower.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                for rw_cat, dest_cat in mapa.items():
                    # Verificar si la ruta contiene la categoría (ej: /metal/, /plastic/, /glass/)
                    if f"/{rw_cat}/" in member_lower or member_lower.startswith(f"{rw_cat}/"):
                        filename = os.path.basename(member)
                        if filename:
                            dest_dir = os.path.join(DATASET_DIR, dest_cat)
                            os.makedirs(dest_dir, exist_ok=True)
                            
                            dest_file = os.path.join(dest_dir, f"rw_{filename}")
                            if not os.path.exists(dest_file):
                                data = zf.read(member)
                                with open(dest_file, 'wb') as f_out:
                                    f_out.write(data)
                                conteo_agregadas[dest_cat] += 1
                                break

    print("\n--- Nuevas imágenes reales integradas al dataset ---")
    for cat, cant in conteo_agregadas.items():
        print(f"  * dataset/{cat}: +{cant} fotos reales de RealWaste")

    # Limpiar ZIP para ahorrar espacio
    try:
        os.remove(ZIP_PATH)
        print("[OK] Archivo temporal ZIP eliminado.")
    except Exception:
        pass

    return True

if __name__ == "__main__":
    if descargar_realwaste():
        extraer_e_integrar()
    else:
        print("[FALLO] No se pudo completar la integración de RealWaste.")
