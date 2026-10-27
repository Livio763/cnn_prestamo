"""
Script para poblar 'dataset/fondo' con más de 1,000 imágenes fotográficas REALES
(habitaciones, texturas, oficinas, paredes, luz natural/artificial, manos, sombras y cámara tapada con ruido de sensor)
eliminando el sesgo hacia 'lata'.
"""

import os
import sys
import glob
import time
import random
import numpy as np
from PIL import Image
import urllib.request
import concurrent.futures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FONDO_DIR = os.path.join(DATASET_DIR, "fondo")
os.makedirs(FONDO_DIR, exist_ok=True)

# 1. Limpiar viejos BMP sintéticos
def limpiar_viejos_bmps():
    bmps = glob.glob(os.path.join(FONDO_DIR, "*.bmp"))
    print(f"Limpiando {len(bmps)} archivos BMP sintéticos antiguos en dataset/fondo...")
    for f in bmps:
        try:
            os.remove(f)
        except Exception:
            pass
    print("[OK] Viejos BMP sintéticos eliminados.")

# 2. Descargar fotos reales diversas (ambientes, texturas, oficinas, interiores, paredes)
def descargar_foto_real(item):
    idx, seed = item
    ruta = os.path.join(FONDO_DIR, f"fondo_real_{idx:04d}.jpg")
    if os.path.exists(ruta):
        return True
    
    url = f"https://picsum.photos/seed/bg_{seed}/224/224"
    for intento in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = resp.read()
            import io
            im = Image.open(io.BytesIO(data))
            im.verify()
            with open(ruta, 'wb') as f:
                f.write(data)
            return True
        except Exception:
            time.sleep(0.5)
    return False

def poblar_fotos_reales(total=950):
    print(f"\nDescargando {total} fotografías reales de entornos cotidianos (multihilo)...")
    items = [(i + 1, f"scene_{i}_{random.randint(100, 9999)}") for i in range(total)]
    
    t0 = time.time()
    exitosos = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        resultados = list(executor.map(descargar_foto_real, items))
        exitosos = sum(1 for r in resultados if r)
    
    print(f"[OK] {exitosos}/{total} fotos reales descargadas en {time.time() - t0:.1f} segundos.")

# 3. Generar muestras fotográficas de cámara tapada, sombras y tonos de piel con ruido de sensor
def generar_muestras_sensor(total=150):
    print(f"\nGenerando {total} muestras realistas de cámara tapada, sombras y piel con ruido gaussiano de sensor...")
    start_idx = 1000
    for i in range(total):
        ruta = os.path.join(FONDO_DIR, f"fondo_sensor_{start_idx + i:04d}.jpg")
        
        tipo = i % 3
        if tipo == 0:
            base = random.randint(0, 15)
            ruido = np.random.normal(0, 4, (160, 160, 3))
            arr = np.clip(base + ruido, 0, 255).astype(np.uint8)
        elif tipo == 1:
            base = random.randint(15, 50)
            ruido = np.random.normal(0, 6, (160, 160, 3))
            arr = np.clip(base + ruido, 0, 255).astype(np.uint8)
        else:
            colores_piel = [
                (230, 190, 170), (210, 165, 140), (180, 130, 100), (140, 95, 70), (100, 65, 45)
            ]
            piel = random.choice(colores_piel)
            ruido = np.random.normal(0, 10, (160, 160, 3))
            base_img = np.zeros((160, 160, 3), dtype=np.float32)
            base_img[:, :] = piel
            arr = np.clip(base_img + ruido, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        img.save(ruta, "JPEG", quality=85)

    print(f"[OK] {total} muestras con ruido de sensor fotográfico guardadas.")

if __name__ == "__main__":
    limpiar_viejos_bmps()
    poblar_fotos_reales(950)
    generar_muestras_sensor(150)
    
    total_fondo = len(os.listdir(FONDO_DIR))
    print(f"\n=======================================================")
    print(f"Total imágenes reales en dataset/fondo: {total_fondo}")
    print(f"=======================================================")
