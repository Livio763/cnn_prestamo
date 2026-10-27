"""
Script de utilidad para descargar o preparar el dataset de reciclaje
Descarga automática de TrashNet (Stanford / Gary Thung) usando únicamente
módulos estándar de Python (sin necesidad de librerías externas).
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import struct

# Compatibilidad UTF-8 para consola de Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRASHNET_URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, (downloaded / total_size) * 100)
        sys.stdout.write(f"\r[Descargando] {percent:.1f}% ({downloaded // (1024*1024)} MB de {total_size // (1024*1024)} MB)")
        sys.stdout.flush()

def generar_fondo_sintetico(cantidad=25):
    """Crea imágenes neutras de fondo en formato BMP (sin dependencias)"""
    carpeta_fondo = os.path.join(DATASET_DIR, "fondo")
    os.makedirs(carpeta_fondo, exist_ok=True)
    
    colores_fondo = [
        (240, 240, 240), # Blanco suave
        (220, 225, 230), # Gris claro
        (210, 200, 190), # Madera clara
        (180, 180, 180), # Mostrador
        (230, 230, 220), # Crema
    ]

    for i in range(cantidad):
        color = colores_fondo[i % len(colores_fondo)]
        ancho, alto = 224, 224
        row_bytes = ancho * 3
        padding = (4 - (row_bytes % 4)) % 4
        file_size = 54 + (row_bytes + padding) * alto

        header = bytearray(b'BM')
        header += struct.pack('<IHHI', file_size, 0, 0, 54)
        header += struct.pack('<IIIHHIIIIII', 40, ancho, alto, 1, 24, 0, (row_bytes + padding) * alto, 2835, 2835, 0, 0)
        
        b, g, r = color[2], color[1], color[0]
        pixel_row = bytearray([b, g, r] * ancho) + bytearray(padding)

        ruta = os.path.join(carpeta_fondo, f"fondo_{i+1}.bmp")
        with open(ruta, 'wb') as f:
            f.write(header)
            for _ in range(alto):
                f.write(pixel_row)
    print(f"[OK] Se generaron {cantidad} imagenes neutras en 'dataset/fondo/'.")

def descargar_trashnet():
    print("\n=======================================================")
    print("  Descarga oficial de TrashNet (Gary Thung - Stanford)  ")
    print("=======================================================")
    print("Tamano aproximado: ~40 MB")
    print(f"URL de descarga: {TRASHNET_URL}\n")

    zip_temp = os.path.join(BASE_DIR, "trashnet_temp.zip")

    try:
        # Descarga con progreso
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(TRASHNET_URL, zip_temp, reporthook=report_progress)
        print("\n\n[OK] Descarga completa. Extrayendo archivos del dataset...")

        with zipfile.ZipFile(zip_temp, 'r') as zip_ref:
            lista_archivos = zip_ref.namelist()
            
            # Mapeo de carpetas de TrashNet a nuestras clases
            mapa = {
                "metal": "lata",
                "plastic": "plastico",
                "glass": "vidrio"
            }

            conteo = {"lata": 0, "plastico": 0, "vidrio": 0}

            for miembro in lista_archivos:
                for carpeta_origen, clase_dest in mapa.items():
                    if f"/{carpeta_origen}/" in miembro and miembro.lower().endswith(('.jpg', '.jpeg', '.png')):
                        nombre_archivo = os.path.basename(miembro)
                        if nombre_archivo:
                            dir_dest = os.path.join(DATASET_DIR, clase_dest)
                            os.makedirs(dir_dest, exist_ok=True)
                            
                            contenido = zip_ref.read(miembro)
                            ruta_salida = os.path.join(dir_dest, nombre_archivo)
                            with open(ruta_salida, 'wb') as f_out:
                                f_out.write(contenido)
                            conteo[clase_dest] += 1

        # Limpiar archivo zip temporal
        if os.path.exists(zip_temp):
            os.remove(zip_temp)

        print("[OK] Imagenes extraidas correctamente:")
        for c, cant in conteo.items():
            print(f"  * dataset/{c}: {cant} fotos")

        # Generar fondos neutros para la clase fondo
        generar_fondo_sintetico(40)

        print("\n[LISTO] Dataset TrashNet instalado con exito.")
        print("Ya puedes ejecutar: python entrenar.py")

    except Exception as e:
        print(f"\n[ERROR] Durante la descarga: {e}")
        if os.path.exists(zip_temp):
            os.remove(zip_temp)

def generar_dataset_sintetico():
    print("\n[INFO] Generando 25 imagenes sinteticas por clase (Dry-run inmediato sin descargas)...")
    categorias = {
        "lata": (192, 192, 192),      # Plateado / Metal
        "plastico": (50, 150, 250),   # Azul traslúcido
        "vidrio": (34, 139, 34),      # Verde botella
        "fondo": (240, 240, 230)      # Fondo neutro
    }

    ancho, alto = 224, 224
    row_bytes = ancho * 3
    padding = (4 - (row_bytes % 4)) % 4
    file_size = 54 + (row_bytes + padding) * alto

    for cat, color in categorias.items():
        dir_cat = os.path.join(DATASET_DIR, cat)
        os.makedirs(dir_cat, exist_ok=True)
        b, g, r = color[2], color[1], color[0]
        pixel_row = bytearray([b, g, r] * ancho) + bytearray(padding)

        for i in range(25):
            header = bytearray(b'BM')
            header += struct.pack('<IHHI', file_size, 0, 0, 54)
            header += struct.pack('<IIIHHIIIIII', 40, ancho, alto, 1, 24, 0, (row_bytes + padding) * alto, 2835, 2835, 0, 0)
            
            ruta = os.path.join(dir_cat, f"muestra_{i+1}.bmp")
            with open(ruta, 'wb') as f:
                f.write(header)
                for _ in range(alto):
                    f.write(pixel_row)

    print("[OK] 100 imagenes sinteticas creadas (25 por categoria) en 'dataset/'.")
    print("Ya puedes ejecutar: python entrenar.py")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if "--trashnet" in sys.argv:
            descargar_trashnet()
            sys.exit(0)
        elif "--dry-run" in sys.argv:
            generar_dataset_sintetico()
            sys.exit(0)

    print("=========================================")
    print("  Gestor de Dataset - Punto Verde AI     ")
    print("=========================================")
    print("1. Descargar TrashNet (El mejor dataset: ~40 MB, fotos reales de latas, botellas y vidrio)")
    print("2. Generar imagenes de prueba rapida (Dry Run inmediato sin internet)")
    print("3. Salir")
    opc = input("\nElige una opcion (1/2/3): ").strip()
    if opc == "1":
        descargar_trashnet()
    elif opc == "2":
        generar_dataset_sintetico()
    else:
        print("Operacion cancelada.")
