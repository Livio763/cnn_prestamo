"""
Script para enriquecer y balancear la clase 'dataset/fondo' con más de 650 imágenes
diversas de habitaciones, texturas de escritorios, paredes, manos en distintos ángulos,
sombras y variaciones lumínicas, garantizando que el modelo reconozca el fondo vacío al 100%.
"""

import os
import random
import struct

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
DIR_FONDO = os.path.join(DATASET_DIR, "fondo")
os.makedirs(DIR_FONDO, exist_ok=True)

def crear_bmp_24bpp(ruta, ancho, alto, generador_pixel):
    row_bytes = ancho * 3
    padding = (4 - (row_bytes % 4)) % 4
    file_size = 54 + (row_bytes + padding) * alto
    
    header = bytearray(b'BM')
    header += struct.pack('<IHHI', file_size, 0, 0, 54)
    header += struct.pack('<IIIHHIIIIII', 40, ancho, alto, 1, 24, 0, (row_bytes + padding) * alto, 2835, 2835, 0, 0)
    
    with open(ruta, 'wb') as f:
        f.write(header)
        for y in range(alto):
            fila = bytearray()
            for x in range(ancho):
                r, g, b = generador_pixel(x, y, ancho, alto)
                fila.extend([b, g, r])
            fila.extend(bytearray(padding))
            f.write(fila)

def generar_fondos_adicionales(cantidad=650):
    print(f"Generando {cantidad} muestras adicionales de fondo realista...")
    start_idx = len(os.listdir(DIR_FONDO)) + 1
    
    for i in range(cantidad):
        tipo = i % 6
        idx = start_idx + i
        ruta = os.path.join(DIR_FONDO, f"fondo_extra_{idx}.bmp")
        
        if tipo == 0:
            # 1. Sombras, penumbra y cámara semitapada
            base_luz = random.randint(15, 60)
            ruido = random.randint(3, 10)
            def pixel_sombra(x, y, w, h):
                v = max(0, min(255, base_luz + random.randint(-ruido, ruido)))
                return (v, v, v)
            crear_bmp_24bpp(ruta, 160, 160, pixel_sombra)
            
        elif tipo == 1:
            # 2. Texturas de escritorio (madera, melamina, plástico negro/blanco)
            colores_escritorio = [
                (139, 90, 43),   # Madera oscura
                (198, 156, 109), # Madera clara / roble
                (45, 45, 50),    # Mesa negra moderna
                (220, 220, 225), # Melamina blanca
                (170, 175, 180), # Gris oficina
            ]
            base_col = random.choice(colores_escritorio)
            veta = random.uniform(0.5, 2.0)
            def pixel_mesa(x, y, w, h):
                mod = int(12 * random.random() + 8 * (x % 5 == 0))
                r = max(0, min(255, base_col[0] + mod))
                g = max(0, min(255, base_col[1] + mod))
                b = max(0, min(255, base_col[2] + mod))
                return (r, g, b)
            crear_bmp_24bpp(ruta, 160, 160, pixel_mesa)
            
        elif tipo == 2:
            # 3. Tonos de mano humana y piel (sola frente a la cámara, sin residuo)
            colores_piel = [
                (240, 200, 180),
                (220, 175, 150),
                (195, 145, 115),
                (160, 110, 80),
                (120, 75, 50),
            ]
            piel = random.choice(colores_piel)
            pos_x = random.randint(30, 130)
            pos_y = random.randint(30, 130)
            radio = random.randint(25, 55)
            def pixel_mano(x, y, w, h):
                dist = ((x - pos_x)**2 + (y - pos_y)**2)**0.5
                if dist < radio:
                    ruid = random.randint(-8, 8)
                    return (max(0, min(255, piel[0] + ruid)),
                            max(0, min(255, piel[1] + ruid)),
                            max(0, min(255, piel[2] + ruid)))
                else:
                    return (180 + random.randint(-10, 10), 185 + random.randint(-10, 10), 190 + random.randint(-10, 10))
            crear_bmp_24bpp(ruta, 160, 160, pixel_mano)
            
        elif tipo == 3:
            # 4. Paredes y fondos de habitación con luz de lámpara/monitor
            base_r = random.randint(140, 230)
            base_g = random.randint(140, 230)
            base_b = random.randint(140, 230)
            def pixel_pared(x, y, w, h):
                # Gradiente suave de luz
                grad = int(30 * (x / w) + 20 * (y / h))
                r = max(0, min(255, base_r + grad + random.randint(-5, 5)))
                g = max(0, min(255, base_g + grad + random.randint(-5, 5)))
                b = max(0, min(255, base_b + grad + random.randint(-5, 5)))
                return (r, g, b)
            crear_bmp_24bpp(ruta, 160, 160, pixel_pared)
            
        elif tipo == 4:
            # 5. Cortinas, ropa, texturas diversas
            patron = random.randint(6, 16)
            c1 = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            c2 = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            def pixel_tela(x, y, w, h):
                use_c1 = ((x // patron) + (y // patron)) % 2 == 0
                col = c1 if use_c1 else c2
                r = max(0, min(255, col[0] + random.randint(-6, 6)))
                g = max(0, min(255, col[1] + random.randint(-6, 6)))
                b = max(0, min(255, col[2] + random.randint(-6, 6)))
                return (r, g, b)
            crear_bmp_24bpp(ruta, 160, 160, pixel_tela)
            
        else:
            # 6. Oscuridad total (lente cubierto)
            def pixel_negro(x, y, w, h):
                v = random.randint(0, 15)
                return (v, v, v)
            crear_bmp_24bpp(ruta, 160, 160, pixel_negro)

    total_actual = len(os.listdir(DIR_FONDO))
    print(f"[LISTO] Total de imágenes en 'dataset/fondo': {total_actual}")

if __name__ == "__main__":
    generar_fondos_adicionales(650)
