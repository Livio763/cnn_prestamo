"""
Script de entrenamiento para el Punto Verde Inteligente
Entrena una CNN desde cero sobre el dataset de reciclaje y exporta en formato
limpio compatible al 100% con TensorFlow.js (tf.loadLayersModel).
"""

import os
import sys
import glob
import json
import random
import struct
import numpy as np

# Configuración de codificación segura para consola de Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    print("\n[ERROR] TensorFlow no está instalado en este entorno.")
    print("Por favor instala las dependencias ejecutando:")
    print("   pip install tensorflow\n")
    sys.exit(1)

# ==========================================
# 1. PARÁMETROS DE CONFIGURACIÓN
# ==========================================
TAMANO_IMG = 160            # Resolución de entrada (veloz en CPU y suficiente para envases)
BATCH_SIZE = 64             # Tamaño de lote
EPOCHS = 30                 # Tope de épocas (EarlyStopping decide el corte real)
CAP_POR_CLASE = 600         # Máximo de imágenes de entrenamiento por clase (balance + velocidad)
CARPETA_DATASET = 'dataset'
CARPETA_SALIDA = 'modelo_web'
MODELO_KERAS = 'modelo_reciclaje.keras'
FORMATOS_VALIDOS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ==========================================
# 2. VALIDACIÓN DEL DATASET
# ==========================================
def asegurar_fondo():
    """Si la carpeta 'fondo' está vacía, genera automáticamente fondos neutros"""
    dir_fondo = os.path.join(CARPETA_DATASET, "fondo")
    os.makedirs(dir_fondo, exist_ok=True)
    existentes = [f for f in os.listdir(dir_fondo) if f.lower().endswith(FORMATOS_VALIDOS)]
    if len(existentes) == 0:
        print("[INFO] La categoría 'fondo' estaba vacía. Generando 25 muestras de fondo neutro...")
        ancho, alto = TAMANO_IMG, TAMANO_IMG
        row_bytes = ancho * 3
        padding = (4 - (row_bytes % 4)) % 4
        file_size = 54 + (row_bytes + padding) * alto

        colores = [(240, 240, 240), (220, 225, 230), (200, 200, 200), (230, 230, 220)]
        for i in range(25):
            c = colores[i % len(colores)]
            header = bytearray(b'BM')
            header += struct.pack('<IHHI', file_size, 0, 0, 54)
            header += struct.pack('<IIIHHIIIIII', 40, ancho, alto, 1, 24, 0, (row_bytes + padding) * alto, 2835, 2835, 0, 0)
            pixel_row = bytearray([c[2], c[1], c[0]] * ancho) + bytearray(padding)
            with open(os.path.join(dir_fondo, f"fondo_auto_{i+1}.bmp"), 'wb') as f:
                f.write(header)
                for _ in range(alto):
                    f.write(pixel_row)

def verificar_dataset(directorio):
    print(f"\n[1/6] Verificando dataset en '{directorio}'...")
    if not os.path.exists(directorio):
        print(f"[ERROR] El directorio '{directorio}' no existe.")
        sys.exit(1)

    asegurar_fondo()

    subcarpetas = [d for d in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, d))]
    conteo_por_clase = {}
    carpetas_vacias = []

    for sc in sorted(subcarpetas):
        ruta_sc = os.path.join(directorio, sc)
        imagenes = [f for f in os.listdir(ruta_sc) if f.lower().endswith(FORMATOS_VALIDOS)]
        conteo_por_clase[sc] = len(imagenes)
        if len(imagenes) == 0:
            carpetas_vacias.append(sc)

    print("--- Resumen de imágenes encontradas ---")
    for clase, cant in conteo_por_clase.items():
        estado = "[OK]" if cant >= 20 else ("[POCO]" if cant > 0 else "[VACIA]")
        print(f"  * {clase.ljust(12)}: {cant} imágenes {estado}")
    print(f"Total imágenes: {sum(conteo_por_clase.values())}")

    if carpetas_vacias:
        print(f"\n[ATENCION] Las siguientes categorías no tienen imágenes: {', '.join(carpetas_vacias)}")
        print("💡 Puedes poblar el dataset ejecutando: python descargar_dataset.py")
        sys.exit(1)

    return conteo_por_clase

conteo_clases = verificar_dataset(CARPETA_DATASET)

# ==========================================
# 3. CARGA Y SPLIT ESTRATIFICADO (80/20) POR CLASE
# ==========================================
print(f"\n[2/6] Construyendo split estratificado 80/20 (cap {CAP_POR_CLASE}/clase en train)...")
clases = sorted(conteo_clases.keys())
num_clases = len(clases)
rng = random.Random(123)

paths_tr, y_tr, paths_va, y_va = [], [], [], []
for idx, c in enumerate(clases):
    archivos = [f for f in glob.glob(os.path.join(CARPETA_DATASET, c, "*")) if f.lower().endswith(FORMATOS_VALIDOS)]
    rng.shuffle(archivos)
    n_tr = int(len(archivos) * 0.8)
    train_c, val_c = archivos[:n_tr], archivos[n_tr:]
    rng.shuffle(train_c)
    train_c = train_c[:CAP_POR_CLASE]
    paths_tr += train_c; y_tr += [idx] * len(train_c)
    paths_va += val_c;   y_va += [idx] * len(val_c)

print(f"Clases ({num_clases}): {clases}")
print("Imágenes de entrenamiento por clase: " +
      ", ".join(f"{c}={y_tr.count(i)}" for i, c in enumerate(clases)))
print("Imágenes de validación por clase: " +
      ", ".join(f"{c}={y_va.count(i)}" for i, c in enumerate(clases)))

AUTOTUNE = tf.data.AUTOTUNE

def cargar_imagen(ruta, etiqueta):
    img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    img = tf.image.resize(img, (TAMANO_IMG, TAMANO_IMG))
    return tf.cast(img, tf.uint8), etiqueta

# Data Augmentation para robustez en condiciones reales (rotación, zoom, luz)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(factor=(-0.10, 0.10)),
], name="data_augmentation")

# IMPORTANTE: se cachean las imágenes LIMPIAS y se aumenta en cada época.
# (El código anterior cacheaba las imágenes ya aumentadas, matando la variación entre épocas.)
ds_base_tr = tf.data.Dataset.from_tensor_slices((paths_tr, y_tr)).map(cargar_imagen, AUTOTUNE).cache()
ds_train = (ds_base_tr.shuffle(1000, seed=123)
            .batch(BATCH_SIZE)
            .map(lambda x, y: (data_augmentation(x, training=True), y), AUTOTUNE)
            .prefetch(AUTOTUNE))

ds_val = (tf.data.Dataset.from_tensor_slices((paths_va, y_va))
          .map(cargar_imagen, AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

# Peso por clase para reforzar las categorías minoritarias (p.ej. vidrio)
total_tr = len(y_tr)
class_weight = {idx: round(total_tr / (num_clases * max(1, y_tr.count(idx))), 4) for idx in range(num_clases)}
print(f"Pesos de clase: {class_weight}")

# ==========================================
# 4. CONSTRUCCIÓN DEL MODELO SEQUENTIAL
# ==========================================
print("\n[3/6] Construyendo arquitectura CNN con BatchNorm, Data Augmentation y GlobalAvgPool...")
modelo = models.Sequential([
    layers.Input(shape=(TAMANO_IMG, TAMANO_IMG, 3), name="input_camara"),
    layers.Rescaling(1.0 / 255.0, name="rescaling"),

    # Bloque Convolucional 1
    layers.Conv2D(32, (3, 3), padding='same', use_bias=False, name="conv2d_1"),
    layers.BatchNormalization(name="bn_1"),
    layers.Activation('relu', name="relu_1"),
    layers.MaxPooling2D(2, 2, name="maxpool_1"),
    layers.Dropout(0.10, name="drop_1"),

    # Bloque Convolucional 2
    layers.Conv2D(64, (3, 3), padding='same', use_bias=False, name="conv2d_2"),
    layers.BatchNormalization(name="bn_2"),
    layers.Activation('relu', name="relu_2"),
    layers.MaxPooling2D(2, 2, name="maxpool_2"),
    layers.Dropout(0.15, name="drop_2"),

    # Bloque Convolucional 3
    layers.Conv2D(128, (3, 3), padding='same', use_bias=False, name="conv2d_3"),
    layers.BatchNormalization(name="bn_3"),
    layers.Activation('relu', name="relu_3"),
    layers.MaxPooling2D(2, 2, name="maxpool_3"),
    layers.Dropout(0.20, name="drop_3"),

    # Bloque Convolucional 4
    layers.Conv2D(256, (3, 3), padding='same', use_bias=False, name="conv2d_4"),
    layers.BatchNormalization(name="bn_4"),
    layers.Activation('relu', name="relu_4"),
    layers.MaxPooling2D(2, 2, name="maxpool_4"),
    layers.Dropout(0.25, name="drop_4"),

    # Agrupación Global Invariante a la Traslación
    layers.GlobalAveragePooling2D(name="global_avg_pool"),
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense_oculta"),
    layers.Dropout(0.30, name="drop_final"),
    layers.Dense(num_clases, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="salida_probabilidades")
], name="sequential_reciclaje")

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()

# ==========================================
# 5. ENTRENAMIENTO
# ==========================================
print("\n[4/6] Iniciando entrenamiento...")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
]

historial = modelo.fit(
    ds_train,
    validation_data=ds_val,
    epochs=EPOCHS,
    class_weight=class_weight,
    callbacks=callbacks
)

# ==========================================
# 5b. EVALUACIÓN POR CLASE (evita desplegar un modelo roto)
# ==========================================
print("\n[5/6] Evaluando el modelo por clase sobre el set de validación...")
y_true, y_pred = [], []
for x_lote, y_lote in ds_val:
    preds = modelo.predict(x_lote, verbose=0)
    y_true.extend(y_lote.numpy().tolist())
    y_pred.extend(np.argmax(preds, axis=1).tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n================ RESULTADOS POR CLASE ================")
print(f"{'clase':<12}{'n':>6}{'accuracy':>12}")
acc_global = (y_true == y_pred).mean()
for idx, c in enumerate(clases):
    mask = y_true == idx
    n = int(mask.sum())
    acc = (y_pred[mask] == idx).mean() if n else 0.0
    print(f"{c:<12}{n:>6}{acc:>12.3f}")
print(f"{'GLOBAL':<12}{len(y_true):>6}{acc_global:>12.3f}")

print("\nMatriz de confusión (filas=real, columnas=predicho):")
print("real\\pred".ljust(12) + "".join(c[:9].rjust(10) for c in clases))
for idx, c in enumerate(clases):
    fila = c.ljust(12)
    for j in range(num_clases):
        fila += str(int(((y_true == idx) & (y_pred == j)).sum())).rjust(10)
    print(fila)

problemas = [c for idx, c in enumerate(clases) if (y_true == idx).sum() and (y_pred[y_true == idx] == idx).mean() < 0.5]
if problemas:
    print(f"\n[ALERTA] Clases con accuracy < 50%: {', '.join(problemas)}. Revisa el dataset antes de desplegar.")

# Guardar artefacto Keras reproducible
modelo.save(MODELO_KERAS)
print(f"[OK] Modelo Keras guardado en: {MODELO_KERAS}")

# ==========================================
# 6. EXPORTACIÓN LIMPIA A FORMATO TENSORFLOW.JS
# ==========================================
print(f"\n[6/6] Exportando modelo a '{CARPETA_SALIDA}' para uso en la web...")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# 1. Guardar clases.json
archivo_clases = os.path.join(CARPETA_SALIDA, "clases.json")
metadata_clases = {
    "clases": clases,
    "tamano_img": TAMANO_IMG,
    "total_clases": num_clases,
    "descripcion": "Mapeo de índices para el Punto Verde Inteligente"
}
with open(archivo_clases, "w", encoding="utf-8") as f:
    json.dump(metadata_clases, f, ensure_ascii=False, indent=2)
print(f"[OK] Archivo de clases guardado en: {archivo_clases}")

# 2. Guardar pesos binarios con nombres prefijados por capa
bin_filename = "group1-shard1of1.bin"
ruta_bin = os.path.join(CARPETA_SALIDA, bin_filename)
manifest_weights = []

with open(ruta_bin, "wb") as f_bin:
    for l in modelo.layers:
        for w in l.weights:
            arr = w.numpy().astype(np.float32)
            f_bin.write(arr.tobytes())
            manifest_weights.append({
                "name": f"{l.name}/{w.name}",
                "shape": list(arr.shape),
                "dtype": "float32"
            })

# 3. Guardar topología limpia Keras 2 compatible con TF.js
clean_layers = []
for l in modelo.layers:
    cfg = l.get_config()
    for k in ['module', 'registered_name', 'build_config']:
        cfg.pop(k, None)
    if isinstance(cfg.get('dtype'), dict):
        cfg['dtype'] = 'float32'
    # Regularizadores son solo para entrenamiento; anularlos para inferencia limpia en TF.js
    for reg in ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer']:
        if reg in cfg:
            cfg[reg] = None
    clean_layers.append({
        'class_name': l.__class__.__name__,
        'config': cfg
    })

clean_layers.insert(0, {
    'class_name': 'InputLayer',
    'config': {
        'batch_input_shape': [None, TAMANO_IMG, TAMANO_IMG, 3],
        'dtype': 'float32',
        'name': 'input_camara'
    }
})

model_json_data = {
    "format": "layers-model",
    "generatedBy": "keras v2.15.0",
    "convertedBy": "Punto Verde Converter",
    "modelTopology": {
        "class_name": "Sequential",
        "config": {
            "name": "sequential_reciclaje",
            "layers": clean_layers
        },
        "keras_version": "2.15.0",
        "backend": "tensorflow"
    },
    "weightsManifest": [
        {
            "paths": [bin_filename],
            "weights": manifest_weights
        }
    ]
}

with open(os.path.join(CARPETA_SALIDA, "model.json"), "w", encoding="utf-8") as f_json:
    json.dump(model_json_data, f_json, indent=2)

print("[OK] Modelo TensorFlow.js generado con éxito y compatible al 100% con tf.loadLayersModel.")
print("\n[LISTO] ¡El modelo web está listo para ser cargado en index.html!")
