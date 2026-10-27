"""
Entrenamiento CONTINUO para el Punto Verde Inteligente.

Objetivo: seguir mejorando el modelo SIN romper la version estable desplegada.

- Hace warm-start desde `modelo_reciclaje.keras` (la version estable actual).
- Entrena con el mismo split de validacion (seed 123) para comparar de forma justa.
- Evalua candidato vs estable por clase.
- SOLO promueve a `modelo_web/` si el candidato mejora el score y no degrada fondo/vidrio.
- Si no mejora, la version estable queda intacta.

Uso:  python entrenar_continuo.py
"""

import os
import sys
import glob
import json
import random
import shutil
import datetime

import numpy as np

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import tensorflow as tf
from tensorflow.keras import layers

# ==========================================
# CONFIGURACION
# ==========================================
TAMANO_IMG = 160
BATCH_SIZE = 64
EPOCHS_CONT = 20            # epocas de continuacion (EarlyStopping decide el corte)
CAP_CONT = 800             # imagenes de entrenamiento por clase en esta continuacion
LR_CONT = 3e-4             # learning rate de fine-tuning
CARPETA_DATASET = 'dataset'
CARPETA_SALIDA = 'modelo_web'
MODELO_KERAS = 'modelo_reciclaje.keras'
CARPETA_VERSIONES = 'versiones'
FORMATOS_VALIDOS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Reglas de promocion
FONDO_MINIMO = 0.90        # no promover si el fondo se degrada por debajo de esto
VIDRIO_TOLERANCIA = 0.02   # permitir como maximo esta caida de vidrio
MEJORA_MINIMA = 0.003      # mejora minima del score para promover

os.makedirs(CARPETA_VERSIONES, exist_ok=True)


def score(m):
    """Score con foco en envases/botellas (plastico y vidrio pesan mas)."""
    return 0.15 * m['fondo'] + 0.25 * m['lata'] + 0.30 * m['plastico'] + 0.30 * m['vidrio']


# ==========================================
# DATASET: split identico a entrenar.py (seed 123) para comparacion justa
# ==========================================
def construir_split(cap_train):
    clases = sorted([d for d in os.listdir(CARPETA_DATASET)
                     if os.path.isdir(os.path.join(CARPETA_DATASET, d))])
    rng = random.Random(123)
    paths_tr, y_tr, paths_va, y_va = [], [], [], []
    for idx, c in enumerate(clases):
        archivos = [f for f in glob.glob(os.path.join(CARPETA_DATASET, c, "*"))
                    if f.lower().endswith(FORMATOS_VALIDOS)]
        rng.shuffle(archivos)
        n_tr = int(len(archivos) * 0.8)
        train_c, val_c = archivos[:n_tr], archivos[n_tr:]
        rng.shuffle(train_c)
        train_c = train_c[:cap_train]
        paths_tr += train_c; y_tr += [idx] * len(train_c)
        paths_va += val_c;   y_va += [idx] * len(val_c)
    return clases, paths_tr, y_tr, paths_va, y_va


clases, paths_tr, y_tr, paths_va, y_va = construir_split(CAP_CONT)
num_clases = len(clases)
print(f"Clases: {clases}")
print("Train/clase: " + ", ".join(f"{c}={y_tr.count(i)}" for i, c in enumerate(clases)))
print("Val/clase:   " + ", ".join(f"{c}={y_va.count(i)}" for i, c in enumerate(clases)))

AUTOTUNE = tf.data.AUTOTUNE


def cargar_imagen(ruta, etiqueta):
    img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    img = tf.image.resize(img, (TAMANO_IMG, TAMANO_IMG))
    return tf.cast(img, tf.uint8), etiqueta


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(factor=(-0.10, 0.10)),
], name="data_augmentation")

ds_base_tr = tf.data.Dataset.from_tensor_slices((paths_tr, y_tr)).map(cargar_imagen, AUTOTUNE).cache()
ds_train = (ds_base_tr.shuffle(1000, seed=123)
            .batch(BATCH_SIZE)
            .map(lambda x, y: (data_augmentation(x, training=True), y), AUTOTUNE)
            .prefetch(AUTOTUNE))

ds_val = (tf.data.Dataset.from_tensor_slices((paths_va, y_va))
          .map(cargar_imagen, AUTOTUNE).cache().batch(BATCH_SIZE).prefetch(AUTOTUNE))

total_tr = len(y_tr)
class_weight = {idx: round(total_tr / (num_clases * max(1, y_tr.count(idx))), 4) for idx in range(num_clases)}
print(f"Pesos de clase: {class_weight}")


def evaluar(modelo):
    y_true, y_pred = [], []
    for x_lote, y_lote in ds_val:
        preds = modelo.predict(x_lote, verbose=0)
        y_true.extend(y_lote.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = {}
    for idx, c in enumerate(clases):
        mask = y_true == idx
        m[c] = float((y_pred[mask] == idx).mean()) if mask.sum() else 0.0
    m['global'] = float((y_true == y_pred).mean())
    return m, y_true, y_pred


def imprimir_metricas(titulo, m, y_true=None, y_pred=None):
    print(f"\n=== {titulo} ===")
    for idx, c in enumerate(clases):
        print(f"  {c:<10} {m[c]:.3f}")
    print(f"  {'GLOBAL':<10} {m['global']:.3f}  (score={score(m):.3f})")
    if y_true is not None:
        print("  matriz (real\\pred): " + "".join(c[:8].rjust(9) for c in clases))
        for idx, c in enumerate(clases):
            fila = "".join(str(int(((y_true == idx) & (y_pred == j)).sum())).rjust(9) for j in range(num_clases))
            print(f"  {c:<14}{fila}")


# ==========================================
# CARGA DEL MODELO ESTABLE (warm-start)
# ==========================================
if not os.path.exists(MODELO_KERAS):
    print(f"[ERROR] No existe {MODELO_KERAS}. Entrena primero con entrenar.py")
    sys.exit(1)

print(f"\nCargando modelo estable desde {MODELO_KERAS}...")
modelo = tf.keras.models.load_model(MODELO_KERAS)
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_CONT),
               loss='sparse_categorical_crossentropy', metrics=['accuracy'])

metricas_base, yt, yp = evaluar(modelo)
imprimir_metricas("BASELINE (estable)", metricas_base, yt, yp)

# ==========================================
# CONTINUAR ENTRENAMIENTO
# ==========================================
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
mejor_ckpt = os.path.join(CARPETA_VERSIONES, f"_candidato_{ts}.keras")

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(mejor_ckpt, monitor='val_loss',
                                       save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=3, min_lr=1e-6, verbose=1),
]

print(f"\nContinuando entrenamiento ({EPOCHS_CONT} epocas max, lr={LR_CONT})...")
modelo.fit(ds_train, validation_data=ds_val, epochs=EPOCHS_CONT,
           class_weight=class_weight, callbacks=callbacks)

if os.path.exists(mejor_ckpt):
    modelo = tf.keras.models.load_model(mejor_ckpt)

metricas_cand, yt2, yp2 = evaluar(modelo)
imprimir_metricas("CANDIDATO", metricas_cand, yt2, yp2)


# ==========================================
# EXPORTACION TF.js (a carpeta temporal, luego swap atomico)
# ==========================================
def exportar_tfjs(modelo, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "clases.json"), "w", encoding="utf-8") as f:
        json.dump({"clases": clases, "tamano_img": TAMANO_IMG, "total_clases": num_clases,
                   "descripcion": "Mapeo de indices para el Punto Verde Inteligente"},
                  f, ensure_ascii=False, indent=2)

    bin_filename = "group1-shard1of1.bin"
    manifest = []
    with open(os.path.join(carpeta, bin_filename), "wb") as fb:
        for l in modelo.layers:
            for w in l.weights:
                arr = w.numpy().astype(np.float32)
                fb.write(arr.tobytes())
                manifest.append({"name": f"{l.name}/{w.name}", "shape": list(arr.shape), "dtype": "float32"})

    clean_layers = []
    for l in modelo.layers:
        cfg = l.get_config()
        for k in ['module', 'registered_name', 'build_config']:
            cfg.pop(k, None)
        if isinstance(cfg.get('dtype'), dict):
            cfg['dtype'] = 'float32'
        for reg in ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer']:
            if reg in cfg:
                cfg[reg] = None
        clean_layers.append({'class_name': l.__class__.__name__, 'config': cfg})
    clean_layers.insert(0, {'class_name': 'InputLayer', 'config': {
        'batch_input_shape': [None, TAMANO_IMG, TAMANO_IMG, 3], 'dtype': 'float32', 'name': 'input_camara'}})

    data = {"format": "layers-model", "generatedBy": "keras v2.15.0", "convertedBy": "Punto Verde Converter",
            "modelTopology": {"class_name": "Sequential",
                              "config": {"name": "sequential_reciclaje", "layers": clean_layers},
                              "keras_version": "2.15.0", "backend": "tensorflow"},
            "weightsManifest": [{"paths": [bin_filename], "weights": manifest}]}
    with open(os.path.join(carpeta, "model.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ==========================================
# DECISION DE PROMOCION
# ==========================================
promover = (
    score(metricas_cand) > score(metricas_base) + MEJORA_MINIMA
    and metricas_cand['fondo'] >= FONDO_MINIMO
    and metricas_cand['vidrio'] >= metricas_base['vidrio'] - VIDRIO_TOLERANCIA
)

log = {
    "fecha": ts,
    "baseline": metricas_base,
    "candidato": metricas_cand,
    "score_baseline": score(metricas_base),
    "score_candidato": score(metricas_cand),
    "promovido": bool(promover),
}

if promover:
    print("\n[PROMOVER] El candidato mejora. Actualizando version desplegada...")
    modelo.save(MODELO_KERAS)
    tmp = CARPETA_SALIDA + "_tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    exportar_tfjs(modelo, tmp)
    for f in os.listdir(tmp):
        shutil.move(os.path.join(tmp, f), os.path.join(CARPETA_SALIDA, f))
    shutil.rmtree(tmp, ignore_errors=True)
    print("[OK] modelo_web/ actualizado. Recarga la app en el movil.")
else:
    print("\n[CONSERVAR] El candidato NO supera a la version estable. No se toca modelo_web/.")
    # dejar el candidato rechazado archivado para inspeccion
    if os.path.exists(mejor_ckpt):
        shutil.move(mejor_ckpt, os.path.join(CARPETA_VERSIONES, f"candidato_rechazado_{ts}.keras"))

with open(os.path.join(CARPETA_VERSIONES, "historial.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps(log, ensure_ascii=False) + "\n")

print(f"\n[FIN] Registro guardado en {CARPETA_VERSIONES}/historial.jsonl")
