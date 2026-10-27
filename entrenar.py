"""
Entrenamiento de la CNN de préstamo de objetos.

- Entrena desde cero (sin pesos preentrenados).
- Lee imágenes de dataset/<clase>/<sesion>/*.jpg
- Genera modelo_web/{model.json, group1-shard1of1.bin, clases.json, metricas.json}
- La web carga ese modelo con TensorFlow.js.

Uso:
    cd prestamo
    python entrenar.py
    python entrenar.py --epochs 40
    python entrenar.py --split aleatorio   # ambos fondos en train y validación

Estructura esperada:
    dataset/
      mouse/
        sesion_01/
        sesion_02/
      calculadora/
      cargador/
      tester/
      usb/
      _desconocido/
"""

import argparse
import datetime
import glob
import json
import os
import random
import shutil
import sys

try:
    import numpy as np
except ImportError:
    print("\n[ERROR] NumPy no está instalado.")
    print("Instala las dependencias con: pip install -r requirements.txt\n")
    sys.exit(1)

# Compatibilidad de consola en Windows.
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError:
    print("\n[ERROR] TensorFlow no está instalado.")
    print("Instala las dependencias con: pip install -r requirements.txt\n")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    Image = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SALIDA_DIR = os.path.join(BASE_DIR, "modelo_web")
MODELO_KERAS = os.path.join(BASE_DIR, "modelo_prestamo.keras")

TAMANO_IMG = 160
BATCH_SIZE = 32
EPOCHS = 30
SEED = 123
FORMATOS_VALIDOS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASE_DESCONOCIDO = "_desconocido"
MIN_FOTOS_POR_CLASE = 10
FOTOS_RECOMENDADAS = 30


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def listar_imagenes(carpeta):
    """Devuelve todas las imágenes de una clase, incluyendo subcarpetas."""
    imagenes = []
    for raiz, _dirs, archivos in os.walk(carpeta):
        for archivo in archivos:
            if archivo.lower().endswith(FORMATOS_VALIDOS):
                imagenes.append(os.path.join(raiz, archivo))
    return sorted(imagenes)


def agrupar_por_sesion(carpeta_clase, imagenes):
    """Agrupa por subcarpeta de sesión. Si no hay subcarpetas, usa 'default'."""
    grupos = {}
    for ruta in imagenes:
        relativa = os.path.relpath(os.path.dirname(ruta), carpeta_clase)
        if relativa == ".":
            grupo = "default"
        else:
            grupo = relativa.split(os.sep)[0]
        grupos.setdefault(grupo, []).append(ruta)
    return grupos


def generar_fondos_fallback(carpeta_desconocido, cantidad=24):
    """Genera fondos neutros si _desconocido está vacío.

    No reemplaza las fotos reales: solo evita que el primer entrenamiento
    falle por no tener la clase de rechazo.
    """
    if Image is None:
        print("[AVISO] Pillow no está instalado; no se pueden generar fondos de respaldo.")
        return 0

    os.makedirs(carpeta_desconocido, exist_ok=True)
    sesiones = [
        os.path.join(carpeta_desconocido, "sesion_01"),
        os.path.join(carpeta_desconocido, "sesion_02"),
    ]
    for sesion in sesiones:
        os.makedirs(sesion, exist_ok=True)

    rng = np.random.default_rng(SEED)
    generadas = 0

    for indice in range(cantidad):
        base = rng.integers(110, 235, size=3)
        gradiente_x = rng.uniform(-25, 25)
        gradiente_y = rng.uniform(-25, 25)
        ruido = rng.normal(0, 7, (TAMANO_IMG, TAMANO_IMG, 3))

        yy, xx = np.mgrid[0:TAMANO_IMG, 0:TAMANO_IMG]
        gradiente = (
            gradiente_x * (xx / TAMANO_IMG)
            + gradiente_y * (yy / TAMANO_IMG)
        )[..., None]

        # Algunas muestras oscuras para simular cámara tapada.
        if indice % 6 == 0:
            base = rng.integers(0, 35, size=3)

        imagen = np.clip(base + gradiente + ruido, 0, 255).astype(np.uint8)
        destino = sesiones[indice % 2]
        ruta = os.path.join(destino, f"fallback_{indice + 1:03d}.jpg")
        Image.fromarray(imagen).save(ruta, "JPEG", quality=85)
        generadas += 1

    print(
        f"[AVISO] dataset/{CLASE_DESCONOCIDO} estaba vacío. "
        f"Se generaron {generadas} fondos neutros de respaldo."
    )
    print("        Agrega fotos reales de escritorio, manos y otros objetos para mejorar el rechazo.")
    print("        Cuando tengas negativos reales, puedes borrar los archivos fallback_*.jpg.")
    return generadas


def preparar_clases(dataset_dir, min_fotos):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"No existe la carpeta de dataset: {dataset_dir}\n"
            "Crea dataset/<objeto>/sesion_01 y coloca allí las fotos."
        )

    clases = sorted(
        nombre
        for nombre in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, nombre)) and not nombre.startswith(".")
    )

    if CLASE_DESCONOCIDO not in clases:
        os.makedirs(os.path.join(dataset_dir, CLASE_DESCONOCIDO), exist_ok=True)
        clases.append(CLASE_DESCONOCIDO)
        clases.sort()

    if len(clases) < 2:
        raise ValueError(
            "Se necesitan al menos 2 clases: un objeto y _desconocido. "
            "Agrega una carpeta por objeto dentro de dataset/."
        )

    clases_utilizables = []

    for clase in clases:
        carpeta = os.path.join(dataset_dir, clase)
        imagenes = listar_imagenes(carpeta)

        if clase == CLASE_DESCONOCIDO and len(imagenes) == 0:
            generar_fondos_fallback(carpeta)
            imagenes = listar_imagenes(carpeta)

        if len(imagenes) == 0:
            print(
                f"[AVISO] La clase '{clase}' no tiene imágenes; se omite. "
                f"Agrega fotos en {carpeta}/sesion_01 y sesion_02."
            )
            continue

        if len(imagenes) < min_fotos:
            print(
                f"[AVISO] La clase '{clase}' solo tiene {len(imagenes)} imágenes "
                f"(mínimo {min_fotos}); se omite. Agrega más fotos o usa --min-fotos."
            )
            continue

        if len(imagenes) < FOTOS_RECOMENDADAS:
            print(
                f"[AVISO] '{clase}' tiene {len(imagenes)} imágenes; "
                f"se recomiendan al menos {FOTOS_RECOMENDADAS}."
            )

        clases_utilizables.append(clase)

    if len(clases_utilizables) < 2:
        raise ValueError(
            "No hay suficientes clases con fotos para entrenar. "
            "Necesitas al menos un objeto con fotos y la clase _desconocido. "
            "Revisa los avisos anteriores y agrega imágenes en dataset/."
        )

    if CLASE_DESCONOCIDO not in clases_utilizables:
        raise ValueError(
            "La clase _desconocido no tiene imágenes. Agrega negativos reales "
            "en dataset/_desconocido/ o instala Pillow para generar los fondos de respaldo."
        )

    return clases_utilizables


def construir_split(dataset_dir, clases, seed=SEED):
    rng = random.Random(seed)
    paths_tr, y_tr, paths_va, y_va = [], [], [], []
    conteos = {}
    sesiones = {}

    for indice, clase in enumerate(clases):
        carpeta = os.path.join(dataset_dir, clase)
        imagenes = listar_imagenes(carpeta)
        conteos[clase] = len(imagenes)

        grupos = agrupar_por_sesion(carpeta, imagenes)
        sesiones[clase] = sorted(grupos.keys())

        if len(grupos) >= 2:
            nombres = sorted(grupos.keys())
            grupo_validacion = nombres[-1]

            train_clase = [
                ruta
                for nombre in nombres
                if nombre != grupo_validacion
                for ruta in grupos[nombre]
            ]
            val_clase = list(grupos[grupo_validacion])

            if not train_clase:
                train_clase, val_clase = [], []
        else:
            print(
                f"[AVISO] '{clase}' no tiene subcarpetas de sesión. "
                "Se hará un split aleatorio 80/20; para una evaluación más honesta, "
                "usa sesion_01 y sesion_02."
            )
            mezcladas = list(imagenes)
            rng.shuffle(mezcladas)
            n_val = max(1, int(len(mezcladas) * 0.2))
            val_clase = mezcladas[:n_val]
            train_clase = mezcladas[n_val:]

        if not val_clase or not train_clase:
            mezcladas = list(imagenes)
            rng.shuffle(mezcladas)
            n_val = max(1, int(len(mezcladas) * 0.2))
            val_clase = mezcladas[:n_val]
            train_clase = mezcladas[n_val:]

        paths_tr.extend(train_clase)
        y_tr.extend([indice] * len(train_clase))
        paths_va.extend(val_clase)
        y_va.extend([indice] * len(val_clase))

    return paths_tr, y_tr, paths_va, y_va, conteos, sesiones


def construir_split_aleatorio(dataset_dir, clases, seed=SEED, fraccion_validacion=0.2):
    """Reparte cada clase y cada sesión entre train y validación.

    Sirve cuando solo hay dos sesiones con fondos/condiciones muy distintas:
    ambas sesiones aportan imágenes a train y a validación. La métrica es más
    optimista que dejar una sesión completa fuera, pero permite que el modelo
    aprenda de los dos fondos en lugar de memorizar solo uno.
    """
    rng = random.Random(seed)
    paths_tr, y_tr, paths_va, y_va = [], [], [], []
    conteos = {}
    sesiones = {}

    for indice, clase in enumerate(clases):
        carpeta = os.path.join(dataset_dir, clase)
        imagenes = listar_imagenes(carpeta)
        conteos[clase] = len(imagenes)

        grupos = agrupar_por_sesion(carpeta, imagenes)
        sesiones[clase] = sorted(grupos.keys())
        train_clase, val_clase = [], []

        for nombre in sorted(grupos):
            grupo = list(grupos[nombre])
            rng.shuffle(grupo)
            n_val = max(1, int(round(len(grupo) * fraccion_validacion)))
            if n_val >= len(grupo):
                n_val = max(0, len(grupo) - 1)

            val_clase.extend(grupo[:n_val])
            train_clase.extend(grupo[n_val:])

        if not val_clase or not train_clase:
            mezcladas = list(imagenes)
            rng.shuffle(mezcladas)
            n_val = max(1, int(round(len(mezcladas) * fraccion_validacion)))
            n_val = min(n_val, max(0, len(mezcladas) - 1))
            val_clase = mezcladas[:n_val]
            train_clase = mezcladas[n_val:]

        paths_tr.extend(train_clase)
        y_tr.extend([indice] * len(train_clase))
        paths_va.extend(val_clase)
        y_va.extend([indice] * len(val_clase))

    return paths_tr, y_tr, paths_va, y_va, conteos, sesiones


# ---------------------------------------------------------------------------
# Modelo y entrenamiento
# ---------------------------------------------------------------------------


def cargar_imagen(ruta, etiqueta):
    imagen = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    forma = tf.shape(imagen)
    h, w = forma[0], forma[1]
    lado = tf.minimum(h, w)
    offset_h = (h - lado) // 2
    offset_w = (w - lado) // 2
    imagen = tf.image.crop_to_bounding_box(imagen, offset_h, offset_w, lado, lado)
    imagen = tf.image.resize(imagen, (TAMANO_IMG, TAMANO_IMG))
    return tf.cast(imagen, tf.uint8), etiqueta


def crear_augmentation():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.05, 0.05),
            layers.RandomBrightness(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def construir_modelo(num_clases, lr=1e-3):
    modelo = models.Sequential(
        [
            layers.Input(shape=(TAMANO_IMG, TAMANO_IMG, 3), name="entrada_camara"),
            layers.Rescaling(1.0 / 255.0, name="rescaling"),
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                name="conv_1",
            ),
            layers.MaxPooling2D(2, 2, name="pool_1"),
            layers.Dropout(0.05, name="drop_1"),
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                name="conv_2",
            ),
            layers.MaxPooling2D(2, 2, name="pool_2"),
            layers.Dropout(0.10, name="drop_2"),
            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                name="conv_3",
            ),
            layers.MaxPooling2D(2, 2, name="pool_3"),
            layers.Dropout(0.12, name="drop_3"),
            layers.Conv2D(
                256,
                (3, 3),
                padding="same",
                activation="relu",
                kernel_initializer="he_normal",
                name="conv_4",
            ),
            layers.MaxPooling2D(2, 2, name="pool_4"),
            layers.Dropout(0.15, name="drop_4"),
            layers.GlobalAveragePooling2D(name="global_avg_pool"),
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name="dense_oculta",
            ),
            layers.Dropout(0.25, name="drop_final"),
            layers.Dense(
                num_clases,
                activation="softmax",
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                name="salida_clases",
            ),
        ],
        name="cnn_prestamo",
    )

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def evaluar(modelo, ds_val, clases):
    y_true, y_pred = [], []

    for x_lote, y_lote in ds_val:
        predicciones = modelo.predict(x_lote, verbose=0)
        y_true.extend(y_lote.numpy().tolist())
        y_pred.extend(np.argmax(predicciones, axis=1).tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        raise ValueError("El conjunto de validación está vacío. Agrega más fotos.")

    accuracy_global = float((y_true == y_pred).mean())
    accuracy_por_clase = {}
    conteos_validacion = {}

    for indice, clase in enumerate(clases):
        mascara = y_true == indice
        n = int(mascara.sum())
        conteos_validacion[clase] = n
        accuracy_por_clase[clase] = float((y_pred[mascara] == indice).mean()) if n else 0.0

    matriz = {
        clase: {otra: 0 for otra in clases}
        for clase in clases
    }

    for real, predicho in zip(y_true, y_pred):
        matriz[clases[int(real)]][clases[int(predicho)]] += 1

    return {
        "accuracy_global": accuracy_global,
        "accuracy_por_clase": accuracy_por_clase,
        "conteos_validacion": conteos_validacion,
        "matriz_confusion": matriz,
        "total_validacion": int(len(y_true)),
    }


def imprimir_resultados(metricas, clases):
    print("\n================ RESULTADOS POR CLASE ================")
    print(f"{'clase':<18}{'n':>6}{'accuracy':>12}")

    for clase in clases:
        n = metricas["conteos_validacion"].get(clase, 0)
        accuracy = metricas["accuracy_por_clase"].get(clase, 0.0)
        print(f"{clase:<18}{n:>6}{accuracy:>12.3f}")

    print(f"{'GLOBAL':<18}{metricas['total_validacion']:>6}{metricas['accuracy_global']:>12.3f}")

    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print("real\\pred".ljust(18) + "".join(clase[:10].rjust(11) for clase in clases))
    for clase in clases:
        fila = clase.ljust(18)
        for otra in clases:
            fila += str(metricas["matriz_confusion"][clase][otra]).rjust(11)
        print(fila)

    problemas = [
        clase
        for clase in clases
        if metricas["conteos_validacion"].get(clase, 0) > 0
        and metricas["accuracy_por_clase"].get(clase, 0.0) < 0.5
    ]

    if problemas:
        print(
            "\n[ALERTA] Clases con accuracy < 50%: "
            + ", ".join(problemas)
            + ". Agrega más fotos y revisa la matriz de confusión."
        )


# ---------------------------------------------------------------------------
# Exportación a TensorFlow.js
# ---------------------------------------------------------------------------


def exportar_tfjs(modelo, clases, carpeta, conteos, metricas):
    os.makedirs(carpeta, exist_ok=True)

    with open(os.path.join(carpeta, "clases.json"), "w", encoding="utf-8") as archivo:
        json.dump(
            {
                "clases": clases,
                "tamano_img": TAMANO_IMG,
                "total_clases": len(clases),
                "descripcion": "Clases del sistema de préstamo de objetos",
                "conteos": conteos,
            },
            archivo,
            ensure_ascii=False,
            indent=2,
        )

    with open(os.path.join(carpeta, "metricas.json"), "w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, ensure_ascii=False, indent=2)

    bin_filename = "group1-shard1of1.bin"
    manifest = []

    with open(os.path.join(carpeta, bin_filename), "wb") as archivo_bin:
        for capa in modelo.layers:
            if isinstance(capa, tf.keras.layers.InputLayer):
                continue
            for peso in capa.weights:
                arreglo = peso.numpy().astype(np.float32)
                archivo_bin.write(arreglo.tobytes())
                nombre_peso = peso.name
                if nombre_peso.endswith(":0"):
                    nombre_peso = nombre_peso[:-2]
                if not nombre_peso.startswith(f"{capa.name}/"):
                    nombre_peso = f"{capa.name}/{nombre_peso}"
                manifest.append(
                    {
                        "name": nombre_peso,
                        "shape": list(arreglo.shape),
                        "dtype": "float32",
                    }
                )

    clean_layers = []

    for capa in modelo.layers:
        if isinstance(capa, tf.keras.layers.InputLayer):
            continue

        config = capa.get_config()
        for clave in ["module", "registered_name", "build_config"]:
            config.pop(clave, None)

        if isinstance(config.get("dtype"), dict):
            config["dtype"] = "float32"

        for regularizador in [
            "kernel_regularizer",
            "bias_regularizer",
            "activity_regularizer",
        ]:
            if regularizador in config:
                config[regularizador] = None

        clean_layers.append(
            {
                "class_name": capa.__class__.__name__,
                "config": config,
            }
        )

    clean_layers.insert(
        0,
        {
            "class_name": "InputLayer",
            "config": {
                "batch_input_shape": [None, TAMANO_IMG, TAMANO_IMG, 3],
                "dtype": "float32",
                "name": "entrada_camara",
            },
        },
    )

    data = {
        "format": "layers-model",
        "generatedBy": "keras",
        "convertedBy": "prestamo_entrenar.py",
        "modelTopology": {
            "class_name": "Sequential",
            "config": {
                "name": "cnn_prestamo",
                "layers": clean_layers,
            },
            "keras_version": "2.15.0",
            "backend": "tensorflow",
        },
        "weightsManifest": [
            {
                "paths": [bin_filename],
                "weights": manifest,
            }
        ],
    }

    with open(os.path.join(carpeta, "model.json"), "w", encoding="utf-8") as archivo_json:
        json.dump(data, archivo_json, indent=2)


def publicar_modelo(modelo, clases, salida_dir, conteos, metricas):
    temporal = salida_dir + "_tmp"

    if os.path.exists(temporal):
        shutil.rmtree(temporal)

    exportar_tfjs(modelo, clases, temporal, conteos, metricas)

    os.makedirs(salida_dir, exist_ok=True)

    # Evitar restos de una exportación anterior con otro número de shards.
    for viejo in glob.glob(os.path.join(salida_dir, "group1-*.bin")):
        os.remove(viejo)

    for nombre in os.listdir(temporal):
        shutil.move(
            os.path.join(temporal, nombre),
            os.path.join(salida_dir, nombre),
        )

    shutil.rmtree(temporal, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def entrenar(args):
    dataset_dir = os.path.abspath(args.dataset)
    salida_dir = os.path.abspath(args.salida)
    modelo_keras = os.path.abspath(args.modelo)

    print("=" * 62)
    print("  Entrenamiento CNN - Préstamo Inteligente")
    print("=" * 62)
    print(f"Dataset: {dataset_dir}")
    print(f"Salida:  {salida_dir}")
    print(f"Épocas:  {args.epochs}")
    print(f"Batch:   {args.batch_size}")
    print(f"Split:   {args.split}")

    clases = preparar_clases(dataset_dir, args.min_fotos)

    print(f"\nClases ({len(clases)}): {', '.join(clases)}")

    if args.split == "aleatorio":
        print(
            "\n[AVISO] Split aleatorio estratificado por clase y sesión: "
            "ambos fondos entran en train y validación. La métrica es optimista; "
            "para una prueba honesta usa --split sesion o una sesión nueva."
        )
        paths_tr, y_tr, paths_va, y_va, conteos, sesiones = construir_split_aleatorio(
            dataset_dir, clases, seed=SEED
        )
    else:
        paths_tr, y_tr, paths_va, y_va, conteos, sesiones = construir_split(
            dataset_dir, clases, seed=SEED
        )

    if not paths_tr or not paths_va:
        raise ValueError("No se pudo construir el split de entrenamiento/validación.")

    print("\nSesiones detectadas:")
    for clase in clases:
        print(f"  {clase:<18} {', '.join(sesiones[clase])}")

    print("\nImágenes:")
    for indice, clase in enumerate(clases):
        print(
            f"  {clase:<18} total={conteos[clase]:>4}  "
            f"train={y_tr.count(indice):>4}  val={y_va.count(indice):>4}"
        )

    autotune = tf.data.AUTOTUNE
    data_augmentation = crear_augmentation()

    ds_base_tr = (
        tf.data.Dataset.from_tensor_slices((paths_tr, y_tr))
        .map(cargar_imagen, autotune)
        .cache()
    )

    ds_train = (
        ds_base_tr.shuffle(min(len(paths_tr), 1000), seed=SEED)
        .batch(args.batch_size)
        .map(lambda x, y: (data_augmentation(x, training=True), y), autotune)
        .prefetch(autotune)
    )

    ds_val = (
        tf.data.Dataset.from_tensor_slices((paths_va, y_va))
        .map(cargar_imagen, autotune)
        .cache()
        .batch(args.batch_size)
        .prefetch(autotune)
    )

    total_tr = len(y_tr)
    class_weight = {
        indice: round(total_tr / (len(clases) * max(1, y_tr.count(indice))), 4)
        for indice in range(len(clases))
    }

    if getattr(args, "boost_dificiles", False):
        print("\n[INFO] Modo boost-dificiles activo: reforzando peso de clases 'cargador' y 'usb'.")
        for i, c in enumerate(clases):
            if c in ("cargador", "usb"):
                class_weight[i] = round(class_weight[i] * 2.5, 4)
            elif c == "_desconocido":
                class_weight[i] = round(class_weight[i] * 1.5, 4)

    print(f"\nPesos de clase: {class_weight}")

    lr = args.lr if args.lr is not None else (3e-4 if args.continuar else 1e-3)

    if args.continuar and os.path.exists(modelo_keras):
        print(f"\n[INFO] Continuando entrenamiento desde modelo existente: {modelo_keras}")
        try:
            modelo = tf.keras.models.load_model(modelo_keras)
            modelo.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            print(f"[OK] Modelo cargado con éxito. LR={lr}")
        except Exception as e:
            print(f"[AVISO] Error al cargar modelo existente ({e}). Creando modelo nuevo.")
            modelo = construir_modelo(len(clases), lr=lr)
    else:
        modelo = construir_modelo(len(clases), lr=lr)

    modelo.summary()

    checkpoint_path = modelo_keras + ".best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.6,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nEntrenando...")
    historial = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    if os.path.exists(checkpoint_path):
        try:
            print("\n[INFO] Restaurando mejores pesos guardados por ModelCheckpoint...")
            modelo = tf.keras.models.load_model(checkpoint_path)
            os.remove(checkpoint_path)
        except Exception:
            pass

    print("\nEvaluando...")
    metricas = evaluar(modelo, ds_val, clases)
    metricas.update(
        {
            "fecha": datetime.datetime.now().isoformat(timespec="seconds"),
            "tamano_img": TAMANO_IMG,
            "split": args.split,
            "clases": clases,
            "conteos": conteos,
            "sesiones": sesiones,
            "epochs_entrenados": len(historial.history.get("loss", [])),
        }
    )

    imprimir_resultados(metricas, clases)

    modelo.save(modelo_keras)
    print(f"\n[OK] Modelo Keras: {modelo_keras}")

    publicar_modelo(modelo, clases, salida_dir, conteos, metricas)
    print(f"[OK] Modelo web actualizado en: {salida_dir}")
    print("[OK] clases.json, model.json, group1-shard1of1.bin y metricas.json")
    print("\nAbre o recarga la web y pulsa 'Recargar modelo'.")


def main():
    parser = argparse.ArgumentParser(
        description="Entrena una CNN desde cero para el sistema de préstamo."
    )
    parser.add_argument("--dataset", default=DATASET_DIR, help="Carpeta del dataset.")
    parser.add_argument("--salida", default=SALIDA_DIR, help="Carpeta de salida para TensorFlow.js.")
    parser.add_argument("--modelo", default=MODELO_KERAS, help="Ruta del modelo Keras.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Máximo de épocas.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Tamaño de lote.")
    parser.add_argument(
        "--split",
        choices=("sesion", "aleatorio"),
        default="sesion",
        help=(
            "Separación de validación: 'sesion' deja fuera la última sesión; "
            "'aleatorio' reparte cada sesión 80/20."
        ),
    )
    parser.add_argument(
        "--min-fotos",
        type=int,
        default=MIN_FOTOS_POR_CLASE,
        help="Mínimo de fotos por clase para permitir el entrenamiento.",
    )
    parser.add_argument(
        "--continuar",
        action="store_true",
        help="Continúa el entrenamiento desde el modelo Keras existente.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate inicial personalizado.",
    )
    parser.add_argument(
        "--boost-dificiles",
        action="store_true",
        help="Refuerza las clases difíciles (cargador y usb) con mayor peso.",
    )
    args = parser.parse_args()

    try:
        entrenar(args)
    except Exception as error:
        print(f"\n[ERROR] {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
