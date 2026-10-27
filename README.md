# Préstamo Inteligente

Sistema de préstamo de objetos con reconocimiento visual. La cámara identifica un objeto, la web consulta su stock y permite **prestar** o **devolver**. Todo se guarda en **IndexedDB**.

- Modelo: CNN propia entrenada **desde cero** con TensorFlow (sin pesos preentrenados).
- Inferencia: TensorFlow.js en el navegador.
- Base de datos: IndexedDB (`prestamo_inteligente`).
- Interfaz: HTML, CSS y JavaScript, sin frameworks ni build step.

## Inicio rápido

```bash
cd prestamo
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1: servidor web
python servidor.py
# abrir http://localhost:8001

# Terminal 2: entrenamiento (cuando haya fotos)
python entrenar.py --split aleatorio
```

Este proyecto usa **TensorFlow 2.15.1** para que el exportador TF.js artesanal siga funcionando. Ese TensorFlow requiere **Python 3.9–3.11**; se recomienda **3.11**. Si tu `python` es 3.12/3.13/3.14, crea el entorno con una versión compatible (por ejemplo, `python3.11 -m venv .venv`).

La interfaz es **mobile-first**: barra de navegación inferior fija, controles táctiles de 44–46 px, cámara 4:3, formularios a una columna y diálogos tipo *bottom sheet*. En pantallas de 780 px o más la navegación pasa arriba, aparecen dos columnas y se muestra el subtítulo; el uso en PC está soportado, pero no es el objetivo principal. Al reconocer un objeto en móvil, la vista se desplaza automáticamente al resultado.

La cámara solo funciona en `localhost` o HTTPS. Si abres el archivo con `file://`, el navegador bloqueará la cámara.

La primera carga de la web necesita internet para descargar TensorFlow.js desde el CDN. Si no hay internet, descarga `tf.min.js` y cámbialo por una ruta local.

## Estructura

```text
prestamo/
├── index.html
├── css/styles.css
├── js/db.js              # IndexedDB: objetos, préstamos, movimientos
├── js/app.js             # cámara, inferencia, UI
├── dataset/
│   ├── _desconocido/
│   ├── mouse/
│   ├── calculadora/
│   ├── cargador/
│   ├── tester/
│   └── usb/
├── modelo_web/           # generado por entrenar.py
├── entrenar.py           # CNN desde cero
├── servidor.py           # static server en :8001
└── requirements.txt
```

## Clases iniciales

La web crea automáticamente estas fichas en IndexedDB:

| ID | Nombre | Stock |
|---|---|---:|
| `mouse` | Mouse | 1 |
| `calculadora` | Calculadora | 1 |
| `cargador` | Cargador | 2 |
| `tester` | Tester | 1 |
| `usb` | USB | 1 |

El modelo debe incluir además la clase `_desconocido` para rechazar fondos, manos y objetos que no están en el catálogo.

La estructura trae 5 objetos + `_desconocido` (6 clases). Si prefieres exactamente 5 clases, deja `dataset/usb/` vacío: el entrenamiento omite las clases sin fotos. El USB seguirá en el inventario como “Sin entrenar” y puedes prestarlo manualmente.

## Cómo agregar fotos

Coloca las imágenes así:

```text
dataset/mouse/sesion_01/foto_001.jpg
dataset/mouse/sesion_02/foto_001.jpg
```

Recomendaciones:

- 40–60 fotos por objeto.
- Dos sesiones con luz/fondo distintos.
- Ángulos variados, con y sin mano.
- Objeto ocupando 60–80% del encuadre.
- Sin blur ni reflejos fuertes.
- 100+ imágenes variadas en `_desconocido`.

Puedes grabar un video de 20–30 segundos girando el objeto y extraer 1 frame cada 0.5–1 segundo.

El split de entrenamiento/validación se hace **por sesión** por defecto. No mezcles frames del mismo video entre train y validación.

Si solo tienes dos sesiones y los fondos son muy distintos, el split por sesión puede castigar demasiado al modelo: aprende un fondo y falla en el otro. Para un primer modelo desplegable, reparte cada sesión 80/20 con `--split aleatorio`. Esa validación es optimista; para medir generalización real, añade una tercera sesión y usa el split por sesión.

## Entrenar

```bash
cd prestamo
python entrenar.py
```

Opciones:

```bash
python entrenar.py --epochs 40
python entrenar.py --min-fotos 20
python entrenar.py --split aleatorio
python entrenar.py --dataset ruta/al/dataset --salida ruta/modelo_web
```

El script:

1. Valida clases y fotos.
2. Omite con un aviso las clases que todavía no tienen fotos suficientes.
3. Genera fondos neutros si `_desconocido` está vacío.
4. Separa train/validación por sesión (o 80/20 dentro de cada sesión con `--split aleatorio`).
5. Aplica augmentation solo a entrenamiento.
6. Entrena una CNN de 4 bloques desde cero.
7. Evalúa accuracy por clase y matriz de confusión.
8. Guarda `modelo_prestamo.keras`.
9. Exporta `modelo_web/` para TensorFlow.js.

Al terminar, recarga la web y pulsa **Recargar modelo**.

### Métricas

El script imprime accuracy por clase y matriz de confusión. No actives un modelo si una clase importante queda muy baja. Revisa especialmente:

- `_desconocido` (rechazo de objetos no registrados),
- pares que se confunden entre sí (por ejemplo, cargador vs tester),
- clases con pocas fotos.

## Agregar una clase nueva

1. En la web, pestaña **Agregar objeto**: nombre, descripción, categoría y stock.
2. La web muestra el ID y la ruta `dataset/<id>/`.
3. Crea `sesion_01` y `sesion_02`, copia 40–60 fotos.
4. Ejecuta `python entrenar.py`.
5. Recarga la web y pulsa **Recargar modelo**.
6. El objeto queda disponible para préstamo.

El clasificador tiene clases fijas: al agregar un objeto hay que **reentrenar**. Para 5–6 clases el entrenamiento es rápido. No hay aprendizaje incremental.

## Préstamos y stock

- **Prestar**: descuenta 1 unidad disponible y crea un préstamo abierto.
- **Devolver**: suma 1 unidad y cierra el préstamo abierto (si existe).
- El reconocimiento **no** descuenta stock por sí solo.
- Cada operación registra un movimiento en el historial.
- Un objeto con stock 1 funciona perfectamente: disponible pasa de 1 a 0 y viceversa.

Los datos viven en IndexedDB. Si borras los datos del sitio, se pierden. Para una demo es suficiente; para producción conviene migrar a SQLite/Postgres con un backend.

## Contrato del modelo

- Entrada: 160×160 RGB, píxeles 0–255.
- El `Rescaling(1/255)` está dentro del modelo.
- Salida: softmax con una neurona por clase.
- `clases.json` define el orden exacto de las clases.
- La web recorta el 70% central del video antes de predecir.
- Si la confianza es menor a `0.40`, o el margen entre top-1 y top-2 es menor a `0.05`, se muestra “No reconocido”.

No edites `clases.json` a mano. Si cambias las carpetas del dataset, reentrena.

## Limitaciones

- Sin backend ni autenticación. Es una app local de una sola persona/tienda.
- Los datos no se sincronizan entre navegadores.
- El modelo desde cero necesita fotos reales; el dataset público solo sirve como complemento.
- Objetos transparentes, reflectantes o muy pequeños son más difíciles.
- Un clasificador con clases fijas no reconoce un objeto nuevo hasta reentrenar.

## Solución de problemas

**La cámara no abre**
- Usa `http://localhost:8001`, no `file://`.
- Revisa permisos de cámara del navegador.

**“Modelo: sin entrenar”**
- Falta `modelo_web/clases.json` o `modelo_web/model.json`.
- Ejecuta `python entrenar.py`.
- Revisa que `group1-shard1of1.bin` exista.

**El modelo se entrena pero no carga en la web**
- El exportador TF.js es artesanal. Si usas una versión muy nueva de TensorFlow/Keras, prueba con `tensorflow==2.15.1` y vuelve a ejecutar `python entrenar.py`.
- Revisa la consola del navegador: si falta el `.bin`, el modelo no puede cargar.

**El modelo carga pero reconoce mal**
- Agrega más fotos y más variedad.
- Revisa la matriz de confusión.
- Agrega negativos a `_desconocido`.
- Verifica que las fotos de validación sean de otra sesión. Si usaste `--split aleatorio`, prueba también con fotos nuevas de un fondo distinto: esa validación es optimista.

**Quiero reiniciar el inventario**
- Borra los datos del sitio en el navegador (IndexedDB `prestamo_inteligente`).
- Al recargar se vuelven a crear los objetos iniciales.
