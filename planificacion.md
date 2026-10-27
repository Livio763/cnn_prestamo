# Planificación — Punto Verde Store (Registradora con IA)

> Pivote del proyecto: se deja a un lado el módulo de reciclaje (lata/plástico/vidrio) y se
> construye una **registradora de tienda** con reconocimiento de productos por cámara.
> La versión estable anterior del reciclaje queda archivada en `versiones/`.

---

## 1. Objetivo

Una **web móvil + backend local** que funcione como registradora de tienda:

- **Vender rápido**: apuntar la cámara a un producto → reconocerlo → mostrar modal con precio → aceptar la venta (descuenta stock).
- **Registrar productos nuevos** con la cámara (sin reentrenar el modelo).
- **Agregar stock** escaneando o buscando un producto.
- **Inventario** con existencias, alertas de stock bajo e historial de ventas.
- Moneda: **Bolivianos (Bs)**.

---

## 2. Decisión clave: cómo reconocer productos sin reentrenar

El modelo actual solo distingue **materiales** (lata/plástico/vidrio), no marcas ni productos.
Para reconocer "Coca-Cola", "Fanta", "Sprite", etc. se usa **reconocimiento por embeddings
(few-shot / image retrieval)**, no un clasificador de clases fijas:

1. Se usa **MobileNetV2 preentrenado** (pesos ImageNet, sin la capa de clasificación) como
   **extractor de características**.
2. Al **registrar un producto** se toman N fotos (5–10). El backend convierte cada foto en un
   **vector ("huella")** L2-normalizado y lo guarda en la base de datos.
3. Al **escanear** se calcula la huella de la foto y se compara por **similitud coseno** contra
   todas las huellas guardadas.
4. Si la mejor coincidencia supera el **umbral de confianza** (0.75, configurable) → se reconoce
   el producto. Si no → "No reconocido → ¿Registrar nuevo?".

**Consecuencia:** agregar un producto nuevo NO requiere entrenar nada. Se fotografía y listo.
(Documentar que esto es el corazón del sistema: "entrenar = tomar fotos de referencia".)

> Fase 3 opcional: si dos productos se confunden (Coca vs Coca Zero), hacer *fine-tuning* del
> embedding con más fotos y pérdida de tripletes. No es parte del MVP.

---

## 3. Arquitectura

```
┌─ Web móvil (web/index.html) ────┐        ┌─ Backend FastAPI (backend/main.py) ──┐        ┌─ SQLite ─────────┐
│  · Cámara (getUserMedia)         │        │  · /api/productos      (CRUD)        │        │  datos/tienda.db │
│  · Captura de foto (JPEG)        │─HTTP──▶│  · /api/productos/{id}/fotos         │────────│  · productos     │
│  · Modales: venta / registro /   │        │  · /api/productos/{id}/stock         │        │  · fotos_ref (vec)│
│    stock / no reconocido         │◀──JSON─│  · /api/reconocer                    │        │  · ventas        │
│  · Interfaz móvil (Bs, stock)    │        │  · /api/ventas                       │        │  · movimientos   │
└──────────────────────────────────┘        │  · /api/resumen /api/config          │        └──────────────────┘
                                             └──────────────────────────────────────┘
                                                  │
                                                  └─ embeddings.py · MobileNetV2 (Keras)
                                                       fotos/ = imágenes de referencia
```

- **El CNN corre en el backend** (TensorFlow ya instalado). El navegador solo sube/recibe
  imágenes y JSON. Backend "pequeño" y fácil de mantener.
- El backend también sirve la web (un solo proceso, un solo puerto).
- Se publica con **ngrok** (mismo esquema que ahora). Conexión preferente por WiFi local; ngrok
  como respaldo/acceso externo.

---

## 4. Estructura de carpetas

```
reciclaje_ai/
├── planificacion.md            # este archivo
├── backend/
│   ├── main.py                 # API FastAPI + servidor de la web
│   ├── db.py                   # conexión, esquema SQLite e init
│   └── embeddings.py           # MobileNetV2 → huella L2 + match por coseno
├── web/
│   └── index.html              # POS móvil (pestañas Escanear / Productos / Historial)
├── datos/
│   ├── tienda.db               # base de datos SQLite (se crea sola)
│   └── fotos/                  # fotos de referencia, por producto: datos/fotos/<id>/1.jpg ...
├── requirements.txt            # (ya incluye fastapi, uvicorn, pillow, tensorflow, python-multipart)
└── versiones/                  # versión estable del reciclaje (archivada)
```

---

## 5. Base de datos (SQLite)

| Tabla | Columnas |
|---|---|
| `productos` | `id` PK, `nombre`, `precio` REAL, `categoria`, `umbral_stock` INT, `stock` INT, `activo` INT, `creado_en` TEXT |
| `fotos_referencia` | `id` PK, `producto_id` FK, `ruta` TEXT, `embedding` BLOB (float32 L2-normalizado), `creado_en` TEXT |
| `ventas` | `id` PK, `producto_id` FK, `cantidad` INT, `precio_unitario` REAL, `total` REAL, `fecha` TEXT |
| `movimientos` | `id` PK, `producto_id` FK, `tipo` TEXT (`entrada`/`salida`/`ajuste`), `cantidad` INT, `nota` TEXT, `fecha` TEXT |
| `config` | `clave` PK, `valor` TEXT (nombre de tienda, moneda=Bs, umbral de reconocimiento) |

- La huella del producto = **promedio normalizado** de las huellas de sus fotos de referencia.
- `stock` se actualiza en la tabla `productos` y se registra en `movimientos` (auditoría).

---

## 6. API (FastAPI)

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/productos?q=` | Lista productos con stock (búsqueda opcional) |
| POST | `/api/productos` | Crea producto: `{nombre, precio, categoria, umbral_stock}` |
| POST | `/api/productos/{id}/fotos` | Añade fotos de referencia (multipart) → calcula y guarda huellas |
| GET | `/api/productos/{id}/fotos` | Lista/entrega fotos de referencia |
| DELETE | `/api/productos/{id}` | Desactiva producto |
| POST | `/api/productos/{id}/stock` | Ajuste de stock: `{cantidad, nota}` (+/-) |
| POST | `/api/reconocer` | Sube 1 foto → `{producto|null, confianza, candidatos}` |
| POST | `/api/ventas` | Venta: `{producto_id, cantidad}` → descuenta stock y registra |
| GET | `/api/ventas?desde=&hasta=` | Historial de ventas |
| GET | `/api/resumen` | Total vendido hoy (Bs), nº ventas, productos con stock bajo |
| GET/PUT | `/api/config` | Nombre de tienda, moneda, umbral |

---

## 7. Flujos de usuario (web móvil)

### 7.1 Vender (Escanear)
1. Cámara activa (trasera). Botón **Capturar**.
2. `POST /api/reconocer` con la foto.
3. **Reconocido** → modal: nombre, precio, selector de cantidad, subtotal en Bs → **Aceptar venta**
   → `POST /api/ventas` → stock descontado, confirmación + sonido.
4. **No reconocido** → modal: "Producto no reconocido" + botón **Registrar nuevo** (pre-carga la
   foto tomada como primera referencia).

### 7.2 Registrar producto
1. Formulario: nombre, precio (Bs), categoría, umbral de stock.
2. Tomar **5–10 fotos** (botón repetir captura, contador visible, mínimo 5).
3. **Guardar** → `POST /api/productos` + `POST .../fotos` → queda reconocible al instante.

### 7.3 Agregar stock
1. Escanear (o buscar) producto → modal con stock actual.
2. Ingresar cantidad → `POST /api/productos/{id}/stock` → suma y registra movimiento.

### 7.4 Inventario / Historial
- **Productos**: lista con nombre, precio, stock; alerta visual si `stock <= umbral_stock`.
- **Historial**: ventas del día, total en Bs, y resumen.

---

## 8. Detalles de comportamiento

- **Moneda**: Bs. Formato `Bs 12.50`.
- **Stock 0**: permite vender pero avisa (puede quedar negativo; se corrige con "agregar stock").
- **Umbral de reconocimiento**: 0.75 por defecto, editable en ajustes.
- **Cámara tapada / poca luz**: evitar llamar al backend (reutilizar la detección de brillo actual).
- **ngrok**: mantener el header `ngrok-skip-browser-warning` en todos los `fetch`.
- **Sin login** (una sola tienda, uso local).

---

## 9. Fases

| Fase | Alcance |
|---|---|
| **Fase 1 (MVP)** | Backend + SQLite + embeddings MobileNetV2 + reconocimiento de **1 producto por escaneo** + venta con cantidad + registro de producto con fotos + stock + web móvil (Bs) |
| **Fase 2** | Detección de **varios productos en una misma foto** (rejilla + deduplicación), **carrito** de varios productos antes de cobrar, reportes de ventas |
| **Fase 3 (opcional)** | *Fine-tuning* del embedding para productos muy parecidos, multiusuario/auth, ticket imprimible, app offline (PWA) |

---

## 10. Entorno verificado

- Python 3.12.8 · TensorFlow 2.21 · FastAPI 0.128 · Uvicorn 0.52 · Pillow 12.1 · python-multipart 0.0.32 · tensorflowjs 4.22.
- **No requiere instalar dependencias nuevas.**
- CPU-only: reconocimiento ≈ 0.5–1 s por foto. Aceptable para punto de venta.

---

## 11. Limitaciones conocidas

- Embeddings ImageNet pueden confundir envases muy parecidos → mitigado con umbral de confianza,
  mensaje "no reconocido" y más fotos de referencia (Fase 3 para casos difíciles).
- Una foto por escaneo en el MVP (Fase 2 resuelve el multi-producto).
- Sin autenticación en v1 (uso en red local / tienda).

---

## 12. Próximos pasos de implementación

1. Crear `backend/db.py` (esquema + init de `datos/tienda.db`).
2. Crear `backend/embeddings.py` (MobileNetV2 → huella L2 + match coseno).
3. Crear `backend/main.py` (endpoints + servido de la web estática).
4. Crear `web/index.html` (POS móvil: Escanear / Productos / Historial + modales).
5. Probar E2E: registrar un producto, venderlo, verificar stock e historial.
6. Publicar con ngrok y probar desde el celular.
