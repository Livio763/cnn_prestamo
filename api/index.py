"""
FastAPI Serverless Application para Préstamo Inteligente.
Compatible con Vercel Serverless Functions y ejecución local con Uvicorn.
Sirve tanto las rutas de API (/api/...) como la interfaz web y el modelo CNN.
"""

import os
import datetime
from typing import Optional, List
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from api import db
except ImportError:
    import db

app = FastAPI(title="Préstamo Inteligente API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    db.init_db()


class CrearObjetoRequest(BaseModel):
    id: str
    nombre: str
    descripcion: Optional[str] = ""
    categoria: Optional[str] = "General"
    stock_total: Optional[int] = 1


class PrestamoRequest(BaseModel):
    objeto_id: str
    persona: Optional[str] = ""
    nota: Optional[str] = ""


class DevolucionRequest(BaseModel):
    objeto_id: str
    persona: Optional[str] = ""


class SincronizarClasesRequest(BaseModel):
    clases: List[str]


router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "motor": "turso" if db.es_turso() else "sqlite_local",
        "timestamp": datetime.datetime.now().isoformat(),
    }


@router.get("/objetos")
def listar_objetos():
    objetos = db.query("SELECT * FROM objetos ORDER BY nombre ASC")
    return objetos


@router.get("/objetos/{objeto_id}")
def obtener_objeto(objeto_id: str):
    filas = db.query("SELECT * FROM objetos WHERE id = ?", [objeto_id])
    if not filas:
        raise HTTPException(status_code=404, detail="Objeto no encontrado")
    return filas[0]


@router.post("/objetos")
def crear_objeto(req: CrearObjetoRequest):
    existente = db.query("SELECT id FROM objetos WHERE id = ?", [req.id])
    if existente:
        raise HTTPException(status_code=400, detail="Ya existe un objeto con ese ID.")

    ahora = datetime.datetime.now().isoformat()
    stock = max(1, req.stock_total or 1)
    db.query(
        """
        INSERT INTO objetos (id, nombre, descripcion, categoria, stock_total, stock_disponible, activo, creado_en)
        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
        """,
        [req.id, req.nombre, req.descripcion, req.categoria, stock, stock, ahora],
        commit=True,
    )
    return {
        "id": req.id,
        "nombre": req.nombre,
        "descripcion": req.descripcion,
        "categoria": req.categoria,
        "stock_total": stock,
        "stock_disponible": stock,
        "activo": 1,
        "creado_en": ahora,
    }


@router.post("/prestamos")
def registrar_prestamo(req: PrestamoRequest):
    filas = db.query("SELECT * FROM objetos WHERE id = ?", [req.objeto_id])
    if not filas:
        raise HTTPException(status_code=404, detail="El objeto no existe.")

    objeto = filas[0]
    if objeto["stock_disponible"] <= 0:
        raise HTTPException(status_code=400, detail="No hay unidades disponibles para prestar.")

    ahora = datetime.datetime.now().isoformat()
    persona = (req.persona or "").strip() or "Anónimo"

    db.query(
        "UPDATE objetos SET stock_disponible = stock_disponible - 1 WHERE id = ?",
        [req.objeto_id],
        commit=True,
    )

    db.query(
        """
        INSERT INTO prestamos (objeto_id, persona, fecha_prestamo, abierto)
        VALUES (?, ?, ?, 1)
        """,
        [req.objeto_id, persona, ahora],
        commit=True,
    )

    db.query(
        """
        INSERT INTO movimientos (objeto_id, tipo, cantidad, persona, fecha, nota)
        VALUES (?, 'prestamo', 1, ?, ?, ?)
        """,
        [req.objeto_id, persona, ahora, req.nota or ""],
        commit=True,
    )

    return {"status": "ok", "mensaje": f"Préstamo registrado a {persona}"}


@router.post("/devoluciones")
def registrar_devolucion(req: DevolucionRequest):
    filas = db.query("SELECT * FROM objetos WHERE id = ?", [req.objeto_id])
    if not filas:
        raise HTTPException(status_code=404, detail="El objeto no existe.")

    objeto = filas[0]
    if objeto["stock_disponible"] >= objeto["stock_total"]:
        raise HTTPException(status_code=400, detail="No hay unidades prestadas de este objeto.")

    ahora = datetime.datetime.now().isoformat()
    nombre_persona = (req.persona or "").strip()

    abiertos = db.query(
        "SELECT * FROM prestamos WHERE objeto_id = ? AND abierto = 1 ORDER BY fecha_prestamo ASC",
        [req.objeto_id],
    )
    prestamo_a_cerrar = None
    if nombre_persona:
        for p in abiertos:
            if (p["persona"] or "").strip().lower() == nombre_persona.lower():
                prestamo_a_cerrar = p
                break
    if not prestamo_a_cerrar and abiertos:
        prestamo_a_cerrar = abiertos[0]

    db.query(
        "UPDATE objetos SET stock_disponible = MIN(stock_total, stock_disponible + 1) WHERE id = ?",
        [req.objeto_id],
        commit=True,
    )

    persona_final = nombre_persona or "Anónimo"
    if prestamo_a_cerrar:
        persona_final = prestamo_a_cerrar["persona"]
        db.query(
            "UPDATE prestamos SET abierto = 0, fecha_devolucion = ? WHERE id = ?",
            [ahora, prestamo_a_cerrar["id"]],
            commit=True,
        )

    nota = "" if prestamo_a_cerrar else "Devolución sin préstamo abierto registrado"
    db.query(
        """
        INSERT INTO movimientos (objeto_id, tipo, cantidad, persona, fecha, nota)
        VALUES (?, 'devolucion', 1, ?, ?, ?)
        """,
        [req.objeto_id, persona_final, ahora, nota],
        commit=True,
    )

    return {"status": "ok", "mensaje": f"Devolución registrada de {persona_final}"}


@router.get("/movimientos")
def listar_movimientos():
    movimientos = db.query("SELECT * FROM movimientos ORDER BY fecha DESC LIMIT 100")
    return movimientos


@router.post("/sincronizar-clases")
def sincronizar_clases(req: SincronizarClasesRequest):
    existentes = db.query("SELECT id FROM objetos")
    ids = set(o["id"] for o in existentes)
    ahora = datetime.datetime.now().isoformat()
    nuevos = []

    for clase in req.clases:
        if not clase or clase.startswith("_"):
            continue
        if clase in ids:
            continue
        nombre = clase.replace("_", " ").replace("-", " ").title()
        db.query(
            """
            INSERT INTO objetos (id, nombre, descripcion, categoria, stock_total, stock_disponible, activo, creado_en)
            VALUES (?, ?, 'Creado automáticamente desde el modelo', 'General', 1, 1, 1, ?)
            """,
            [clase, nombre, ahora],
            commit=True,
        )
        nuevos.append(clase)
        ids.add(clase)

    return {"sincronizados": nuevos}


# Montar las rutas API tanto en /api/... como directamente
app.include_router(router, prefix="/api")
app.include_router(router)

# Montar los archivos estáticos en la raíz (index.html, css, js, modelo_web)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(ROOT_DIR, "index.html")):
    app.mount("/", StaticFiles(directory=ROOT_DIR, html=True), name="static")
