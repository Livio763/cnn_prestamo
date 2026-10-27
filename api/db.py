"""
Capa de base de datos para Préstamo Inteligente.
Compatible con:
1. SQLite local (prestamo.db) para desarrollo local.
2. Turso (libSQL Serverless) vía HTTP API cuando TURSO_DATABASE_URL y TURSO_AUTH_TOKEN están configurados.
"""

import os
import sqlite3
import datetime
from typing import Any, Dict, List, Optional
import httpx

TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "").strip()
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "").strip()
LOCAL_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prestamo.db")

OBJETOS_DEFECTO = [
    {
        "id": "mouse",
        "nombre": "Mouse",
        "descripcion": "Mouse USB de escritorio",
        "categoria": "Accesorios de computadora",
        "stock_total": 1,
        "stock_disponible": 1,
    },
    {
        "id": "calculadora",
        "nombre": "Calculadora",
        "descripcion": "Calculadora científica",
        "categoria": "Útiles de estudio",
        "stock_total": 1,
        "stock_disponible": 1,
    },
    {
        "id": "cargador",
        "nombre": "Cargador",
        "descripcion": "Cargador de celular",
        "categoria": "Accesorios de computadora",
        "stock_total": 2,
        "stock_disponible": 2,
    },
    {
        "id": "tester",
        "nombre": "Tester",
        "descripcion": "Multímetro o tester de mano",
        "categoria": "Herramientas",
        "stock_total": 1,
        "stock_disponible": 1,
    },
    {
        "id": "usb",
        "nombre": "USB",
        "descripcion": "Memoria USB pequeña",
        "categoria": "Accesorios de computadora",
        "stock_total": 1,
        "stock_disponible": 1,
    },
]


def es_turso() -> bool:
    return bool(TURSO_URL and TURSO_TOKEN)


def _turso_http_url() -> str:
    url = TURSO_URL
    if url.startswith("libsql://"):
        url = "https://" + url[len("libsql://") :]
    if not url.startswith("http"):
        url = "https://" + url
    if not url.endswith("/v2/pipeline"):
        url = url.rstrip("/") + "/v2/pipeline"
    return url


def ejecutar_turso(sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    endpoint = _turso_http_url()
    headers = {
        "Authorization": f"Bearer {TURSO_TOKEN}",
        "Content-Type": "application/json",
    }

    args = []
    for p in params or []:
        if p is None:
            args.append({"type": "null"})
        elif isinstance(p, int):
            args.append({"type": "integer", "value": str(p)})
        elif isinstance(p, float):
            args.append({"type": "float", "value": p})
        else:
            args.append({"type": "text", "value": str(p)})

    payload = {
        "requests": [
            {
                "type": "execute",
                "stmt": {
                    "sql": sql,
                    "args": args,
                },
            },
            {"type": "close"},
        ]
    }

    with httpx.Client(timeout=10.0) as client:
        resp = client.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results or results[0].get("type") != "ok":
        error_msg = results[0].get("error", {}).get("message", "Error de Turso") if results else "Error"
        raise RuntimeError(f"Error en consulta Turso: {error_msg}")

    response = results[0].get("response", {}).get("result", {})
    cols = [c["name"] for c in response.get("cols", [])]
    filas = []
    for row in response.get("rows", []):
        d = {}
        for c, v in zip(cols, row):
            val = v.get("value")
            if v.get("type") == "integer" and val is not None:
                val = int(val)
            elif v.get("type") == "float" and val is not None:
                val = float(val)
            d[c] = val
        filas.append(d)
    return filas


def ejecutar_sqlite(sql: str, params: Optional[List[Any]] = None, commit: bool = False) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(LOCAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        filas = [dict(row) for row in cur.fetchall()]
        if commit:
            conn.commit()
        return filas
    finally:
        conn.close()


def query(sql: str, params: Optional[List[Any]] = None, commit: bool = False) -> List[Dict[str, Any]]:
    if es_turso():
        return ejecutar_turso(sql, params)
    return ejecutar_sqlite(sql, params, commit=commit)


def init_db():
    schema = """
    CREATE TABLE IF NOT EXISTS objetos (
        id TEXT PRIMARY KEY,
        nombre TEXT NOT NULL,
        descripcion TEXT,
        categoria TEXT,
        stock_total INTEGER NOT NULL DEFAULT 1,
        stock_disponible INTEGER NOT NULL DEFAULT 1,
        activo INTEGER NOT NULL DEFAULT 1,
        creado_en TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS prestamos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        objeto_id TEXT NOT NULL,
        persona TEXT NOT NULL,
        fecha_prestamo TEXT NOT NULL,
        fecha_devolucion TEXT,
        abierto INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (objeto_id) REFERENCES objetos(id)
    );

    CREATE TABLE IF NOT EXISTS movimientos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        objeto_id TEXT NOT NULL,
        tipo TEXT NOT NULL,
        cantidad INTEGER NOT NULL DEFAULT 1,
        persona TEXT NOT NULL,
        fecha TEXT NOT NULL,
        nota TEXT
    );
    """
    if es_turso():
        for stmt in schema.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                query(stmt)
    else:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        try:
            conn.executescript(schema)
            conn.commit()
        finally:
            conn.close()

    # Sembrar objetos por defecto si la tabla está vacía
    filas = query("SELECT COUNT(*) AS total FROM objetos")
    total = filas[0]["total"] if filas else 0
    if total == 0:
        ahora = datetime.datetime.now().isoformat()
        for obj in OBJETOS_DEFECTO:
            query(
                """
                INSERT OR IGNORE INTO objetos (id, nombre, descripcion, categoria, stock_total, stock_disponible, activo, creado_en)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                [
                    obj["id"],
                    obj["nombre"],
                    obj["descripcion"],
                    obj["categoria"],
                    obj["stock_total"],
                    obj["stock_disponible"],
                    ahora,
                ],
                commit=True,
            )
