/*
 * Capa de datos del préstamo.
 *
 * Arquitectura híbrida:
 * 1. Primaria: API REST del backend (/api) conectada a Turso / SQLite Serverless.
 *    Permite sincronización instantánea entre celular, PC y cualquier navegador.
 * 2. Respaldo (Fallback): IndexedDB local si el backend no está disponible (modo offline).
 */

const DB_NAME = "prestamo_inteligente";
const DB_VERSION = 1;

let usandoBackend = null;
let motorBD = "indexeddb";

const OBJETOS_POR_DEFECTO = [
  {
    id: "mouse",
    nombre: "Mouse",
    descripcion: "Mouse USB de escritorio",
    categoria: "Accesorios de computadora",
    stock_total: 1,
    stock_disponible: 1
  },
  {
    id: "calculadora",
    nombre: "Calculadora",
    descripcion: "Calculadora científica",
    categoria: "Útiles de estudio",
    stock_total: 1,
    stock_disponible: 1
  },
  {
    id: "cargador",
    nombre: "Cargador",
    descripcion: "Cargador de celular",
    categoria: "Accesorios de computadora",
    stock_total: 2,
    stock_disponible: 2
  },
  {
    id: "tester",
    nombre: "Tester",
    descripcion: "Multímetro o tester de mano",
    categoria: "Herramientas",
    stock_total: 1,
    stock_disponible: 1
  },
  {
    id: "usb",
    nombre: "USB",
    descripcion: "Memoria USB pequeña",
    categoria: "Accesorios de computadora",
    stock_total: 1,
    stock_disponible: 1
  }
];

/* ------------------------------------------------------------------ */
/* Conexión y Detección de Backend                                     */
/* ------------------------------------------------------------------ */

async function comprobarBackend() {
  if (usandoBackend !== null) return usandoBackend;
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    if (res.ok) {
      const data = await res.json();
      usandoBackend = true;
      motorBD = data.motor || "backend";
      return true;
    }
  } catch (_) {
    // Backend no disponible, usar fallback
  }
  usandoBackend = false;
  motorBD = "indexeddb";
  return false;
}

let dbPromise = null;

function abrirIndexedDB() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const solicitud = indexedDB.open(DB_NAME, DB_VERSION);

    solicitud.onupgradeneeded = (evento) => {
      const db = evento.target.result;

      if (!db.objectStoreNames.contains("objetos")) {
        const store = db.createObjectStore("objetos", { keyPath: "id" });
        store.createIndex("nombre", "nombre", { unique: false });
      }

      if (!db.objectStoreNames.contains("prestamos")) {
        const store = db.createObjectStore("prestamos", {
          keyPath: "id",
          autoIncrement: true
        });
        store.createIndex("objeto_id", "objeto_id", { unique: false });
        store.createIndex("abierto", "abierto", { unique: false });
      }

      if (!db.objectStoreNames.contains("movimientos")) {
        const store = db.createObjectStore("movimientos", {
          keyPath: "id",
          autoIncrement: true
        });
        store.createIndex("fecha", "fecha", { unique: false });
      }
    };

    solicitud.onsuccess = () => resolve(solicitud.result);
    solicitud.onerror = () => reject(solicitud.error || new Error("No se pudo abrir IndexedDB"));
  });

  return dbPromise;
}

async function abrirDB() {
  const hayApi = await comprobarBackend();
  if (hayApi) return true;
  await abrirIndexedDB();
  return true;
}

function pedir(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Error de IndexedDB"));
  });
}

/* ------------------------------------------------------------------ */
/* Operaciones: Catálogo y Stock                                       */
/* ------------------------------------------------------------------ */

async function listarObjetos() {
  if (await comprobarBackend()) {
    const res = await fetch("/api/objetos", { cache: "no-store" });
    if (!res.ok) throw new Error("Error al consultar objetos en el servidor.");
    const lista = await res.json();
    return lista.sort((a, b) => (a.nombre || "").localeCompare(b.nombre || "", "es"));
  }

  const db = await abrirIndexedDB();
  const tx = db.transaction("objetos", "readonly");
  const objetos = await pedir(tx.objectStore("objetos").getAll());
  return objetos.sort((a, b) => (a.nombre || "").localeCompare(b.nombre || "", "es"));
}

async function obtenerObjeto(id) {
  if (await comprobarBackend()) {
    const res = await fetch(`/api/objetos/${encodeURIComponent(id)}`, { cache: "no-store" });
    if (res.status === 404) return null;
    if (!res.ok) throw new Error("Error al consultar objeto.");
    return res.json();
  }

  const db = await abrirIndexedDB();
  const tx = db.transaction("objetos", "readonly");
  return pedir(tx.objectStore("objetos").get(id));
}

async function crearObjeto(datos) {
  if (await comprobarBackend()) {
    const res = await fetch("/api/objetos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(datos)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "No se pudo crear el objeto en el servidor.");
    }
    return res.json();
  }

  const db = await abrirIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction("objetos", "readwrite");
    const store = tx.objectStore("objetos");
    const consulta = store.get(datos.id);

    let terminado = false;
    const fallar = (mensaje) => {
      if (terminado) return;
      terminado = true;
      try { tx.abort(); } catch (_) { /* ignorar */ }
      reject(new Error(mensaje));
    };

    consulta.onsuccess = () => {
      if (consulta.result) {
        fallar("Ya existe un objeto con ese ID de clase.");
        return;
      }

      const objeto = {
        id: datos.id,
        nombre: datos.nombre,
        descripcion: datos.descripcion || "",
        categoria: datos.categoria || "General",
        stock_total: Number(datos.stock_total) || 1,
        stock_disponible: Number(datos.stock_total) || 1,
        activo: true,
        creado_en: new Date().toISOString()
      };

      store.add(objeto);
      tx.oncomplete = () => {
        if (terminado) return;
        terminado = true;
        resolve(objeto);
      };
    };

    consulta.onerror = () => fallar("No se pudo comprobar el ID del objeto.");
  });
}

/* ------------------------------------------------------------------ */
/* Operaciones: Préstamos y Devoluciones                              */
/* ------------------------------------------------------------------ */

async function registrarPrestamo(objetoId, persona, nota = "") {
  if (await comprobarBackend()) {
    const res = await fetch("/api/prestamos", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ objeto_id: objetoId, persona, nota })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Error al registrar el préstamo en el servidor.");
    }
    return res.json();
  }

  const db = await abrirIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["objetos", "prestamos", "movimientos"], "readwrite");
    const objetos = tx.objectStore("objetos");
    const prestamos = tx.objectStore("prestamos");
    const movimientos = tx.objectStore("movimientos");
    const consulta = objetos.get(objetoId);

    let terminado = false;
    const fallar = (mensaje) => {
      if (terminado) return;
      terminado = true;
      try { tx.abort(); } catch (_) { /* ignorar */ }
      reject(new Error(mensaje));
    };

    consulta.onsuccess = () => {
      const objeto = consulta.result;
      if (!objeto) {
        fallar("El objeto no existe en el catálogo.");
        return;
      }
      if (objeto.stock_disponible <= 0) {
        fallar("No hay unidades disponibles de este objeto.");
        return;
      }

      objeto.stock_disponible -= 1;
      objetos.put(objeto);

      const ahora = new Date().toISOString();
      const personaFinal = (persona || "").trim() || "Anónimo";

      prestamos.add({
        objeto_id: objetoId,
        persona: personaFinal,
        fecha_prestamo: ahora,
        fecha_devolucion: null,
        abierto: 1
      });

      movimientos.add({
        objeto_id: objetoId,
        tipo: "prestamo",
        cantidad: 1,
        persona: personaFinal,
        fecha: ahora,
        nota: nota || ""
      });
    };

    consulta.onerror = () => fallar("No se pudo leer el objeto.");
    tx.oncomplete = () => {
      if (terminado) return;
      terminado = true;
      resolve();
    };
    tx.onerror = () => fallar(tx.error?.message || "Error al registrar el préstamo.");
  });
}

async function registrarDevolucion(objetoId, persona) {
  if (await comprobarBackend()) {
    const res = await fetch("/api/devoluciones", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ objeto_id: objetoId, persona })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Error al registrar la devolución en el servidor.");
    }
    return res.json();
  }

  const db = await abrirIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(["objetos", "prestamos", "movimientos"], "readwrite");
    const objetos = tx.objectStore("objetos");
    const prestamos = tx.objectStore("prestamos");
    const movimientos = tx.objectStore("movimientos");
    const consulta = objetos.get(objetoId);

    let terminado = false;
    const fallar = (mensaje) => {
      if (terminado) return;
      terminado = true;
      try { tx.abort(); } catch (_) { /* ignorar */ }
      reject(new Error(mensaje));
    };

    consulta.onsuccess = () => {
      const objeto = consulta.result;
      if (!objeto) {
        fallar("El objeto no existe.");
        return;
      }
      if (objeto.stock_disponible >= objeto.stock_total) {
        fallar("No hay unidades prestadas de este objeto.");
        return;
      }

      const indice = prestamos.index("abierto");
      const abiertosReq = indice.getAll(1);

      abiertosReq.onsuccess = () => {
        const abiertos = (abiertosReq.result || [])
          .filter((p) => p.objeto_id === objetoId)
          .sort((a, b) => new Date(a.fecha_prestamo) - new Date(b.fecha_prestamo));

        let prestamo = null;
        const nombre = (persona || "").trim();

        if (nombre) {
          prestamo = abiertos.find(
            (p) => (p.persona || "").toLowerCase() === nombre.toLowerCase()
          );
        }
        if (!prestamo) prestamo = abiertos[0] || null;

        const ahora = new Date().toISOString();

        objeto.stock_disponible += 1;
        if (objeto.stock_disponible > objeto.stock_total) {
          objeto.stock_disponible = objeto.stock_total;
        }
        objetos.put(objeto);

        if (prestamo) {
          prestamo.fecha_devolucion = ahora;
          prestamo.abierto = 0;
          prestamos.put(prestamo);
        }

        movimientos.add({
          objeto_id: objetoId,
          tipo: "devolucion",
          cantidad: 1,
          persona: prestamo?.persona || nombre || "Anónimo",
          fecha: ahora,
          nota: prestamo ? "" : "Devolución sin préstamo abierto"
        });
      };

      abiertosReq.onerror = () => fallar("No se pudieron leer los préstamos abiertos.");
    };

    consulta.onerror = () => fallar("No se pudo leer el objeto.");
    tx.oncomplete = () => {
      if (terminado) return;
      terminado = true;
      resolve();
    };
    tx.onerror = () => fallar(tx.error?.message || "Error al registrar la devolución.");
  });
}

async function listarMovimientos() {
  if (await comprobarBackend()) {
    const res = await fetch("/api/movimientos", { cache: "no-store" });
    if (!res.ok) throw new Error("Error al consultar movimientos en el servidor.");
    return res.json();
  }

  const db = await abrirIndexedDB();
  const tx = db.transaction("movimientos", "readonly");
  const movimientos = await pedir(tx.objectStore("movimientos").getAll());
  return movimientos.sort((a, b) => new Date(b.fecha) - new Date(a.fecha));
}

/* ------------------------------------------------------------------ */
/* Inicialización y Sincronización                                    */
/* ------------------------------------------------------------------ */

async function sembrarObjetos() {
  if (await comprobarBackend()) {
    // El backend ya siembra los objetos iniciales en startup si la tabla está vacía
    return;
  }

  const objetos = await listarObjetos();
  if (objetos.length > 0) return;

  const db = await abrirIndexedDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction("objetos", "readwrite");
    const store = tx.objectStore("objetos");

    OBJETOS_POR_DEFECTO.forEach((objeto) => {
      store.add({
        ...objeto,
        activo: true,
        creado_en: new Date().toISOString()
      });
    });

    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error("No se pudieron crear los objetos iniciales."));
  });
}

function humanizarId(id) {
  return id
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letra) => letra.toUpperCase());
}

async function sincronizarClases(clases) {
  if (await comprobarBackend()) {
    const res = await fetch("/api/sincronizar-clases", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clases })
    });
    if (res.ok) {
      const data = await res.json();
      return data.sincronizados || [];
    }
  }

  const existentes = await listarObjetos();
  const ids = new Set(existentes.map((o) => o.id));
  const nuevos = [];

  (clases || []).forEach((clase) => {
    if (!clase || clase.startsWith("_")) return;
    if (ids.has(clase)) return;

    nuevos.push({
      id: clase,
      nombre: humanizarId(clase),
      descripcion: "Creado automáticamente desde el dataset",
      categoria: "General",
      stock_total: 1,
      stock_disponible: 1,
      activo: true,
      creado_en: new Date().toISOString()
    });
  });

  if (nuevos.length === 0) return [];

  const db = await abrirIndexedDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction("objetos", "readwrite");
    const store = tx.objectStore("objetos");
    nuevos.forEach((objeto) => store.add(objeto));
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error || new Error("No se pudieron sincronizar las clases."));
  });

  return nuevos;
}

window.DB = {
  abrirDB,
  listarObjetos,
  obtenerObjeto,
  crearObjeto,
  registrarPrestamo,
  registrarDevolucion,
  listarMovimientos,
  sembrarObjetos,
  sincronizarClases,
  getMotor: () => motorBD
};
