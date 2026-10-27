/*
 * Lógica de la interfaz de préstamo.
 * - Cámara + TensorFlow.js para reconocer objetos.
 * - IndexedDB para catálogo, stock, préstamos e historial.
 * - El modelo se entrena aparte con entrenar.py.
 */

const UMBRAL_CONFIANZA = 0.40;
const MARGEN_MINIMO = 0.05;
const TAMANO_DEFECTO = 160;
const FRACCION_ROI = 0.70;
const RUTA_MODELO = "modelo_web";

const estado = {
  modelo: null,
  clases: [],
  tamano: TAMANO_DEFECTO,
  objetos: [],
  movimientos: [],
  metricas: null,
  objetoActual: null,
  accionPendiente: null,
  camaraActiva: false,
  facing: "environment",
  ultimaProb: 0,
  vistaResultado: false,
  errorModelo: "",
  rutaAlta: "",
  intervaloEscaneo: null,
  escaneoPausado: false,
  analizandoFrame: false,
  rachaActual: 0,
  claseCandidata: null,
  MIN_RACHA: 2,
  CADENCIA_MS: 220
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

document.addEventListener("DOMContentLoaded", iniciar);

async function iniciar() {
  configurarTabs();
  configurarEventos();

  const dbLista = await iniciarDB();
  if (!dbLista) {
    actualizarPillModelo("danger", "Modelo: sin DB");
    return;
  }

  await cargarModelo();
  await recargarDatos();
  renderTodo();
}

/* ------------------------------------------------------------------ */
/* Arranque y estado                                                   */
/* ------------------------------------------------------------------ */

async function iniciarDB() {
  try {
    await DB.abrirDB();
    await DB.sembrarObjetos();
    actualizarPillDB("ok", "DB: lista");
    return true;
  } catch (error) {
    actualizarPillDB("danger", "DB: error");
    toast(`No se pudo abrir la base de datos: ${error.message}`, true);
    return false;
  }
}

function actualizarPillDB(tipo, texto) {
  const el = $("#pill-db");
  el.className = `pill ${tipo === "ok" ? "pill-ok" : tipo === "danger" ? "pill-danger" : "pill-muted"}`;
  el.textContent = texto;
}

function actualizarPillModelo(tipo, texto) {
  const el = $("#pill-modelo");
  el.className = `pill ${
    tipo === "ok" ? "pill-ok" : tipo === "warn" ? "pill-warn" : tipo === "danger" ? "pill-danger" : "pill-muted"
  }`;
  el.textContent = texto;
}

async function recargarDatos() {
  estado.objetos = await DB.listarObjetos();
  estado.movimientos = await DB.listarMovimientos();
}

function renderTodo() {
  renderInventario();
  renderHistorial();
  renderMetricas();
  renderModeloDetalle();
  actualizarResultadoActual();
}

/* ------------------------------------------------------------------ */
/* Pestañas                                                            */
/* ------------------------------------------------------------------ */

function configurarTabs() {
  $$(".tab").forEach((boton) => {
    boton.addEventListener("click", () => activarTab(boton.dataset.tab));
  });
}

function activarTab(nombre) {
  $$(".tab").forEach((boton) => {
    const activo = boton.dataset.tab === nombre;
    boton.classList.toggle("is-active", activo);
    if (activo) boton.setAttribute("aria-current", "page");
    else boton.removeAttribute("aria-current");
  });

  $$(".tab-panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === `tab-${nombre}`);
  });

  if (nombre !== "escanear" && estado.camaraActiva) {
    detenerCamara();
  }

  if (nombre === "historial") renderHistorial();

  window.scrollTo({ top: 0, behavior: comportamientoScroll() });
}

function comportamientoScroll() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

/* ------------------------------------------------------------------ */
/* Cámara y modelo                                                     */
/* ------------------------------------------------------------------ */

async function cargarModelo() {
  actualizarPillModelo("muted", "Modelo: cargando…");

  if (estado.modelo) {
    estado.modelo.dispose();
    estado.modelo = null;
  }

  estado.clases = [];
  estado.metricas = null;
  limpiarResultado();
  await cargarMetricas();

  try {
    if (typeof tf === "undefined") {
      throw new Error("TensorFlow.js no se cargó. Revisa la conexión a internet.");
    }

    const NGROK_HEADERS = {
      "ngrok-skip-browser-warning": "true",
      "bypass-tunnel-reminder": "true"
    };
    const sello = Date.now();
    const respuestaClases = await fetch(`${RUTA_MODELO}/clases.json?t=${sello}`, {
      cache: "no-store",
      headers: NGROK_HEADERS
    });

    if (!respuestaClases.ok) {
      throw new Error("No existe modelo_web/clases.json. Entrena el modelo con python entrenar.py.");
    }

    const meta = await respuestaClases.json();
    if (!Array.isArray(meta.clases) || meta.clases.length === 0) {
      throw new Error("clases.json no contiene clases válidas.");
    }

    estado.clases = meta.clases;
    estado.tamano = Number(meta.tamano_img) || TAMANO_DEFECTO;
    ajustarCanvas();

    const modelo = await tf.loadLayersModel(
      tf.io.browserHTTPRequest(`${RUTA_MODELO}/model.json?t=${sello}`, {
        requestInit: {
          cache: "no-store",
          headers: NGROK_HEADERS
        }
      })
    );

    estado.modelo = modelo;
    estado.errorModelo = "";
    actualizarPillModelo("ok", `Modelo: ${estado.clases.length} clases`);
    await DB.sincronizarClases(estado.clases);
    await recargarDatos();
    renderTodo();
    habilitarCaptura(Boolean(estado.modelo));
  } catch (error) {
    if (estado.modelo) {
      estado.modelo.dispose();
      estado.modelo = null;
    }
    estado.clases = [];
    estado.errorModelo = error.message;
    actualizarPillModelo("warn", "Modelo: sin entrenar");
    renderModeloDetalle();
    habilitarCaptura(false);
  } finally {
    renderMetricas();
  }
}

async function cargarMetricas() {
  try {
    const respuesta = await fetch(`${RUTA_MODELO}/metricas.json?t=${Date.now()}`, {
      cache: "no-store",
      headers: {
        "ngrok-skip-browser-warning": "true",
        "bypass-tunnel-reminder": "true"
      }
    });
    estado.metricas = respuesta.ok ? await respuesta.json() : null;
  } catch (_) {
    estado.metricas = null;
  }
}

function ajustarCanvas() {
  const canvas = $("#canvas-captura");
  canvas.width = estado.tamano;
  canvas.height = estado.tamano;
}

function habilitarCaptura(habilitado) {
  $("#btn-capturar").disabled = !(habilitado && estado.camaraActiva);
}

async function iniciarCamara() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    $("#camara-estado").textContent = "Este navegador no soporta cámara o no estás en localhost/HTTPS.";
    return;
  }

  try {
    detenerCamara(false);

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: estado.facing },
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    const video = $("#video");
    video.srcObject = stream;
    await video.play();

    estado.camaraActiva = true;
    estado.escaneoPausado = false;
    estado.rachaActual = 0;
    estado.claseCandidata = null;

    $("#camara-placeholder").hidden = true;
    $("#btn-camara").textContent = "Detener cámara";
    $("#camara-estado").textContent = "Cámara activa · Auto-detección activada. Enfoca un objeto.";

    const hud = $("#scanner-hud");
    if (hud) {
      hud.hidden = false;
      hud.classList.remove("is-detected");
    }
    const hudTexto = $("#scanner-hud-text");
    if (hudTexto) hudTexto.textContent = "Buscando objeto…";

    const laser = $("#scanner-laser");
    if (laser) laser.hidden = false;

    const guide = $("#guide");
    if (guide) {
      guide.classList.add("is-scanning");
      guide.classList.remove("is-detected");
    }

    $("#btn-reanudar").hidden = true;
    habilitarCaptura(Boolean(estado.modelo));

    iniciarBucleEscaneo();
  } catch (error) {
    detenerCamara(false);
    estado.camaraActiva = false;
    $("#btn-camara").textContent = "Activar cámara";
    $("#camara-placeholder").hidden = false;
    $("#camara-estado").textContent = `No se pudo acceder a la cámara: ${error.message}`;
    habilitarCaptura(false);
  }
}

function detenerCamara(actualizarVista = true) {
  detenerBucleEscaneo();

  const video = $("#video");
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }

  estado.camaraActiva = false;
  estado.escaneoPausado = false;
  estado.rachaActual = 0;
  estado.claseCandidata = null;
  habilitarCaptura(false);

  const hud = $("#scanner-hud");
  if (hud) hud.hidden = true;

  const laser = $("#scanner-laser");
  if (laser) laser.hidden = true;

  const guide = $("#guide");
  if (guide) guide.classList.remove("is-scanning", "is-detected");

  $("#btn-reanudar").hidden = true;

  if (actualizarVista) {
    $("#camara-placeholder").hidden = false;
    $("#btn-camara").textContent = "Activar cámara";
    $("#camara-estado").textContent = "La cámara requiere localhost o HTTPS.";
  }
}

function iniciarBucleEscaneo() {
  detenerBucleEscaneo();
  estado.intervaloEscaneo = setInterval(tickEscaneoEnVivo, estado.CADENCIA_MS);
}

function detenerBucleEscaneo() {
  if (estado.intervaloEscaneo) {
    clearInterval(estado.intervaloEscaneo);
    estado.intervaloEscaneo = null;
  }
}

async function tickEscaneoEnVivo() {
  if (!estado.camaraActiva || !estado.modelo || estado.escaneoPausado || estado.analizandoFrame) {
    return;
  }

  const video = $("#video");
  if (!video || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return;
  }

  estado.analizandoFrame = true;

  const canvas = $("#canvas-captura");
  const ctx = canvas.getContext("2d");
  const lado = Math.min(video.videoWidth, video.videoHeight) * FRACCION_ROI;
  const sx = (video.videoWidth - lado) / 2;
  const sy = (video.videoHeight - lado) / 2;

  canvas.width = estado.tamano;
  canvas.height = estado.tamano;
  ctx.drawImage(video, sx, sy, lado, lado, 0, 0, estado.tamano, estado.tamano);

  let tensor = null;
  let salida = null;

  try {
    tensor = tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().expandDims(0));
    salida = estado.modelo.predict(tensor);
    const predicciones = await salida.data();

    const lista = estado.clases
      .map((clase, indice) => ({ clase, prob: Number(predicciones[indice]) || 0 }))
      .sort((a, b) => b.prob - a.prob);

    const top3 = lista.slice(0, 3);
    const top1 = top3[0];

    if (!top1) return;

    // Actualizar barras de probabilidades en tiempo real
    renderTop3(top3);

    const margen = top1.prob - (top3[1]?.prob || 0);
    const esDesconocido =
      top1.clase.startsWith("_") || top1.prob < UMBRAL_CONFIANZA || margen < MARGEN_MINIMO;

    const hud = $("#scanner-hud");
    const hudTexto = $("#scanner-hud-text");
    const guide = $("#guide");

    if (esDesconocido) {
      estado.rachaActual = 0;
      estado.claseCandidata = null;

      if (hud) hud.classList.remove("is-detected");
      if (guide) {
        guide.classList.remove("is-detected");
        guide.classList.add("is-scanning");
      }

      if (hudTexto) {
        if (top1.clase.startsWith("_") && top1.prob >= 0.70) {
          hudTexto.textContent = "Fondo / espacio libre";
        } else {
          hudTexto.textContent = "Buscando objeto…";
        }
      }
    } else {
      const objeto = estado.objetos.find((item) => item.id === top1.clase);
      const nombreVisible = objeto ? objeto.nombre : top1.clase;
      const pct = Math.round(top1.prob * 100);

      if (estado.claseCandidata === top1.clase) {
        estado.rachaActual++;
      } else {
        estado.claseCandidata = top1.clase;
        estado.rachaActual = 1;
      }

      if (hudTexto) {
        hudTexto.textContent = `Detectando ${nombreVisible} (${pct}%)…`;
      }

      // Si se confirma la racha de frames consecutivos
      if (estado.rachaActual >= estado.MIN_RACHA) {
        pausarYConfirmarDeteccion(top1, objeto, top3);
      }
    }
  } catch (_) {
    // Si falla un fotograma aislado no interrumpimos la experiencia
  } finally {
    if (salida) salida.dispose();
    if (tensor) tensor.dispose();
    estado.analizandoFrame = false;
  }
}

function pausarYConfirmarDeteccion(top1, objeto, top3) {
  estado.escaneoPausado = true;

  const video = $("#video");
  if (video && !video.paused) {
    try {
      video.pause();
    } catch (_) {}
  }

  const guide = $("#guide");
  if (guide) {
    guide.classList.remove("is-scanning");
    guide.classList.add("is-detected");
  }

  const laser = $("#scanner-laser");
  if (laser) laser.hidden = true;

  const hud = $("#scanner-hud");
  const hudTexto = $("#scanner-hud-text");
  if (hud) {
    hud.hidden = false;
    hud.classList.add("is-detected");
  }

  const pct = Math.round(top1.prob * 100);
  const nombreVisible = objeto ? objeto.nombre : top1.clase;
  if (hudTexto) {
    hudTexto.textContent = `✓ ${nombreVisible} (${pct}%)`;
  }

  $("#btn-reanudar").hidden = false;
  $("#camara-estado").textContent = "Objeto detectado y cámara en pausa. Puedes prestar, devolver o reanudar el escaneo.";

  if (navigator.vibrate) {
    try {
      navigator.vibrate(60);
    } catch (_) {}
  }

  if (!objeto) {
    mostrarDesconocido(
      `Se detectó la clase "${top1.clase}", pero no tiene ficha. Créala en Agregar objeto.`
    );
  } else {
    estado.objetoActual = objeto;
    estado.ultimaProb = top1.prob;
    renderResultado(objeto, top1.prob);
  }

  enfocarResultado();
}

function reanudarEscaneo() {
  if (!estado.camaraActiva) {
    iniciarCamara();
    return;
  }

  estado.escaneoPausado = false;
  estado.rachaActual = 0;
  estado.claseCandidata = null;

  const video = $("#video");
  if (video && video.paused) {
    video.play().catch(() => {});
  }

  const guide = $("#guide");
  if (guide) {
    guide.classList.remove("is-detected");
    guide.classList.add("is-scanning");
  }

  const laser = $("#scanner-laser");
  if (laser) laser.hidden = false;

  const hud = $("#scanner-hud");
  const hudTexto = $("#scanner-hud-text");
  if (hud) {
    hud.hidden = false;
    hud.classList.remove("is-detected");
  }
  if (hudTexto) {
    hudTexto.textContent = "Buscando objeto…";
  }

  $("#btn-reanudar").hidden = true;
  $("#camara-estado").textContent = "Cámara activa · Auto-detección activada.";
  iniciarBucleEscaneo();
}

async function capturarYReconocer() {
  if (!estado.modelo) {
    toast("Primero entrena el modelo y recarga la página.", true);
    return;
  }

  const video = $("#video");
  if (!video.videoWidth || !video.videoHeight) {
    toast("La cámara todavía no está lista.", true);
    return;
  }

  const canvas = $("#canvas-captura");
  const ctx = canvas.getContext("2d");
  const lado = Math.min(video.videoWidth, video.videoHeight) * FRACCION_ROI;
  const sx = (video.videoWidth - lado) / 2;
  const sy = (video.videoHeight - lado) / 2;

  canvas.width = estado.tamano;
  canvas.height = estado.tamano;
  ctx.drawImage(video, sx, sy, lado, lado, 0, 0, estado.tamano, estado.tamano);

  let tensor = null;
  let salida = null;

  try {
    tensor = tf.tidy(() => tf.browser.fromPixels(canvas).toFloat().expandDims(0));
    salida = estado.modelo.predict(tensor);
    const predicciones = await salida.data();
    const lista = estado.clases
      .map((clase, indice) => ({ clase, prob: Number(predicciones[indice]) || 0 }))
      .sort((a, b) => b.prob - a.prob);

    const top3 = lista.slice(0, 3);
    const top1 = top3[0];

    if (!top1) {
      mostrarDesconocido("El modelo no devolvió probabilidades.");
      return;
    }

    const margen = top1.prob - (top3[1]?.prob || 0);
    const esDesconocido =
      top1.clase.startsWith("_") || top1.prob < UMBRAL_CONFIANZA || margen < MARGEN_MINIMO;

    renderTop3(top3);

    if (esDesconocido) {
      mostrarDesconocido(
        top1.clase.startsWith("_")
          ? "El modelo detectó fondo o un objeto desconocido."
          : "La confianza es baja o hay dos clases muy parejas."
      );
      return;
    }

    const objeto = estado.objetos.find((item) => item.id === top1.clase);
    pausarYConfirmarDeteccion(top1, objeto, top3);
  } catch (error) {
    toast(`No se pudo reconocer: ${error.message}`, true);
    $("#top3").hidden = true;
  } finally {
    if (salida) salida.dispose();
    if (tensor) tensor.dispose();
  }
}

function enfocarResultado() {
  if (!window.matchMedia("(max-width: 779px)").matches) return;

  const visible = !$("#resultado").hidden
    ? $("#resultado")
    : !$("#resultado-desconocido").hidden
      ? $("#resultado-desconocido")
      : null;

  if (!visible) return;

  window.setTimeout(() => {
    visible.scrollIntoView({ behavior: comportamientoScroll(), block: "start" });
  }, 80);
}

function procesarPredicciones(predicciones) {
  const lista = estado.clases
    .map((clase, indice) => ({ clase, prob: Number(predicciones[indice]) || 0 }))
    .sort((a, b) => b.prob - a.prob);

  const top3 = lista.slice(0, 3);
  const top1 = top3[0];

  if (!top1) {
    mostrarDesconocido("El modelo no devolvió probabilidades.");
    return;
  }

  const margen = top1.prob - (top3[1]?.prob || 0);
  const esDesconocido =
    top1.clase.startsWith("_") || top1.prob < UMBRAL_CONFIANZA || margen < MARGEN_MINIMO;

  renderTop3(top3);

  if (esDesconocido) {
    mostrarDesconocido(
      top1.clase.startsWith("_")
        ? "El modelo detectó fondo o un objeto desconocido."
        : "La confianza es baja o hay dos clases muy parejas."
    );
    return;
  }

  const objeto = estado.objetos.find((item) => item.id === top1.clase);
  pausarYConfirmarDeteccion(top1, objeto, top3);
}

function renderTop3(top3) {
  const contenedor = $("#top3-lista");
  contenedor.innerHTML = "";

  top3.forEach((item) => {
    const fila = document.createElement("div");
    fila.className = "top3-item";

    const nombre = document.createElement("span");
    nombre.textContent = item.clase.startsWith("_") ? "desconocido" : item.clase;

    const barra = document.createElement("div");
    barra.className = "bar";
    const relleno = document.createElement("span");
    relleno.style.width = `${Math.max(0, Math.min(100, Math.round(item.prob * 100)))}%`;
    barra.appendChild(relleno);

    const porcentaje = document.createElement("span");
    porcentaje.textContent = `${Math.round(item.prob * 100)}%`;

    fila.append(nombre, barra, porcentaje);
    contenedor.appendChild(fila);
  });

  $("#top3").hidden = false;
}

function renderResultado(objeto, probabilidad) {
  estado.vistaResultado = true;
  $("#resultado-vacio").hidden = true;
  $("#resultado-desconocido").hidden = true;
  $("#resultado").hidden = false;

  $("#resultado-nombre").textContent = objeto.nombre;
  $("#resultado-descripcion").textContent = objeto.descripcion || objeto.categoria || "";
  $("#resultado-confianza").textContent = `${Math.round(probabilidad * 100)}%`;
  $("#resultado-stock").textContent = `${objeto.stock_disponible} / ${objeto.stock_total}`;

  $("#btn-prestar").disabled = objeto.stock_disponible < 1;
  $("#btn-devolver").disabled = objeto.stock_disponible >= objeto.stock_total;
}

function mostrarDesconocido(mensaje) {
  estado.objetoActual = null;
  estado.vistaResultado = false;
  $("#resultado-vacio").hidden = true;
  $("#resultado").hidden = true;
  $("#resultado-desconocido").hidden = false;
  $("#resultado-desconocido").querySelector("p.muted").textContent = mensaje;
}

function limpiarResultado() {
  estado.objetoActual = null;
  estado.vistaResultado = false;
  $("#resultado").hidden = true;
  $("#resultado-desconocido").hidden = true;
  $("#top3").hidden = true;
  $("#resultado-vacio").hidden = false;
}

function actualizarResultadoActual() {
  if (!estado.vistaResultado || !estado.objetoActual) return;

  const actualizado = estado.objetos.find((item) => item.id === estado.objetoActual.id);
  if (!actualizado) {
    estado.objetoActual = null;
    estado.vistaResultado = false;
    return;
  }

  estado.objetoActual = actualizado;
  renderResultado(actualizado, estado.ultimaProb);
}

/* ------------------------------------------------------------------ */
/* Préstamos y devoluciones                                            */
/* ------------------------------------------------------------------ */

function abrirDialogoPara(objetoId, accion) {
  const objeto = estado.objetos.find((item) => item.id === objetoId);
  if (!objeto) return;

  limpiarResultado();
  estado.objetoActual = objeto;
  abrirDialogo(accion);
}

function abrirDialogo(accion) {
  const objeto = estado.objetoActual;
  if (!objeto) return;

  estado.accionPendiente = { accion, objetoId: objeto.id };

  $("#dialogo-titulo").textContent = accion === "prestamo" ? "Prestar objeto" : "Devolver objeto";
  $("#dialogo-detalle").textContent = `${objeto.nombre} · ${objeto.stock_disponible} de ${objeto.stock_total} disponibles`;
  $("#dialogo-persona").value = "";
  $("#dialogo-accion").showModal();

  setTimeout(() => $("#dialogo-persona").focus(), 30);
}

function cerrarDialogo() {
  estado.accionPendiente = null;
  if ($("#dialogo-accion").open) $("#dialogo-accion").close();
}

async function confirmarAccion(evento) {
  evento.preventDefault();

  const pendiente = estado.accionPendiente;
  if (!pendiente) return;

  const persona = $("#dialogo-persona").value.trim();

  try {
    if (pendiente.accion === "prestamo") {
      await DB.registrarPrestamo(pendiente.objetoId, persona);
      toast("Préstamo registrado.");
    } else {
      await DB.registrarDevolucion(pendiente.objetoId, persona);
      toast("Devolución registrada.");
    }

    cerrarDialogo();
    await recargarDatos();
    renderTodo();

    // Reanudar escaneo automáticamente si la cámara sigue activa
    if (estado.camaraActiva && estado.escaneoPausado) {
      setTimeout(() => {
        reanudarEscaneo();
      }, 1200);
    }
  } catch (error) {
    toast(error.message, true);
  }
}

/* ------------------------------------------------------------------ */
/* Inventario                                                          */
/* ------------------------------------------------------------------ */

function renderInventario() {
  const contenedor = $("#lista-inventario");
  const filtro = ($("#buscar-inventario").value || "").toLowerCase().trim();

  const objetos = estado.objetos.filter((objeto) => {
    if (!filtro) return true;
    return `${objeto.nombre} ${objeto.descripcion} ${objeto.categoria}`
      .toLowerCase()
      .includes(filtro);
  });

  contenedor.innerHTML = "";

  if (objetos.length === 0) {
    const vacio = document.createElement("p");
    vacio.className = "muted";
    vacio.textContent = "No hay objetos que coincidan.";
    contenedor.appendChild(vacio);
    return;
  }

  objetos.forEach((objeto) => {
    const entrenado = estado.clases.includes(objeto.id);
    const tarjeta = document.createElement("article");
    tarjeta.className = `inventory-item${objeto.stock_disponible === 0 ? " is-empty" : ""}`;

    const superior = document.createElement("div");
    superior.className = "inventory-top";

    const grupo = document.createElement("div");
    const titulo = document.createElement("h3");
    titulo.textContent = objeto.nombre;
    const descripcion = document.createElement("p");
    descripcion.textContent = objeto.descripcion || objeto.categoria || "";
    grupo.append(titulo, descripcion);

    const etiqueta = document.createElement("span");
    etiqueta.className = `pill ${
      !entrenado ? "pill-warn" : objeto.stock_disponible === 0 ? "pill-warn" : "pill-ok"
    }`;
    etiqueta.textContent = !entrenado
      ? "Sin entrenar"
      : objeto.stock_disponible === 0
        ? "Sin stock"
        : "Disponible";
    etiqueta.title = entrenado
      ? ""
      : "Este objeto todavía no aparece en modelo_web/clases.json. Agrega fotos y entrena.";

    superior.append(grupo, etiqueta);

    const stock = document.createElement("p");
    stock.className = "inventory-stock";
    stock.innerHTML = `<strong>${objeto.stock_disponible}</strong> de ${objeto.stock_total} disponibles`;

    const acciones = document.createElement("div");
    acciones.className = "inventory-actions";

    const prestar = document.createElement("button");
    prestar.className = "btn btn-primary btn-small";
    prestar.type = "button";
    prestar.textContent = "Prestar";
    prestar.disabled = objeto.stock_disponible < 1;
    prestar.addEventListener("click", () => abrirDialogoPara(objeto.id, "prestamo"));

    const devolver = document.createElement("button");
    devolver.className = "btn btn-ghost btn-small";
    devolver.type = "button";
    devolver.textContent = "Devolver";
    devolver.disabled = objeto.stock_disponible >= objeto.stock_total;
    devolver.addEventListener("click", () => abrirDialogoPara(objeto.id, "devolucion"));

    acciones.append(prestar, devolver);
    tarjeta.append(superior, stock, acciones);
    contenedor.appendChild(tarjeta);
  });
}

/* ------------------------------------------------------------------ */
/* Historial                                                           */
/* ------------------------------------------------------------------ */

function renderHistorial() {
  const contenedor = $("#lista-historial");
  const nombres = new Map(estado.objetos.map((objeto) => [objeto.id, objeto.nombre]));

  contenedor.innerHTML = "";

  if (estado.movimientos.length === 0) {
    const vacio = document.createElement("p");
    vacio.className = "muted";
    vacio.textContent = "Sin movimientos todavía.";
    contenedor.appendChild(vacio);
    return;
  }

  estado.movimientos.slice(0, 100).forEach((movimiento) => {
    const fila = document.createElement("div");
    fila.className = "history-item";

    const principal = document.createElement("div");
    principal.className = "history-main";

    const tipo = document.createElement("span");
    tipo.className = `tag tag-${movimiento.tipo}`;
    tipo.textContent = movimiento.tipo;

    const nombre = document.createElement("strong");
    nombre.textContent = nombres.get(movimiento.objeto_id) || movimiento.objeto_id;

    const persona = document.createElement("span");
    persona.className = "muted";
    persona.textContent = `· ${movimiento.persona || "Anónimo"}`;

    principal.append(tipo, nombre, persona);

    const meta = document.createElement("span");
    meta.className = "history-meta";
    meta.textContent = formatearFecha(movimiento.fecha);
    if (movimiento.nota) meta.title = movimiento.nota;

    fila.append(principal, meta);
    contenedor.appendChild(fila);
  });
}

function formatearFecha(iso) {
  try {
    return new Intl.DateTimeFormat("es-BO", {
      dateStyle: "short",
      timeStyle: "short"
    }).format(new Date(iso));
  } catch (_) {
    return iso || "";
  }
}

/* ------------------------------------------------------------------ */
/* Métricas y estado del modelo                                        */
/* ------------------------------------------------------------------ */

function renderMetricas() {
  const contenedor = $("#metricas");
  const metricas = estado.metricas;

  contenedor.innerHTML = "";

  if (!metricas) {
    contenedor.textContent = "Sin métricas todavía. Entrena el modelo y pulsa Recargar modelo.";
    return;
  }

  const resumen = document.createElement("div");
  resumen.className = "metrics-grid";

  const global = document.createElement("div");
  global.className = "metric";
  global.innerHTML = `<span>Accuracy global</span><strong>${formatearPorcentaje(
    metricas.accuracy_global
  )}</strong>`;

  const total = document.createElement("div");
  total.className = "metric";
  total.innerHTML = `<span>Imágenes de validación</span><strong>${metricas.total_validacion ?? "—"}</strong>`;

  const clases = document.createElement("div");
  clases.className = "metric";
  clases.innerHTML = `<span>Clases</span><strong>${metricas.clases?.length ?? estado.clases.length}</strong>`;

  const fecha = document.createElement("div");
  fecha.className = "metric";
  fecha.innerHTML = `<span>Entrenado</span><strong>${metricas.fecha ? formatearFecha(metricas.fecha) : "—"}</strong>`;

  resumen.append(global, total, clases, fecha);
  contenedor.appendChild(resumen);

  if (metricas.accuracy_por_clase) {
    const tabla = document.createElement("table");
    tabla.className = "confusion";
    const thead = document.createElement("thead");
    thead.innerHTML = "<tr><th>Clase</th><th>Fotos</th><th>Accuracy</th></tr>";
    const tbody = document.createElement("tbody");

    Object.entries(metricas.accuracy_por_clase).forEach(([clase, accuracy]) => {
      const tr = document.createElement("tr");
      const tdClase = document.createElement("td");
      tdClase.textContent = clase;
      const tdFotos = document.createElement("td");
      tdFotos.textContent = metricas.conteos?.[clase] ?? "—";
      const tdAccuracy = document.createElement("td");
      tdAccuracy.textContent = formatearPorcentaje(accuracy);
      tr.append(tdClase, tdFotos, tdAccuracy);
      tbody.appendChild(tr);
    });

    tabla.append(thead, tbody);
    contenedor.appendChild(tabla);
  }
}

function formatearPorcentaje(valor) {
  if (typeof valor !== "number" || Number.isNaN(valor)) return "—";
  return `${(valor * 100).toFixed(1)}%`;
}

function renderModeloDetalle(mensajeError = estado.errorModelo) {
  const detalle = $("#modelo-detalle");

  if (estado.modelo) {
    const clases = estado.clases.filter((clase) => !clase.startsWith("_"));
    detalle.textContent = `Modelo activo con ${estado.clases.length} clases (${clases.join(
      ", "
    )}). El reconocimiento está habilitado.`;
    return;
  }

  detalle.textContent =
    mensajeError ||
    "No hay modelo cargado. Agrega fotos a dataset/ y ejecuta python entrenar.py.";
}

/* ------------------------------------------------------------------ */
/* Agregar objeto                                                      */
/* ------------------------------------------------------------------ */

function slugify(texto) {
  return texto
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 40);
}

function actualizarVistaId() {
  const nombre = $("#obj-nombre").value.trim();
  const id = nombre ? slugify(nombre) : "";
  $("#obj-id-preview").textContent = id || "—";
}

async function crearObjeto(evento) {
  evento.preventDefault();

  $("#alta-resultado").hidden = true;

  const nombre = $("#obj-nombre").value.trim();
  const descripcion = $("#obj-descripcion").value.trim();
  const categoria = $("#obj-categoria").value.trim() || "General";
  const stock = Number.parseInt($("#obj-stock").value, 10);

  if (!nombre) {
    toast("Escribe un nombre.", true);
    return;
  }
  if (!Number.isFinite(stock) || stock < 1) {
    toast("El stock debe ser un número mayor o igual a 1.", true);
    return;
  }

  const base = slugify(nombre);
  if (!base) {
    toast("El nombre debe contener letras o números.", true);
    return;
  }

  let id = base;
  let contador = 2;
  while (estado.objetos.some((objeto) => objeto.id === id)) {
    id = `${base}_${contador}`;
    contador += 1;
  }

  try {
    await DB.crearObjeto({
      id,
      nombre,
      descripcion,
      categoria,
      stock_total: stock
    });

    await recargarDatos();
    renderTodo();

    estado.rutaAlta = `dataset/${id}/`;
    $("#alta-ruta").textContent = estado.rutaAlta;
    $("#alta-resultado").hidden = false;
    $("#form-objeto").reset();
    $("#obj-categoria").value = "General";
    $("#obj-stock").value = "1";
    actualizarVistaId();
    enfocarAlta();

    toast(`Objeto "${nombre}" creado. Agrega las fotos en ${estado.rutaAlta}`);
  } catch (error) {
    toast(error.message, true);
  }
}

function enfocarAlta() {
  if (!window.matchMedia("(max-width: 779px)").matches) return;

  window.setTimeout(() => {
    $("#alta-resultado").scrollIntoView({
      behavior: comportamientoScroll(),
      block: "center"
    });
  }, 80);
}

async function copiarRuta() {
  const ruta = estado.rutaAlta || $("#alta-ruta").textContent;
  if (!ruta) return;

  try {
    await navigator.clipboard.writeText(ruta);
    toast("Ruta copiada.");
  } catch (_) {
    toast(`Copia manualmente: ${ruta}`, true);
  }
}

/* ------------------------------------------------------------------ */
/* Eventos                                                             */
/* ------------------------------------------------------------------ */

function configurarEventos() {
  $("#btn-camara").addEventListener("click", () => {
    if (estado.camaraActiva) detenerCamara();
    else iniciarCamara();
  });

  $("#btn-cambiar-camara").addEventListener("click", async () => {
    estado.facing = estado.facing === "environment" ? "user" : "environment";
    if (estado.camaraActiva) await iniciarCamara();
  });

  $("#btn-capturar").addEventListener("click", capturarYReconocer);
  $("#btn-reanudar").addEventListener("click", reanudarEscaneo);
  $("#btn-resultado-reanudar").addEventListener("click", reanudarEscaneo);

  $("#btn-prestar").addEventListener("click", () => abrirDialogo("prestamo"));
  $("#btn-devolver").addEventListener("click", () => abrirDialogo("devolucion"));

  $("#btn-refrescar-historial").addEventListener("click", async () => {
    await recargarDatos();
    renderHistorial();
    toast("Historial actualizado.");
  });

  $("#btn-reload-model").addEventListener("click", async () => {
    toast("Cargando modelo…");
    await cargarModelo();
  });

  $("#btn-tab-agregar").addEventListener("click", () => activarTab("agregar"));

  $("#form-objeto").addEventListener("submit", crearObjeto);
  $("#obj-nombre").addEventListener("input", actualizarVistaId);
  $("#btn-copiar-ruta").addEventListener("click", copiarRuta);

  $("#buscar-inventario").addEventListener("input", renderInventario);

  $("#form-accion").addEventListener("submit", confirmarAccion);
  $("#btn-dialogo-cancelar").addEventListener("click", cerrarDialogo);
  $("#dialogo-accion").addEventListener("cancel", () => {
    estado.accionPendiente = null;
  });

  window.addEventListener("beforeunload", () => {
    if (estado.camaraActiva) detenerCamara(false);
  });
}

/* ------------------------------------------------------------------ */
/* Avisos                                                              */
/* ------------------------------------------------------------------ */

let temporizadorToast = null;

function toast(mensaje, esError = false) {
  const el = $("#toast");
  el.textContent = mensaje;
  el.hidden = false;
  el.classList.toggle("is-error", esError);

  clearTimeout(temporizadorToast);
  temporizadorToast = setTimeout(() => {
    el.hidden = true;
  }, 4000);
}
