#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║   Nikon D3200 · Auto-Proceso Inteligente · Balonmano Pro        ║
║   OPPO Reno 2Z · Termux · Ubuntu proot                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Cada foto se ANALIZA individualmente antes de procesarse:      ║
║  • Detecta si está subexpuesta/sobreexpuesta y lo corrige       ║
║  • Mide el nivel de ruido real (ISO alto en pabellón)           ║
║  • Detecta dominante de color (fluorescente/tungsteno)          ║
║  • Evalúa la nitidez real y aplica la corrección exacta         ║
║  • Ajusta contraste según el histograma real de la imagen       ║
║  • Todos los parámetros son 100% automáticos por foto           ║
╚══════════════════════════════════════════════════════════════════╝

Uso:
  python3 nikon_capture.py            → modo automático completo
  python3 nikon_capture.py --once     → descarga + procesa y sale
  python3 nikon_capture.py --config   → muestra config de la cámara
  python3 nikon_capture.py --process DSC_0001.NEF  → procesa un NEF
  python3 nikon_capture.py --pending  → procesa todos los NEF pendientes
  python3 nikon_capture.py --dir /ruta/custom  → ruta personalizada
"""

import os
import sys
import time
import logging
import argparse
import datetime
import math
from pathlib import Path

# ════════════════════════════════════════════════════════════
#  RUTAS
# ════════════════════════════════════════════════════════════
POSIBLES_RUTAS = [
    # Memoria interna Android (OPPO Reno 2Z)
    "/storage/emulated/0/NIKON_RAW",        # ruta real memoria interna
    "/storage/emulated/0/Pictures/NIKON_RAW",  # alternativa dentro de Fotos
    os.path.expanduser("~/storage/shared/NIKON_RAW"),  # Termux → memoria interna
    os.path.expanduser("~/storage/pictures/NIKON_RAW"),  # Termux → carpeta Fotos
    os.path.expanduser("~/NIKON_RAW"),      # fallback: home de Ubuntu proot
]

def obtener_ruta_base():
    for ruta in POSIBLES_RUTAS:
        padre = str(Path(ruta).parent)
        if os.path.exists(padre) or os.path.exists(ruta):
            for sub in ("originales", "procesadas", "logs"):
                os.makedirs(f"{ruta}/{sub}", exist_ok=True)
            return ruta
    fb = os.path.expanduser("~/NIKON_RAW")
    for sub in ("originales", "procesadas", "logs"):
        os.makedirs(f"{fb}/{sub}", exist_ok=True)
    return fb

BASE_DIR = obtener_ruta_base()
DIR_ORIG = f"{BASE_DIR}/originales"
DIR_PROC = f"{BASE_DIR}/procesadas"
DIR_LOGS = f"{BASE_DIR}/logs"

# ════════════════════════════════════════════════════════════
#  LOGGING
# ════════════════════════════════════════════════════════════
log_file = f"{DIR_LOGS}/nikon_{datetime.datetime.now():%Y%m%d}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("nikon")

# ════════════════════════════════════════════════════════════
#  IMPORTACIONES OPCIONALES
# ════════════════════════════════════════════════════════════
try:
    import gphoto2 as gp
    GPHOTO_OK = True
except ImportError:
    GPHOTO_OK = False
    log.warning("gphoto2 no disponible → pip install gphoto2")

try:
    import rawpy
    RAW_OK = True
except ImportError:
    RAW_OK = False
    log.warning("rawpy no disponible → pip install rawpy")

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np
    PIL_OK = True
except ImportError:
    PIL_OK = False
    log.warning("Pillow/numpy no disponibles → pip install Pillow numpy")


# ════════════════════════════════════════════════════════════
#  MÓDULO 1 · ANALIZADOR INTELIGENTE
#  Mide la imagen real y devuelve parámetros exactos para ella
# ════════════════════════════════════════════════════════════

class AnalizadorImagen:
    """
    Analiza un array numpy RGB uint16 o uint8.
    Devuelve un dict con todos los parámetros de corrección
    calculados automáticamente para esa foto concreta.
    """

    def __init__(self, arr):
        self.arr_orig = arr
        if arr.dtype == np.uint16:
            self.f = arr.astype(np.float32) / 65535.0
        else:
            self.f = arr.astype(np.float32) / 255.0
        self.h, self.w = self.f.shape[:2]

    # ── Luminosidad perceptual ─────────────────────────────
    def _lum(self):
        r, g, b = self.f[:,:,0], self.f[:,:,1], self.f[:,:,2]
        return float(np.mean(0.2126*r + 0.7152*g + 0.0722*b))

    # ── Histograma: zonas quemadas y aplastadas ────────────
    def _histograma(self):
        gray = np.mean(self.f, axis=2)
        hist, _ = np.histogram(gray, bins=256, range=(0,1))
        total = max(hist.sum(), 1)
        sombras = hist[:15].sum() / total   # píxeles muy oscuros
        altas   = hist[240:].sum() / total  # píxeles quemados
        return float(sombras), float(altas)

    # ── Nivel de ruido ─────────────────────────────────────
    def _ruido(self):
        gray  = np.mean(self.f, axis=2).astype(np.float32)
        dx    = np.diff(gray, axis=1)
        dy    = np.diff(gray, axis=0)
        valor = float(np.std(dx)) + float(np.std(dy))
        return min(valor / 0.30, 1.0)

    # ── Nitidez (varianza del Laplaciano simplificado) ─────
    def _nitidez(self):
        gray    = np.mean(self.f, axis=2)
        dx2     = np.diff(gray, 2, axis=1)
        dy2     = np.diff(gray, 2, axis=0)
        varianza = float(np.var(dx2)) + float(np.var(dy2))
        return min(varianza / 0.005, 1.0)

    # ── Contraste global ───────────────────────────────────
    def _contraste(self):
        gray = np.mean(self.f, axis=2)
        return min(float(np.std(gray)) / 0.35, 1.0)

    # ── Dominante de color (cast por luz artificial) ───────
    def _dominante(self):
        mr = float(np.mean(self.f[:,:,0]))
        mg = float(np.mean(self.f[:,:,1]))
        mb = float(np.mean(self.f[:,:,2]))
        media = (mr + mg + mb) / 3.0
        if media < 0.001:
            return 1.0, 1.0, 1.0
        cr = max(0.70, min(media / max(mr, 0.001), 1.50))
        cg = max(0.70, min(media / max(mg, 0.001), 1.50))
        cb = max(0.70, min(media / max(mb, 0.001), 1.50))
        # Sólo corregir si la dominante es notable
        if abs(cr-1) < 0.06 and abs(cg-1) < 0.06 and abs(cb-1) < 0.06:
            return 1.0, 1.0, 1.0
        return cr, cg, cb

    # ── MÉTODO PRINCIPAL ──────────────────────────────────
    def calcular(self):
        lum            = self._lum()
        sombras, altas = self._histograma()
        ruido          = self._ruido()
        contraste      = self._contraste()
        nitidez        = self._nitidez()
        cr, cg, cb     = self._dominante()

        log.info(f"    Análisis → lum={lum:.3f}  ruido={ruido:.3f}  "
                 f"nitidez={nitidez:.3f}  contraste={contraste:.3f}  "
                 f"sombras={sombras:.1%}  altas={altas:.1%}")

        # ── Brillo ────────────────────────────────────────
        # Objetivo: luminosidad ~0.44 (sombras ricas, altas luces vivas)
        OBJETIVO = 0.44
        if lum < 0.001:
            fb = 1.60
        else:
            fb = OBJETIVO / lum
        if altas > 0.05:
            fb = min(fb, 1.08)   # no quemar si ya hay zonas saturadas
        fb = max(0.65, min(fb, 2.30))

        # ── Contraste ─────────────────────────────────────
        if contraste < 0.35:
            fc = 1.55   # imagen muy plana
        elif contraste < 0.60:
            fc = 1.28
        else:
            fc = 1.10   # ya contrastada: toque suave

        # ── Saturación ────────────────────────────────────
        # Equipaciones de balonmano: colores vivos pero no irreales
        fc_color = 1.18

        # ── Reducción de ruido en rawpy ───────────────────
        noise_thr = int(80 + ruido * 220)   # rango 80-300

        # ── Filtro de mediana (post-rawpy) ────────────────
        if ruido > 0.75:
            nr = 3       # ISO 6400 en pabellón
        elif ruido > 0.40:
            nr = 2       # ISO 1600-3200
        else:
            nr = 1       # ISO 400-800

        # ── Unsharp Mask ──────────────────────────────────
        if nitidez < 0.20:
            usm_r, usm_p, usm_t = 1, 120, 4   # imagen borrosa: suave
        elif nitidez < 0.55:
            usm_r, usm_p, usm_t = 1, 185, 2   # normal: estándar
        else:
            usm_r, usm_p, usm_t = 2, 225, 1   # muy nítida: agresivo

        # ── Gamma ─────────────────────────────────────────
        if lum < 0.25:
            gamma = 0.72    # muy oscura → aclarar sombras
        elif lum < 0.38:
            gamma = 0.84
        else:
            gamma = 0.95

        # ── Intensidad curva tonal en S ───────────────────
        if sombras > 0.25 or altas > 0.10:
            curva = 0.12    # histograma extremo → curva suave
        else:
            curva = 0.28    # normal → curva marcada

        return {
            # rawpy
            "use_camera_wb":         True,
            "use_auto_wb":           False,
            "no_auto_bright":        False,
            "output_bps":            16,
            "demosaic_algorithm":    rawpy.DemosaicAlgorithm.AHD,
            "median_filter_passes":  2,
            "dcb_enhance":           False,
            "fbdd_noise_reduction":  rawpy.FBDDNoiseReductionMode.Full,
            "noise_thr":             noise_thr,
            # Pillow
            "brillo":                round(fb, 3),
            "contraste":             round(fc, 3),
            "color":                 round(fc_color, 3),
            "gamma":                 round(gamma, 3),
            "usm_r":                 usm_r,
            "usm_p":                 usm_p,
            "usm_t":                 usm_t,
            "nr":                    nr,
            "cr":                    round(cr, 3),
            "cg":                    round(cg, 3),
            "cb":                    round(cb, 3),
            "curva":                 curva,
            "calidad":               97,
        }


# ════════════════════════════════════════════════════════════
#  MÓDULO 2 · PROCESADOR PRO
# ════════════════════════════════════════════════════════════

class ProcesadorPro:

    def procesar(self, ruta_nef):
        if not RAW_OK or not PIL_OK:
            log.error("rawpy o Pillow no instalados.")
            return None

        nombre_base = Path(ruta_nef).stem
        fecha       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        salida      = os.path.join(DIR_PROC, f"{nombre_base}_{fecha}.jpg")

        try:
            log.info(f"▶ {Path(ruta_nef).name}")

            # ── 1. Primera lectura RAW para análisis ─────────
            with rawpy.imread(ruta_nef) as raw:
                rgb16 = raw.postprocess(
                    use_camera_wb        = True,
                    use_auto_wb          = False,
                    no_auto_bright       = True,
                    output_bps           = 16,
                    demosaic_algorithm   = rawpy.DemosaicAlgorithm.AHD,
                    median_filter_passes = 1,
                    dcb_enhance          = False,
                    fbdd_noise_reduction = rawpy.FBDDNoiseReductionMode.Full,
                    noise_thr            = 100,
                )
            log.info(f"  {rgb16.shape[1]}×{rgb16.shape[0]} px  "
                     f"({rgb16.shape[1]*rgb16.shape[0]/1e6:.1f} Mpx)")

            # ── 2. Analizar y calcular parámetros ─────────────
            p = AnalizadorImagen(rgb16).calcular()
            log.info(f"  Params → brillo={p['brillo']}  contraste={p['contraste']}  "
                     f"color={p['color']}  gamma={p['gamma']}  "
                     f"USM={p['usm_p']}%  NR={p['nr']}  "
                     f"noise_thr={p['noise_thr']}")

            # ── 3. Segunda lectura RAW con parámetros finales ─
            with rawpy.imread(ruta_nef) as raw:
                rgb16 = raw.postprocess(
                    use_camera_wb        = p["use_camera_wb"],
                    use_auto_wb          = p["use_auto_wb"],
                    no_auto_bright       = p["no_auto_bright"],
                    output_bps           = 16,
                    demosaic_algorithm   = p["demosaic_algorithm"],
                    median_filter_passes = p["median_filter_passes"],
                    dcb_enhance          = p["dcb_enhance"],
                    fbdd_noise_reduction = p["fbdd_noise_reduction"],
                    noise_thr            = p["noise_thr"],
                )

            # ── 4. Normalizar a uint8 ─────────────────────────
            arr = (rgb16.astype(np.float32) / 256.0).astype(np.uint8)
            img = Image.fromarray(arr)

            # ── 5. Corrección de dominante de color ───────────
            if not (p["cr"] == 1.0 and p["cg"] == 1.0 and p["cb"] == 1.0):
                a = np.array(img, dtype=np.float32)
                a[:,:,0] = np.clip(a[:,:,0] * p["cr"], 0, 255)
                a[:,:,1] = np.clip(a[:,:,1] * p["cg"], 0, 255)
                a[:,:,2] = np.clip(a[:,:,2] * p["cb"], 0, 255)
                img = Image.fromarray(a.astype(np.uint8))

            # ── 6. Corrección de gamma ─────────────────────────
            lut = [int(255 * (i/255.0) ** p["gamma"]) for i in range(256)]
            img = img.point(lut * 3)

            # ── 7. Reducción de ruido ─────────────────────────
            if p["nr"] > 1:
                img = img.filter(ImageFilter.MedianFilter(size=p["nr"]))

            # ── 8. Brillo ─────────────────────────────────────
            img = ImageEnhance.Brightness(img).enhance(p["brillo"])

            # ── 9. Contraste ──────────────────────────────────
            img = ImageEnhance.Contrast(img).enhance(p["contraste"])

            # ── 10. Saturación / Color ────────────────────────
            img = ImageEnhance.Color(img).enhance(p["color"])

            # ── 11. Nitidez Unsharp Mask ──────────────────────
            img = img.filter(ImageFilter.UnsharpMask(
                radius    = p["usm_r"],
                percent   = p["usm_p"],
                threshold = p["usm_t"],
            ))

            # ── 12. Curva tonal en S ──────────────────────────
            a       = np.array(img, dtype=np.float32) / 255.0
            curvada = 0.5 * np.sin(math.pi * (a - 0.5)) + 0.5
            a       = (1 - p["curva"]) * a + p["curva"] * curvada
            img     = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

            # ── 13. Recuperar altas luces quemadas ────────────
            a     = np.array(img, dtype=np.float32) / 255.0
            lum_a = 0.2126*a[:,:,0] + 0.7152*a[:,:,1] + 0.0722*a[:,:,2]
            mask  = np.clip((lum_a - 0.85) / 0.15, 0, 1)[:,:, np.newaxis]
            a     = a * (1 - mask * 0.08)
            img   = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

            # ── 14. Guardar JPEG de máxima calidad ───────────
            img.save(
                salida,
                format      = "JPEG",
                quality     = p["calidad"],
                optimize    = True,
                subsampling = 0,        # 4:4:4 → sin pérdida de color
                progressive = True,
            )

            mb = os.path.getsize(salida) / (1024*1024)
            log.info(f"  ✓ {Path(salida).name}  ({mb:.1f} MB)")
            return salida

        except Exception as e:
            log.error(f"  ✗ Error procesando {ruta_nef}: {e}", exc_info=True)
            return None


# ════════════════════════════════════════════════════════════
#  MÓDULO 3 · CÁMARA (gphoto2)
# ════════════════════════════════════════════════════════════

PARAMS_CAM = [
    ("iso",                  "ISO"),
    ("shutterspeed",         "Velocidad obturador"),
    ("aperture",             "Apertura"),
    ("expprogram",           "Programa"),
    ("imageformat",          "Formato imagen"),
    ("whitebalance",         "Balance blancos"),
    ("focusmode",            "Modo enfoque"),
    ("meteringmode",         "Medición"),
    ("flashmode",            "Flash"),
    ("exposurecompensation", "Comp. exposición"),
    ("capturetarget",        "Destino captura"),
    ("batterylevel",         "Batería"),
    ("datetime",             "Fecha/hora"),
    ("cameramodel",          "Modelo"),
]

def conectar_camara(reintentos=999, espera=3):
    if not GPHOTO_OK:
        raise RuntimeError("gphoto2 no instalado → pip install gphoto2")
    log.info("Buscando Nikon D3200 por USB…")
    for i in range(1, reintentos+1):
        try:
            ctx = gp.Context()
            cam = gp.Camera()
            cam.init(ctx)
            modelo = cam.get_abilities().model
            log.info(f"✓ Cámara detectada: {modelo}")
            return cam, ctx
        except gp.GPhoto2Error:
            if i % 10 == 0:
                log.info(f"  [{i}] Cámara no encontrada, esperando…")
            time.sleep(espera)
    raise RuntimeError("No se pudo conectar con la cámara.")

def leer_config(cam, ctx):
    cfg = cam.get_config(ctx)
    res = {}
    for clave, etiqueta in PARAMS_CAM:
        try:
            res[etiqueta] = cfg.get_child_by_name(clave).get_value()
        except gp.GPhoto2Error:
            res[etiqueta] = "(no disponible)"
    return res

def mostrar_config(cfg):
    W = 50
    print("\n┌" + "─"*W + "┐")
    print(f"│{'  CONFIGURACIÓN NIKON D3200':^{W}}│")
    print("├" + "─"*W + "┤")
    for k, v in cfg.items():
        linea = f"  {k:<26} {str(v)}"
        print(f"│{linea:<{W}}│")
    print("└" + "─"*W + "┘\n")

def listar_archivos(cam, ctx, carpeta="/"):
    archivos = []
    try:
        for nombre, _ in cam.folder_list_files(carpeta, ctx):
            archivos.append((carpeta, nombre))
    except gp.GPhoto2Error:
        pass
    try:
        for sub, _ in cam.folder_list_folders(carpeta, ctx):
            archivos.extend(listar_archivos(cam, ctx, os.path.join(carpeta, sub)))
    except gp.GPhoto2Error:
        pass
    return archivos

def descargar(cam, ctx, carpeta, nombre, destino_dir):
    destino = os.path.join(destino_dir, nombre)
    if os.path.exists(destino):
        return None
    log.info(f"  ↓ {nombre}")
    f = cam.file_get(carpeta, nombre, gp.GP_FILE_TYPE_NORMAL, ctx)
    f.save(destino)
    mb = os.path.getsize(destino) / (1024*1024)
    log.info(f"    → {mb:.1f} MB")
    return destino

def descargar_todos(cam, ctx):
    log.info("Listando archivos en la cámara…")
    archivos = listar_archivos(cam, ctx)
    log.info(f"  Archivos encontrados: {len(archivos)}")
    descargados = []
    for carpeta, nombre in archivos:
        if Path(nombre).suffix.upper() in (".NEF", ".NRW"):
            ruta = descargar(cam, ctx, carpeta, nombre, DIR_ORIG)
            if ruta:
                descargados.append(ruta)
    log.info(f"  Descargados: {len(descargados)} nuevo(s)")
    return descargados

def captura_continua(cam, ctx, intervalo=2):
    log.info("Modo captura continua (Ctrl+C para detener)…")
    vistos     = set()
    procesador = ProcesadorPro()
    while True:
        try:
            archivos = listar_archivos(cam, ctx)
            for item in archivos:
                if item not in vistos:
                    carpeta, nombre = item
                    if Path(nombre).suffix.upper() in (".NEF", ".NRW"):
                        ruta = descargar(cam, ctx, carpeta, nombre, DIR_ORIG)
                        if ruta:
                            procesador.procesar(ruta)
                    vistos.add(item)
            time.sleep(intervalo)
        except gp.GPhoto2Error as e:
            log.warning(f"Cámara desconectada: {e}")
            break
        except KeyboardInterrupt:
            log.info("Captura continua detenida.")
            break


# ════════════════════════════════════════════════════════════
#  MÓDULO 4 · MODO AUTOMÁTICO PRINCIPAL
# ════════════════════════════════════════════════════════════

def modo_automatico():
    W = 56
    print("\n╔" + "═"*W + "╗")
    print(f"║{'  NIKON D3200 · AUTO-PROCESO INTELIGENTE':^{W}}║")
    print(f"║{'  Balonmano · 100% Automático':^{W}}║")
    print("╠" + "═"*W + "╣")
    print(f"║  Destino  : {BASE_DIR:<{W-13}}║")
    print(f"║  Análisis : por foto (brillo, ruido, nitidez, color){'':<{W-53}}║")
    print(f"║  Ctrl+C   : detener{'':<{W-20}}║")
    print("╚" + "═"*W + "╝\n")

    procesador = ProcesadorPro()

    while True:
        try:
            log.info("Esperando conexión de la cámara…")
            cam, ctx = conectar_camara()
            try:
                mostrar_config(leer_config(cam, ctx))
            except Exception as e:
                log.warning(f"No se pudo leer config: {e}")

            log.info("Descargando fotos existentes…")
            nuevos = descargar_todos(cam, ctx)
            for nef in nuevos:
                procesador.procesar(nef)

            captura_continua(cam, ctx)

            try:
                cam.exit(ctx)
            except Exception:
                pass

            log.info("Cámara desconectada. Esperando reconexión…\n")
            time.sleep(5)

        except RuntimeError as e:
            log.error(str(e))
            sys.exit(1)
        except KeyboardInterrupt:
            log.info("\nSistema detenido.")
            sys.exit(0)

def procesar_pendientes():
    nefs = (list(Path(DIR_ORIG).glob("*.NEF")) +
            list(Path(DIR_ORIG).glob("*.nef")) +
            list(Path(DIR_ORIG).glob("*.NRW")))
    if not nefs:
        log.info("No hay RAW pendientes.")
        return
    log.info(f"Procesando {len(nefs)} archivo(s) pendiente(s)…")
    p = ProcesadorPro()
    for nef in nefs:
        p.procesar(str(nef))


# ════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Nikon D3200 · Auto-proceso inteligente · Balonmano Pro"
    )
    parser.add_argument("--config",  action="store_true",
                        help="Mostrar configuración de la cámara y salir")
    parser.add_argument("--once",    action="store_true",
                        help="Descargar y procesar una vez y salir")
    parser.add_argument("--process", metavar="ARCHIVO.NEF",
                        help="Procesar un NEF concreto")
    parser.add_argument("--pending", action="store_true",
                        help="Procesar todos los NEF pendientes en originales/")
    parser.add_argument("--dir",     metavar="RUTA",
                        help="Ruta base personalizada")
    args = parser.parse_args()

    if args.dir:
        global BASE_DIR, DIR_ORIG, DIR_PROC, DIR_LOGS
        BASE_DIR = args.dir
        DIR_ORIG = f"{BASE_DIR}/originales"
        DIR_PROC = f"{BASE_DIR}/procesadas"
        DIR_LOGS = f"{BASE_DIR}/logs"
        for d in [DIR_ORIG, DIR_PROC, DIR_LOGS]:
            os.makedirs(d, exist_ok=True)

    if args.process:
        r = ProcesadorPro().procesar(args.process)
        print(f"\n{'✓ ' + r if r else '✗ Falló el procesado'}")
        sys.exit(0 if r else 1)

    if args.pending:
        procesar_pendientes()
        sys.exit(0)

    if not GPHOTO_OK:
        log.error("gphoto2 no instalado → pip install gphoto2")
        sys.exit(1)

    if args.config:
        cam, ctx = conectar_camara(reintentos=3)
        mostrar_config(leer_config(cam, ctx))
        cam.exit(ctx)
        sys.exit(0)

    if args.once:
        cam, ctx = conectar_camara()
        mostrar_config(leer_config(cam, ctx))
        nuevos = descargar_todos(cam, ctx)
        p = ProcesadorPro()
        for nef in nuevos:
            p.procesar(nef)
        cam.exit(ctx)
        log.info(f"Listo. {len(nuevos)} foto(s) procesada(s).")
        sys.exit(0)

    modo_automatico()


if __name__ == "__main__":
    main()
