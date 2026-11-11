"""
Aplicación de escritorio — Detección de objetos con YOLO en video / webcam
Ejecutar:  python gui.py
"""
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("QtAgg")
matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                            "axes.labelsize": 8, "legend.fontsize": 7.5})
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSizePolicy,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from yolo_video import cargar_modelo, generar_hallazgos, guardar_resultados, procesar_video

# En Linux, opencv-python apunta Qt a sus propios plugins y rompe PySide6.
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_FONTDIR", None)

BASE = Path(__file__).parent
ACENTO = "#2563eb"
VERDE, ROJO, NARANJA = "#16a34a", "#dc2626", "#f59e0b"

ESTILO = f"""
QMainWindow, QWidget {{ background: #f6f7fb; color: #1f2937; font-size: 13px; }}
QGroupBox {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px;
             margin-top: 14px; padding: 12px 10px 10px 10px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #374151; }}
QPushButton {{ background: white; border: 1px solid #d1d5db; border-radius: 7px; padding: 7px 10px; }}
QPushButton:hover {{ background: #eef2ff; border-color: {ACENTO}; }}
QPushButton:disabled {{ color: #9ca3af; background: #f3f4f6; }}
QPushButton#primario {{ background: {ACENTO}; color: white; border: none; font-weight: 600; padding: 10px; }}
QPushButton#primario:hover {{ background: #1d4ed8; }}
QPushButton#primario:disabled {{ background: #93c5fd; }}
QPushButton#detener {{ background: white; color: {ROJO}; border: 1px solid #fca5a5; font-weight: 600; padding: 10px; }}
QTabWidget::pane {{ border: 1px solid #e5e7eb; border-radius: 10px; background: white; top: -1px; }}
QTabBar::tab {{ background: transparent; padding: 8px 16px; margin-right: 2px; color: #6b7280; }}
QTabBar::tab:selected {{ color: {ACENTO}; border-bottom: 2px solid {ACENTO}; font-weight: 600; }}
QFrame#tarjeta {{ background: white; border: 1px solid #e5e7eb; border-radius: 10px; }}
QLabel {{ background: transparent; }}
QGroupBox QCheckBox {{ background: white; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid #9ca3af; border-radius: 3px; background: white; }}
QCheckBox::indicator:checked {{ background: {ACENTO}; border-color: {ACENTO}; }}
QLabel#video {{ background: #111827; border-radius: 10px; color: #9ca3af; }}
QProgressBar {{ border: none; background: #e5e7eb; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {ACENTO}; border-radius: 4px; }}
QListWidget, QComboBox, QDoubleSpinBox, QTableWidget, QTextBrowser {{
    background: white; border: 1px solid #d1d5db; border-radius: 6px; padding: 3px; }}
"""


# ------------------------------------------------------------------ widgets
class Tarjeta(QFrame):
    """Tarjeta de indicador: título, valor grande y subtítulo."""

    def __init__(self, titulo):
        super().__init__()
        self.setObjectName("tarjeta")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(2)
        self.t = QLabel(titulo)
        self.t.setStyleSheet("color:#6b7280; font-size:12px; background:transparent;")
        self.v = QLabel("—")
        self.v.setStyleSheet("font-size:24px; font-weight:600; background:transparent;")
        self.s = QLabel("")
        self.s.setStyleSheet("color:#6b7280; font-size:11px; background:transparent;")
        for w in (self.t, self.v, self.s):
            lay.addWidget(w)

    def set(self, valor, sub="", color=None):
        self.v.setText(str(valor))
        self.v.setStyleSheet(f"font-size:24px; font-weight:600; background:transparent;"
                             f"{'color:' + color if color else ''}")
        self.s.setText(sub)


class Grafica(FigureCanvasQTAgg):
    def __init__(self, filas=1, cols=1, alto=4):
        self.fig = Figure(figsize=(8, alto), tight_layout=True)
        super().__init__(self.fig)
        self.filas, self.cols = filas, cols
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def ejes(self):
        self.fig.clear()
        axs = self.fig.subplots(self.filas, self.cols, squeeze=False).ravel()
        for ax in axs:
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25)
        return axs


def a_pixmap(frame_bgr, ancho, alto):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(img).scaled(ancho, alto, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ------------------------------------------------------------------ hilo de trabajo
class Trabajador(QThread):
    frame = Signal(object, dict, float)
    terminado = Signal(object)
    error = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self.parar = False
        self._ultimo = 0.0

    def on_frame(self, frame, fila, prog):
        ahora = time.perf_counter()
        if ahora - self._ultimo > 0.03:          # máx ~30 actualizaciones/s a la ventana
            self._ultimo = ahora
            self.frame.emit(frame.copy(), dict(fila), prog)
        return not self.parar

    def run(self):
        try:
            rows, res, ex = procesar_video(on_frame=self.on_frame, **self.p)
            self.terminado.emit((rows, res, ex))
        except Exception as e:
            self.error.emit(str(e))


# ------------------------------------------------------------------ ventana
class Ventana(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección de objetos con YOLO en video")
        self.resize(1400, 860)
        self.fuente = None
        self.trabajador = None
        self.resultado = None
        self.carpeta = None
        self.nombres = cargar_modelo("yolo11n.pt").names

        central = QWidget()
        self.setCentralWidget(central)
        raiz = QHBoxLayout(central)
        raiz.setContentsMargins(14, 14, 14, 14)
        raiz.setSpacing(14)
        raiz.addWidget(self._panel_izquierdo())
        raiz.addWidget(self._panel_derecho(), 1)

    # ---------- panel de control
    def _panel_izquierdo(self):
        panel = QWidget()
        panel.setFixedWidth(300)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        titulo = QLabel("🎯 YOLO en video")
        titulo.setStyleSheet("font-size:20px; font-weight:700;")
        lay.addWidget(titulo)
        sub = QLabel("Detección y seguimiento de objetos frame a frame")
        sub.setStyleSheet("color:#6b7280;")
        sub.setWordWrap(True)
        lay.addWidget(sub)

        g = QGroupBox("1. Fuente")
        gl = QVBoxLayout(g)
        b1 = QPushButton("📂  Abrir video…")
        b1.clicked.connect(self.abrir_video)
        b2 = QPushButton("📷  Usar webcam")
        b2.clicked.connect(lambda: self.set_fuente("0"))
        b3 = QPushButton("🎞️  Video de ejemplo")
        b3.clicked.connect(lambda: self.set_fuente(str(BASE / "video_prueba.mp4")))
        b3.setEnabled((BASE / "video_prueba.mp4").exists())
        self.lbl_fuente = QLabel("Ninguna fuente seleccionada")
        self.lbl_fuente.setWordWrap(True)
        self.lbl_fuente.setStyleSheet("color:#6b7280; font-weight:400;")
        for w in (b1, b2, b3, self.lbl_fuente):
            gl.addWidget(w)
        lay.addWidget(g)

        g = QGroupBox("2. Configuración")
        gl = QGridLayout(g)
        self.cb_modelo = QComboBox()
        self.cb_modelo.addItems(["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"])
        self.sp_conf = QDoubleSpinBox(minimum=0.05, maximum=0.95, singleStep=0.05, value=0.35)
        self.sp_iou = QDoubleSpinBox(minimum=0.1, maximum=0.9, singleStep=0.05, value=0.5)
        self.cb_imgsz = QComboBox()
        self.cb_imgsz.addItems(["320", "416", "512", "640", "800"])
        self.cb_imgsz.setCurrentText("640")
        filas = [("Modelo", self.cb_modelo), ("Confianza mín.", self.sp_conf),
                 ("IoU (NMS)", self.sp_iou), ("Tamaño entrada", self.cb_imgsz)]
        for i, (t, w) in enumerate(filas):
            l = QLabel(t)
            l.setStyleSheet("font-weight:400;")
            gl.addWidget(l, i, 0)
            gl.addWidget(w, i, 1)
        l = QLabel("Clases (ninguna marcada = todas)")
        l.setStyleSheet("font-weight:400;")
        gl.addWidget(l, 4, 0, 1, 2)
        self.lista_clases = QListWidget()
        self.lista_clases.setFixedHeight(110)
        self.lista_clases.setSelectionMode(QAbstractItemView.NoSelection)
        for k, v in self.nombres.items():
            it = QListWidgetItem(v)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, k)
            self.lista_clases.addItem(it)
        gl.addWidget(self.lista_clases, 5, 0, 1, 2)
        self.ck_track = QCheckBox("Seguimiento (ByteTrack)")
        self.ck_track.setChecked(True)
        self.ck_rt = QCheckBox("Simular tiempo real")
        self.ck_rt.setToolTip("Descarta frames si el modelo es más lento que el video,\n"
                              "como pasaría con una cámara en vivo.")
        for i, w in enumerate((self.ck_track, self.ck_rt)):
            w.setStyleSheet("font-weight:400;")
            gl.addWidget(w, 6 + i, 0, 1, 2)
        lay.addWidget(g)

        self.btn_detectar = QPushButton("▶  Detectar objetos")
        self.btn_detectar.setObjectName("primario")
        self.btn_detectar.setEnabled(False)
        self.btn_detectar.clicked.connect(self.iniciar)
        self.btn_detener = QPushButton("■  Detener")
        self.btn_detener.setObjectName("detener")
        self.btn_detener.setEnabled(False)
        self.btn_detener.clicked.connect(self.detener)
        self.barra = QProgressBar()
        self.barra.setTextVisible(False)
        self.lbl_estado = QLabel("")
        self.lbl_estado.setStyleSheet("color:#6b7280;")
        for w in (self.btn_detectar, self.btn_detener, self.barra, self.lbl_estado):
            lay.addWidget(w)
        lay.addStretch()
        return panel

    # ---------- pestañas
    def _panel_derecho(self):
        self.tabs = QTabWidget()

        # Detección en vivo
        w = QWidget()
        l = QVBoxLayout(w)
        self.lbl_video = QLabel("Selecciona una fuente y pulsa «Detectar objetos»")
        self.lbl_video.setObjectName("video")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setMinimumSize(640, 400)
        self.lbl_video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        l.addWidget(self.lbl_video, 1)
        fila = QHBoxLayout()
        self.k_vivo = [Tarjeta(t) for t in ("FPS", "Latencia", "Objetos en el frame", "Frame del video")]
        for k in self.k_vivo:
            fila.addWidget(k)
        l.addLayout(fila)
        self.tabs.addTab(w, "🎥 Detección")

        # Resumen
        w = QWidget()
        l = QHBoxLayout(w)
        izq = QVBoxLayout()
        grid = QGridLayout()
        nombres = ["FPS de inferencia", "Latencia media", "¿Tiempo real?",
                   "Frames procesados", "Detecciones", "Objetos distintos"]
        self.k_res = {}
        for i, n in enumerate(nombres):
            self.k_res[n] = Tarjeta(n)
            grid.addWidget(self.k_res[n], i // 3, i % 3)
        izq.addLayout(grid)
        t = QLabel("💡 Datos interesantes")
        t.setStyleSheet("font-size:16px; font-weight:600; margin-top:8px;")
        izq.addWidget(t)
        self.txt_hallazgos = QTextBrowser()
        self.txt_hallazgos.setStyleSheet("font-size:13px;")
        izq.addWidget(self.txt_hallazgos, 1)
        l.addLayout(izq, 3)
        der = QVBoxLayout()
        self.lbl_mejor_t = QLabel("Momento con más objetos")
        self.lbl_mejor_t.setStyleSheet("font-weight:600;")
        self.lbl_mejor = QLabel()
        self.lbl_mejor.setObjectName("video")
        self.lbl_mejor.setAlignment(Qt.AlignCenter)
        self.lbl_mejor.setFixedHeight(300)
        self.lbl_mejor.setMinimumWidth(420)
        der.addWidget(self.lbl_mejor_t)
        der.addWidget(self.lbl_mejor)
        bv = QPushButton("▶  Reproducir video anotado")
        bv.clicked.connect(lambda: self.abrir(self.carpeta / "video_anotado.mp4"))
        bc = QPushButton("📁  Abrir carpeta de resultados")
        bc.clicked.connect(lambda: self.abrir(self.carpeta))
        der.addWidget(bv)
        der.addWidget(bc)
        der.addStretch()
        l.addLayout(der, 2)
        self.tabs.addTab(w, "📊 Resumen")

        self.g_rend = Grafica(2, 2, 6)
        self.g_obj = Grafica(2, 2, 6)
        self.g_conf = Grafica(2, 2, 6)
        self.tabs.addTab(self.g_rend, "⚡ Rendimiento")
        self.tabs.addTab(self.g_obj, "🧩 Objetos")
        self.tabs.addTab(self.g_conf, "🎯 Confianza")

        self.tabla = QTableWidget()
        self.tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tabs.addTab(self.tabla, "📁 Datos")

        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, False)
        return self.tabs

    # ---------- acciones
    def abrir_video(self):
        ruta, _ = QFileDialog.getOpenFileName(self, "Elegir video", str(BASE),
                                              "Videos (*.mp4 *.avi *.mov *.mkv *.webm)")
        if ruta:
            self.set_fuente(ruta)

    def set_fuente(self, fuente):
        self.fuente = fuente
        if fuente == "0":
            self.lbl_fuente.setText("📷 Webcam (cámara 0)\nPulsa «Detener» para terminar.")
        else:
            cap = cv2.VideoCapture(fuente)
            n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, prim = cap.read()
            cap.release()
            if not ok:
                QMessageBox.warning(self, "Video", "No se pudo leer el video.")
                return
            self.lbl_fuente.setText(f"{Path(fuente).name}\n{w}×{h} · {fps:.0f} FPS · "
                                    f"{n} frames · {n / fps:.1f} s")
            self.lbl_video.setPixmap(a_pixmap(prim, self.lbl_video.width(), self.lbl_video.height()))
        self.btn_detectar.setEnabled(True)

    def iniciar(self):
        clases = [self.lista_clases.item(i).data(Qt.UserRole)
                  for i in range(self.lista_clases.count())
                  if self.lista_clases.item(i).checkState() == Qt.Checked] or None
        self.carpeta = BASE / "resultados" / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.carpeta.mkdir(parents=True, exist_ok=True)
        params = dict(source=self.fuente, modelo=self.cb_modelo.currentText(),
                      conf=self.sp_conf.value(), iou=self.sp_iou.value(),
                      imgsz=int(self.cb_imgsz.currentText()), classes=clases,
                      track=self.ck_track.isChecked(), realtime=self.ck_rt.isChecked(),
                      save_path=self.carpeta / "video_anotado.mp4")
        self.conf_usada = params["conf"]
        self.trabajador = Trabajador(params)
        self.trabajador.frame.connect(self.al_frame)
        self.trabajador.terminado.connect(self.al_terminar)
        self.trabajador.error.connect(self.al_error)
        self.btn_detectar.setEnabled(False)
        self.btn_detener.setEnabled(True)
        self.barra.setRange(0, 0 if self.fuente == "0" else 1000)
        self.barra.setValue(0)
        self.lbl_estado.setText("Cargando modelo…")
        self.tabs.setCurrentIndex(0)
        self.trabajador.start()

    def detener(self):
        if self.trabajador:
            self.trabajador.parar = True
            self.lbl_estado.setText("Deteniendo…")

    def al_frame(self, frame, fila, prog):
        self.lbl_video.setPixmap(a_pixmap(frame, self.lbl_video.width(), self.lbl_video.height()))
        self.k_vivo[0].set(f"{fila['fps']:.1f}")
        self.k_vivo[1].set(f"{fila['latencia_ms']:.0f} ms")
        self.k_vivo[2].set(fila["n_detecciones"], fila["clases"].replace("|", ", "))
        self.k_vivo[3].set(fila["frame_video"], f"{fila['tiempo_video_s']:.1f} s")
        if self.fuente != "0":
            self.barra.setValue(int(prog * 1000))
            self.lbl_estado.setText(f"Procesando… {prog * 100:.0f}%")
        else:
            self.lbl_estado.setText(f"Webcam · frame {fila['frame']}")

    def al_error(self, msg):
        self._fin()
        QMessageBox.critical(self, "Error", f"Error al procesar:\n{msg}")

    def _fin(self):
        self.btn_detectar.setEnabled(True)
        self.btn_detener.setEnabled(False)
        self.barra.setRange(0, 1000)
        self.barra.setValue(1000)

    def al_terminar(self, resultado):
        self._fin()
        rows, res, ex = resultado
        self.resultado = resultado
        guardar_resultados(rows, res, self.carpeta)
        self.mostrar_resumen(res, ex)
        self.graficar_rendimiento(rows, res)
        self.graficar_objetos(rows, res, ex)
        self.graficar_confianza(rows, ex)
        self.llenar_tabla(rows)
        for g, n in ((self.g_rend, "rendimiento"), (self.g_obj, "objetos"), (self.g_conf, "confianza")):
            g.fig.savefig(self.carpeta / f"grafica_{n}.png", dpi=130)
        for i in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(i, True)
        self.tabs.setCurrentIndex(1)
        self.lbl_estado.setText(f"✅ Listo. Resultados en:\nresultados/{self.carpeta.name}")

    # ---------- resultados
    def mostrar_resumen(self, res, ex):
        k = self.k_res
        dif = res["fps_inferencia_medio"] - res["fps_fuente"]
        k["FPS de inferencia"].set(f"{res['fps_inferencia_medio']:.1f}",
                                   f"factor RT {res['factor_tiempo_real']:.2f} · video a {res['fps_fuente']:.0f} FPS")
        k["Latencia media"].set(f"{res['latencia_ms']['media']:.0f} ms",
                                f"p95 {res['latencia_ms']['p95']:.0f} ms · solo red {res['desglose_ms']['inferencia']:.0f} ms")
        k["¿Tiempo real?"].set("Sí ✅" if res["tiempo_real"] else "No ❌",
                               f"límite {1000 / res['fps_fuente']:.0f} ms por frame",
                               VERDE if res["tiempo_real"] else ROJO)
        k["Frames procesados"].set(res["frames_procesados"],
                                   f"{res['frames_descartados']} descartados ({res['pct_descartados']}%)" if res["frames_descartados"] else "ninguno descartado")
        k["Detecciones"].set(res["detecciones_totales"], f"{res['detecciones_por_frame_media']} por frame")
        unicos = sum(res["objetos_unicos_por_clase"].values()) if res["objetos_unicos_por_clase"] else "—"
        k["Objetos distintos"].set(unicos, f"confianza media {res['conf_media']:.2f}")

        frases = [re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", f) for f in generar_hallazgos(res, None, ex)]
        self.txt_hallazgos.setHtml("<ul style='margin-left:-20px'>" +
                                   "".join(f"<li style='margin-bottom:8px'>{f}</li>" for f in frases) + "</ul>")
        if ex["mejor_frame"] is not None:
            self.lbl_mejor_t.setText(f"Momento con más objetos — segundo "
                                     f"{ex['mejor_frame_idx'] / res['fps_fuente']:.1f} "
                                     f"({res['max_detecciones_en_un_frame']} detecciones)")
            self.lbl_mejor.setPixmap(a_pixmap(ex["mejor_frame"], 420, 296))

    def graficar_rendimiento(self, rows, res):
        fv = np.array([r["frame_video"] for r in rows])
        lat = np.array([r["latencia_ms"] for r in rows])
        fps = np.array([r["fps"] for r in rows])
        ax = self.g_rend.ejes()
        lim = 1000 / res["fps_fuente"]
        ax[0].plot(fv, lat, color=ACENTO, lw=1)
        ax[0].axhline(lat.mean(), ls="--", color=ROJO, label=f"media {lat.mean():.0f} ms")
        ax[0].axhline(lim, ls=":", color=VERDE, label=f"límite tiempo real {lim:.0f} ms")
        ax[0].set(title="Latencia por frame", xlabel="frame del video", ylabel="ms")
        ax[0].legend(fontsize=8)
        ax[1].plot(fv, pd.Series(fps).rolling(15, min_periods=1).mean(), color=ACENTO)
        ax[1].axhline(res["fps_fuente"], ls=":", color=VERDE, label="FPS del video")
        ax[1].set(title="FPS (media móvil)", xlabel="frame del video", ylabel="FPS")
        ax[1].legend(fontsize=8)
        ax[2].hist(lat, bins=30, color=ACENTO)
        ax[2].set(title="Distribución de la latencia", xlabel="ms", ylabel="frames")
        lt = res["latencia_ms"]
        ax[3].bar(["media", "mediana", "p95", "máx"], [lt["media"], lt["mediana"], lt["p95"], lt["max"]],
                  color=[ACENTO, ACENTO, NARANJA, ROJO])
        ax[3].axhline(lim, ls=":", color=VERDE)
        ax[3].set(title="Resumen de latencia", ylabel="ms")
        self.g_rend.draw()

    def graficar_objetos(self, rows, res, ex):
        ax = self.g_obj.ejes()
        fps_src = res["fps_fuente"]
        clases = list(res["detecciones_por_clase"].keys())[:10]
        det = [res["detecciones_por_clase"][c] for c in clases]
        y = np.arange(len(clases))
        if res["objetos_unicos_por_clase"]:
            uni = [res["objetos_unicos_por_clase"].get(c, 0) for c in clases]
            ax[0].barh(y - 0.2, det, 0.4, color=ACENTO, label="detecciones")
            ax[0].barh(y + 0.2, uni, 0.4, color=NARANJA, label="objetos distintos")
            ax[0].legend(fontsize=8)
        else:
            ax[0].barh(y, det, color=ACENTO)
        ax[0].set_yticks(y, clases)
        ax[0].invert_yaxis()
        ax[0].set(title="Detecciones por clase")
        t = [r["tiempo_video_s"] for r in rows]
        n = [r["n_detecciones"] for r in rows]
        ax[1].fill_between(t, n, step="post", color=ACENTO, alpha=0.35)
        ax[1].step(t, n, where="post", color=ACENTO)
        ax[1].set(title="Objetos por frame en el tiempo", xlabel="segundo", ylabel="nº objetos")
        # línea de tiempo de apariciones, ocupa toda la fila inferior
        ax[2].remove()
        ax[3].remove()
        axb = self.g_obj.fig.add_subplot(2, 1, 2)
        axb.spines[["top", "right"]].set_visible(False)
        cmap = matplotlib.colormaps["tab10"]
        for i, c in enumerate(clases):
            seg = np.array(ex["apariciones"].get(c, [])) / fps_src
            axb.scatter(seg, np.full(len(seg), i), marker="|", s=120, color=cmap(i % 10))
        axb.set_yticks(range(len(clases)), clases)
        axb.invert_yaxis()
        axb.set(title="¿Cuándo aparece cada clase?  (ráfagas cortas = parpadeo del modelo)",
                xlabel="segundo")
        axb.grid(axis="x", alpha=0.25)
        self.g_obj.draw()

    def graficar_confianza(self, rows, ex):
        ax = self.g_conf.ejes()
        if ex["confianzas"]:
            ax[0].hist(ex["confianzas"], bins=25, color=ACENTO)
            ax[0].axvline(self.conf_usada, ls=":", color=ROJO, label="umbral")
            ax[0].legend(fontsize=8)
        ax[0].set(title="Distribución de la confianza", xlabel="confianza", ylabel="detecciones")
        cpc = ex["conf_por_clase"]
        if cpc:
            orden = sorted(cpc, key=lambda c: -len(cpc[c]))[:8]
            ax[1].boxplot([cpc[c] for c in orden], tick_labels=orden, patch_artist=True,
                          boxprops=dict(facecolor="#bfdbfe", color=ACENTO),
                          medianprops=dict(color=ROJO))
            ax[1].tick_params(axis="x", rotation=30)
        ax[1].set(title="Confianza por clase", ylabel="confianza")
        ax[2].remove()
        ax[3].remove()
        axb = self.g_conf.fig.add_subplot(2, 1, 2)
        axb.spines[["top", "right"]].set_visible(False)
        d = [(r["tiempo_video_s"], r["conf_media"]) for r in rows if r["n_detecciones"]]
        if d:
            tt, cc = zip(*d)
            axb.plot(tt, cc, color=ACENTO, lw=1, marker=".", ms=3)
        axb.set(title="Confianza media por frame a lo largo del video", xlabel="segundo",
                ylabel="confianza", ylim=(0, 1))
        axb.grid(alpha=0.25)
        self.g_conf.draw()

    def llenar_tabla(self, rows):
        cols = list(rows[0].keys())
        self.tabla.setColumnCount(len(cols))
        self.tabla.setHorizontalHeaderLabels(cols)
        self.tabla.setRowCount(len(rows))
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                self.tabla.setItem(i, j, QTableWidgetItem(str(r[c])))
        self.tabla.resizeColumnsToContents()

    def abrir(self, ruta):
        if ruta and Path(ruta).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(ruta)))

    def closeEvent(self, e):
        if self.trabajador and self.trabajador.isRunning():
            self.trabajador.parar = True
            self.trabajador.wait(3000)
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(ESTILO)
    app.setFont(QFont("Segoe UI", 10))
    v = Ventana()
    v.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
