"""
Interfaz web — Detección de objetos con YOLO en video
Ejecutar:  streamlit run app.py
"""
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from yolo_video import cargar_modelo, generar_hallazgos, guardar_resultados, procesar_video

st.set_page_config(page_title="YOLO en video", page_icon="🎯", layout="wide")

BASE = Path(__file__).parent
TMP = Path(tempfile.gettempdir()) / "yolo_app"
TMP.mkdir(exist_ok=True)
ACENTO = "#2563eb"


# ---------------------------------------------------------------- utilidades
@st.cache_resource(show_spinner="Cargando modelo YOLO…")
def modelo_cache(nombre):
    return cargar_modelo(nombre)


def a_h264(entrada, salida):
    """El navegador no reproduce mp4v; se convierte a H.264 con ffmpeg."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = "ffmpeg"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(entrada), "-vcodec", "libx264",
           "-pix_fmt", "yuv420p", "-preset", "veryfast", "-movflags", "+faststart", str(salida)]
    try:
        subprocess.run(cmd, check=True)
        return salida
    except Exception:
        return entrada


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    modelo = st.selectbox("Modelo YOLO", ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"],
                          help="n = nano (más rápido), s = small (más preciso)")
    conf = st.slider("Umbral de confianza", 0.05, 0.95, 0.35, 0.05)
    iou = st.slider("Umbral IoU (NMS)", 0.1, 0.9, 0.5, 0.05)
    imgsz = st.select_slider("Tamaño de entrada", [320, 416, 512, 640, 800], value=640,
                             help="Menor = más rápido, menos preciso")
    nombres = modelo_cache(modelo).names
    sel = st.multiselect("Filtrar clases (vacío = todas)", list(nombres.values()))
    classes = [k for k, v in nombres.items() if v in sel] or None
    track = st.toggle("Seguimiento de objetos (ByteTrack)", True)
    realtime = st.toggle("Simular tiempo real", False,
                         help="Descarta frames si el modelo es más lento que el video, como una cámara en vivo")
    vista_cada = st.slider("Vista en vivo: actualizar cada N frames", 1, 10, 2)
    max_frames = st.number_input("Máx. frames (0 = todo)", 0, 100000, 0, 50)

# ---------------------------------------------------------------- encabezado
st.title("🎯 Detección de objetos con YOLO en video")
st.caption("Sube un video, YOLO detecta y sigue los objetos frame a frame, y obtienes el video "
           "anotado con métricas de rendimiento y calidad.")

c1, c2 = st.columns([3, 1])
with c1:
    subido = st.file_uploader("Sube un video", type=["mp4", "avi", "mov", "mkv", "webm"])
with c2:
    st.write("")
    st.write("")
    ejemplo = BASE / "video_prueba.mp4"
    usar_ejemplo = st.button("Usar video de ejemplo", disabled=not ejemplo.exists(),
                             use_container_width=True)

if subido is not None:
    ruta = TMP / f"entrada_{subido.name}"
    ruta.write_bytes(subido.getbuffer())
    st.session_state["video"] = str(ruta)
elif usar_ejemplo:
    st.session_state["video"] = str(ejemplo)

video = st.session_state.get("video")
if not video:
    st.info("👆 Sube un video (o usa el de ejemplo) para empezar.")
    st.stop()

cap = cv2.VideoCapture(video)
info = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30,
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.release()
st.write(f"**Video:** `{Path(video).name}` · {info[2]}×{info[3]} · {info[1]:.0f} FPS · "
         f"{info[0]} frames · {info[0] / info[1]:.1f} s")

if st.button("▶️ Detectar objetos", type="primary", use_container_width=True):
    st.session_state.pop("resultado", None)
    col_v, col_m = st.columns([3, 1])
    vista = col_v.empty()
    with col_m:
        m_fps, m_lat, m_det, m_drop = st.empty(), st.empty(), st.empty(), st.empty()
    barra = st.progress(0.0, text="Procesando…")

    def on_frame(frame, fila, prog):
        if fila["frame"] % vista_cada == 0:
            vista.image(bgr2rgb(frame), channels="RGB", use_container_width=True)
            m_fps.metric("FPS", f"{fila['fps']:.1f}")
            m_lat.metric("Latencia", f"{fila['latencia_ms']:.0f} ms")
            m_det.metric("Objetos en el frame", fila["n_detecciones"])
            m_drop.metric("Frame del video", fila["frame_video"])
        barra.progress(prog, text=f"Procesando… {prog * 100:.0f}%")

    salida_raw = TMP / "anotado_raw.mp4"
    try:
        rows, res, ex = procesar_video(video, modelo, conf, iou, imgsz, classes, track,
                                       realtime, save_path=salida_raw,
                                       max_frames=int(max_frames), on_frame=on_frame)
    except Exception as e:
        st.error(f"Error al procesar: {e}")
        st.stop()
    barra.progress(1.0, text="Convirtiendo video para el navegador…")
    salida = a_h264(salida_raw, TMP / "anotado.mp4")
    guardar_resultados(rows, res, TMP / "resultados")
    barra.empty()
    vista.empty()
    st.session_state["resultado"] = {"rows": rows, "res": res, "ex": ex, "video": str(salida)}
    st.rerun()

# ---------------------------------------------------------------- resultados
R = st.session_state.get("resultado")
if not R:
    st.stop()
rows, res, ex = R["rows"], R["res"], R["ex"]
df = pd.DataFrame(rows)

st.divider()
st.subheader("📊 Resumen")
k = st.columns(6)
k[0].metric("FPS de inferencia", f"{res['fps_inferencia_medio']:.1f}",
            f"{res['fps_inferencia_medio'] - res['fps_fuente']:+.1f} vs video")
k[1].metric("Latencia media", f"{res['latencia_ms']['media']:.0f} ms",
            f"p95 {res['latencia_ms']['p95']:.0f} ms", delta_color="off")
k[2].metric("¿Tiempo real?", "Sí ✅" if res["tiempo_real"] else "No ❌")
k[3].metric("Frames procesados", res["frames_procesados"],
            f"-{res['frames_descartados']} descartados" if res["frames_descartados"] else None)
k[4].metric("Detecciones", res["detecciones_totales"],
            f"{res['detecciones_por_frame_media']} por frame", delta_color="off")
unicos = sum(res["objetos_unicos_por_clase"].values()) if res["objetos_unicos_por_clase"] else "—"
k[5].metric("Objetos distintos", unicos, f"conf. media {res['conf_media']:.2f}", delta_color="off")

with st.container(border=True):
    st.markdown("#### 💡 Datos interesantes")
    for frase in generar_hallazgos(res, df, ex):
        st.markdown(f"- {frase}")

t1, t2, t3, t4, t5 = st.tabs(["🎬 Video resultado", "⚡ Rendimiento", "🧩 Objetos",
                              "🎯 Confianza", "📁 Datos"])

with t1:
    a, b = st.columns([3, 2])
    a.video(R["video"])
    if ex["mejor_frame"] is not None:
        b.markdown(f"**Momento con más objetos** — segundo "
                   f"{ex['mejor_frame_idx'] / res['fps_fuente']:.1f} "
                   f"({res['max_detecciones_en_un_frame']} detecciones)")
        b.image(bgr2rgb(ex["mejor_frame"]), use_container_width=True)

with t2:
    presupuesto = 1000 / res["fps_fuente"]
    fig = go.Figure()
    fig.add_scatter(x=df.frame_video, y=df.latencia_ms, mode="lines", name="latencia",
                    line=dict(color=ACENTO, width=1.5))
    fig.add_hline(y=res["latencia_ms"]["media"], line_dash="dash", line_color="#dc2626",
                  annotation_text=f"media {res['latencia_ms']['media']:.0f} ms")
    fig.add_hline(y=presupuesto, line_dash="dot", line_color="#16a34a",
                  annotation_text=f"límite tiempo real {presupuesto:.0f} ms",
                  annotation_position="bottom right")
    fig.update_layout(title="Latencia por frame (debajo de la línea verde = tiempo real)",
                      xaxis_title="frame del video", yaxis_title="ms", height=360,
                      margin=dict(t=50, b=40), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    fps_s = df.fps.rolling(15, min_periods=1).mean()
    fig = go.Figure(go.Scatter(x=df.frame_video, y=fps_s, line=dict(color=ACENTO)))
    fig.add_hline(y=res["fps_fuente"], line_dash="dot", line_color="#16a34a",
                  annotation_text="FPS del video")
    fig.update_layout(title="FPS (media móvil de 15 frames)", height=320,
                      xaxis_title="frame del video", yaxis_title="FPS", margin=dict(t=50, b=40))
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(df, x="latencia_ms", nbins=30, title="Distribución de la latencia",
                       color_discrete_sequence=[ACENTO])
    fig.update_layout(height=320, xaxis_title="ms", yaxis_title="frames", margin=dict(t=50, b=40))
    c2.plotly_chart(fig, use_container_width=True)

with t3:
    c1, c2 = st.columns(2)
    cl = pd.DataFrame({"clase": list(res["detecciones_por_clase"].keys()),
                       "detecciones": list(res["detecciones_por_clase"].values())})
    if res["objetos_unicos_por_clase"]:
        cl["objetos distintos"] = cl.clase.map(res["objetos_unicos_por_clase"]).fillna(0)
        fig = px.bar(cl, y="clase", x=["detecciones", "objetos distintos"], orientation="h",
                     barmode="group", title="Detecciones vs objetos distintos por clase",
                     color_discrete_sequence=[ACENTO, "#f59e0b"])
    else:
        fig = px.bar(cl, y="clase", x="detecciones", orientation="h",
                     title="Detecciones por clase", color_discrete_sequence=[ACENTO])
    fig.update_layout(height=380, yaxis=dict(autorange="reversed"), legend_title="",
                      xaxis_title="", margin=dict(t=50, b=40))
    c1.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(go.Scatter(x=df.tiempo_video_s, y=df.n_detecciones, fill="tozeroy",
                               line=dict(color=ACENTO, shape="hv")))
    fig.update_layout(title="Objetos detectados a lo largo del video", height=380,
                      xaxis_title="segundo", yaxis_title="nº objetos", margin=dict(t=50, b=40))
    c2.plotly_chart(fig, use_container_width=True)

    if ex["apariciones"]:
        ap = pd.DataFrame([(c, f / res["fps_fuente"]) for c, fs in ex["apariciones"].items() for f in fs],
                          columns=["clase", "segundo"])
        orden = list(res["detecciones_por_clase"].keys())
        fig = px.scatter(ap, x="segundo", y="clase", color="clase", category_orders={"clase": orden},
                         title="¿Cuándo aparece cada clase?")
        fig.update_traces(marker=dict(symbol="line-ns-open", size=14, line_width=2))
        fig.update_layout(height=120 + 40 * len(orden), showlegend=False, margin=dict(t=50, b=40))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Si una clase aparece y desaparece en ráfagas cortas, el modelo está "
                   "'parpadeando' — algo que en imágenes sueltas no se nota.")

with t4:
    c1, c2 = st.columns(2)
    if ex["confianzas"]:
        fig = px.histogram(x=ex["confianzas"], nbins=25, title="Distribución de la confianza",
                           color_discrete_sequence=[ACENTO])
        fig.add_vline(x=conf, line_dash="dot", line_color="#dc2626", annotation_text="umbral")
        fig.update_layout(height=360, xaxis_title="confianza", yaxis_title="detecciones",
                          margin=dict(t=50, b=40))
        c1.plotly_chart(fig, use_container_width=True)
        cc = pd.DataFrame([(k, v) for k, vs in ex["conf_por_clase"].items() for v in vs],
                          columns=["clase", "confianza"])
        fig = px.box(cc, x="clase", y="confianza", title="Confianza por clase",
                     color_discrete_sequence=[ACENTO])
        fig.update_layout(height=360, margin=dict(t=50, b=40))
        c2.plotly_chart(fig, use_container_width=True)
    det = df[df.n_detecciones > 0]
    fig = go.Figure(go.Scatter(x=det.tiempo_video_s, y=det.conf_media, mode="lines+markers",
                               marker=dict(size=4), line=dict(color=ACENTO, width=1)))
    fig.update_layout(title="Confianza media por frame a lo largo del video", height=320,
                      xaxis_title="segundo", yaxis_title="confianza", yaxis_range=[0, 1],
                      margin=dict(t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)

with t5:
    st.dataframe(df, use_container_width=True, height=320)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Métricas por frame (CSV)", df.to_csv(index=False).encode(),
                       "metricas_por_frame.csv", "text/csv", use_container_width=True)
    d2.download_button("⬇️ Resumen (JSON)", json.dumps(res, indent=2, ensure_ascii=False).encode(),
                       "resumen_metricas.json", "application/json", use_container_width=True)
    d3.download_button("⬇️ Video anotado (MP4)", Path(R["video"]).read_bytes(),
                       "video_anotado.mp4", "video/mp4", use_container_width=True)
    with st.expander("Resumen completo (JSON)"):
        st.json(res)
