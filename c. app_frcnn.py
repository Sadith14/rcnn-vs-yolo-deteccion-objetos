import gc, tempfile, time, unicodedata, zlib
from collections import Counter
from pathlib import Path
import json

import cv2
import pandas as pd
import streamlit as st
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights)
from torchvision.transforms.functional import to_tensor

st.set_page_config(page_title="Faster R-CNN · Detección en video", page_icon="🎯", layout="wide")

# ------------------------- Configuración -------------------------
BASE = Path(__file__).resolve().parent
CARPETA = BASE / "videos"
SALIDA = BASE / "resultados"
COLA = BASE / "frames_anotar"          # frames capturados para anotar
RUTA_GT = BASE / "ground_truth.json"
for d in (CARPETA, SALIDA, COLA):
    d.mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def cargar_gt():
    """Lee ground_truth.json en cada uso para ver siempre las anotaciones más recientes."""
    if RUTA_GT.exists():
        with open(RUTA_GT, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def guardar_gt(data):
    with open(RUTA_GT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


MODELOS = {
    "fasterrcnn_resnet50_fpn":               (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights),
    "fasterrcnn_mobilenet_v3_large_fpn":     (fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights),
    "fasterrcnn_resnet50_fpn_v2":            (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights),
    "fasterrcnn_mobilenet_v3_large_320_fpn": (fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights),
    "maskrcnn_resnet50_fpn_v2":              (maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights),
}

ES = {
    "person": "persona", "bicycle": "bicicleta", "car": "auto", "motorcycle": "moto",
    "airplane": "avión", "bus": "bus", "train": "tren", "truck": "camión", "boat": "bote",
    "traffic light": "semáforo", "fire hydrant": "hidrante", "stop sign": "señal de pare",
    "parking meter": "parquímetro", "bench": "banco", "bird": "ave", "cat": "gato",
    "dog": "perro", "horse": "caballo", "sheep": "oveja", "cow": "vaca", "elephant": "elefante",
    "bear": "oso", "zebra": "cebra", "giraffe": "jirafa", "backpack": "mochila",
    "umbrella": "paraguas", "handbag": "cartera", "tie": "corbata", "suitcase": "maleta",
    "frisbee": "frisbee", "skis": "esquís", "snowboard": "snowboard", "sports ball": "pelota",
    "kite": "cometa", "baseball bat": "bate", "skateboard": "patineta", "surfboard": "tabla de surf",
    "bottle": "botella", "wine glass": "copa", "cup": "taza", "fork": "tenedor", "knife": "cuchillo",
    "spoon": "cuchara", "bowl": "bol", "banana": "plátano", "apple": "manzana", "sandwich": "sándwich",
    "orange": "naranja", "broccoli": "brócoli", "carrot": "zanahoria", "hot dog": "hot dog",
    "pizza": "pizza", "donut": "dona", "cake": "pastel", "chair": "silla", "couch": "sofá",
    "potted plant": "planta", "bed": "cama", "dining table": "mesa", "toilet": "inodoro",
    "tv": "televisor", "laptop": "laptop", "mouse": "mouse", "remote": "control remoto",
    "keyboard": "teclado", "cell phone": "celular", "microwave": "microondas", "oven": "horno",
    "toaster": "tostadora", "sink": "lavadero", "refrigerator": "refrigerador", "book": "libro",
    "clock": "reloj", "vase": "florero", "scissors": "tijeras", "teddy bear": "peluche",
    "hair drier": "secador", "toothbrush": "cepillo de dientes",
}
PALETA = [(0, 200, 0), (255, 140, 0), (0, 160, 255), (200, 0, 200), (0, 200, 200),
          (60, 60, 255), (255, 80, 120), (160, 200, 0), (255, 200, 0), (140, 90, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLS_BOX = ["x1", "y1", "x2", "y2", "clase"]


def es(nombre):
    return ES.get(nombre, nombre)


def ascii_(txt):  # OpenCV no dibuja tildes/ñ
    return unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()


def color_de(nombre):
    return PALETA[zlib.crc32(nombre.encode()) % len(PALETA)]


@st.cache_resource(show_spinner="Cargando modelo (la primera vez descarga los pesos)…")
def cargar_modelo(nombre):
    fn, W = MODELOS[nombre]
    pesos = W.DEFAULT
    return fn(weights=pesos).to(device).eval(), pesos.meta["categories"]


def dibujar(frame, dets, estado):
    h, w = frame.shape[:2]
    esc = max(1.0, w / 1000)
    gr = max(2, int(2 * esc))
    fs = 0.5 * esc
    for (x1, y1, x2, y2), nom, s in dets:
        c = color_de(nom)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, gr)
        txt = f"{ascii_(es(nom))} {s * 100:.0f}%"
        (tw, th), bl = cv2.getTextSize(txt, FONT, fs, max(1, gr - 1))
        y0 = max(y1, th + bl + 4)
        cv2.rectangle(frame, (x1, y0 - th - bl - 4), (x1 + tw + 6, y0), c, -1)
        cv2.putText(frame, txt, (x1 + 3, y0 - bl - 2), FONT, fs, (255, 255, 255),
                    max(1, gr - 1), cv2.LINE_AA)
    col = (0, 160, 0) if estado == "PROCESADO" else (0, 120, 255)
    (tw, th), bl = cv2.getTextSize(estado, FONT, 0.7 * esc, gr)
    cv2.rectangle(frame, (10, 10), (30 + tw, 30 + th + bl), col, -1)
    cv2.putText(frame, estado, (20, 20 + th), FONT, 0.7 * esc, (255, 255, 255), gr, cv2.LINE_AA)
    return frame


# ------------------------- Evaluación de calidad -------------------------
def contar_aciertos(boxes, scores, pred_labels, target, iou=0.5):
    gt_boxes = target["boxes"]
    gt_labels = target["labels"]

    tp = fp = 0
    matched = set()

    # Se evalúan primero las detecciones con mayor confianza
    order = torch.argsort(scores, descending=True)

    for idx in order:
        box = boxes[idx]
        label = pred_labels[idx]

        candidatos = torch.where(gt_labels == label)[0]
        if len(candidatos) == 0:
            fp += 1
            continue

        ious = box_iou(box.unsqueeze(0), gt_boxes[candidatos])[0]

        for j in torch.argsort(ious, descending=True):
            gt_idx = int(candidatos[j])
            if ious[j] >= iou and gt_idx not in matched:
                matched.add(gt_idx)
                tp += 1
                break
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def evaluar_modelo(model, video_path, gt_video, umbral=0.6,
                   modo_prueba=False, ver_gt=False):
    """Evalúa solo los frames anotados. Devuelve None si no hay nada que evaluar."""
    if not gt_video and not modo_prueba:
        return None

    # En modo prueba se evalúan frames espaciados (cada 30) sin necesidad de JSON
    if modo_prueba:
        indices = None
        ultimo = float("inf")
    else:
        indices = {int(k) for k in gt_video}
        ultimo = max(indices)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    )

    tp = fp = fn = 0
    evaluados = 0
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx > ultimo:
            break

        if modo_prueba:
            if frame_idx % 30 != 0:
                continue
        elif frame_idx not in indices:
            continue

        img = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
        with torch.no_grad():
            pred = model([img])[0]

        prediction = {
            "boxes": pred["boxes"].cpu(),
            "scores": pred["scores"].cpu(),
            "labels": pred["labels"].cpu(),
        }

        if modo_prueba:
            # Ground truth artificial: detecciones del propio modelo con alta confianza
            fuerte = prediction["scores"] >= 0.8
            target = {"boxes": prediction["boxes"][fuerte],
                      "labels": prediction["labels"][fuerte]}
        else:
            gt = gt_video[str(frame_idx)]
            target = {
                "boxes": torch.tensor(gt["boxes"], dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(gt["labels"], dtype=torch.int64),
            }

        # mAP: predicciones sin filtrar por umbral
        metric.update([prediction], [target])

        # Precision y Recall: solo detecciones sobre el umbral elegido
        keep = prediction["scores"] >= umbral
        t, f, m = contar_aciertos(
            prediction["boxes"][keep], prediction["scores"][keep],
            prediction["labels"][keep], target, iou=0.5)
        tp += t
        fp += f
        fn += m
        evaluados += 1

        if ver_gt:
            vis = frame.copy()
            for b in target["boxes"].int().tolist():
                cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)      # GT
            for b, s in zip(prediction["boxes"].int().tolist(), prediction["scores"].tolist()):
                if s >= umbral:
                    cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # Predicción
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     caption=f"Frame {frame_idx}: verde = GT, rojo = predicción (umbral {umbral})",
                     use_container_width=True)

    cap.release()

    if evaluados == 0:
        return None

    result = metric.compute()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "Precision": precision,
        "Recall": recall,
        "mAP@0.5": result["map_50"].item(),
        "mAP@0.5:0.95": result["map"].item(),
        "Frames evaluados": evaluados,
        "TP": tp, "FP": fp, "FN": fn,
    }


# ------------------------- Procesamiento -------------------------
def procesar(ruta, nombre_video, nombre_modelo, umbral, simular_rt, guardar, paso_ui,
             modo_prueba, ver_gt, capturar, cada_n):
    model, labels = cargar_modelo(nombre_modelo)

    def detectar(frame):
        img = to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            p = model([img])[0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        dets = [(list(map(int, b)), labels[l], float(s))
                for b, l, s in zip(p["boxes"], p["labels"], p["scores"]) if s >= umbral]
        return dets, dt

    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        st.error(f"No se pudo abrir el video: {ruta}")
        return
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # ---- Diseño de la pantalla en vivo ----
    st.caption(f"**{nombre_video}** · {w}x{h} · {fps_in:.1f} FPS · {total_frames} frames · "
               f"modelo **{nombre_modelo}**")
    barra = st.progress(0.0)
    c_video, c_met = st.columns([3, 2])
    ph_video = c_video.empty()
    with c_met:
        f1 = st.columns(2); f2 = st.columns(2); f3 = st.columns(2)
        m_lat, m_media = f1[0].empty(), f1[1].empty()
        m_fps, m_rt = f2[0].empty(), f2[1].empty()
        m_proc, m_desc = f3[0].empty(), f3[1].empty()
        st.markdown("**🔎 Objetos detectados en este frame**")
        ph_tabla = st.empty()
    g1, g2 = st.columns(2)
    g1.markdown("**⏱ Latencia por frame procesado (ms)**")
    ph_lat = g1.empty()
    g2.markdown("**📊 Objetos acumulados por clase**")
    ph_clases = g2.empty()

    # ---- Calentamiento ----
    ok, f0 = cap.read()
    if not ok:
        st.error("El video no tiene frames legibles.")
        return
    with st.spinner("Calentando el modelo…"):
        for _ in range(3):
            detectar(f0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out = None
    if guardar:
        ruta_out = SALIDA / f"salida_{nombre_video}_{nombre_modelo}.mp4"
        out = cv2.VideoWriter(str(ruta_out), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    # Carpeta de frames para anotar (se limpia la cola anterior de este video)
    capturados = 0
    if capturar:
        dir_cola = COLA / nombre_video
        dir_cola.mkdir(exist_ok=True)

    libre_en, ultimas = 0.0, []
    procesados, descartados, latencias, n_dets, i = 0, 0, [], [], 0
    conteo = Counter()
    dt = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Se guarda el frame limpio ANTES de dibujar (dibujar modifica el frame).
        # El índice i coincide con el índice que usa evaluar_modelo.
        if capturar and i % cada_n == 0:
            cv2.imwrite(str(dir_cola / f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            capturados += 1

        if (not simular_rt) or (i / fps_in >= libre_en):
            ultimas, dt = detectar(frame)
            libre_en = i / fps_in + dt
            procesados += 1
            latencias.append(dt)
            n_dets.append(len(ultimas))
            conteo.update(es(n) for _, n, _ in ultimas)
            estado = "PROCESADO"
        else:
            descartados += 1
            estado = "DESCARTADO"

        marcado = dibujar(frame, ultimas, estado)
        if out is not None:
            out.write(marcado)

        if i % paso_ui == 0:
            lat_media = sum(latencias) / len(latencias)
            if simular_rt:
                fps_ef = procesados / ((i + 1) / fps_in)
            else:
                fps_ef = procesados / sum(latencias)
            factor = (1 / lat_media) / fps_in

            vista = marcado
            if w > 960:
                vista = cv2.resize(marcado, (960, int(h * 960 / w)))
            ph_video.image(cv2.cvtColor(vista, cv2.COLOR_BGR2RGB), use_container_width=True)
            barra.progress(min((i + 1) / total_frames, 1.0))

            m_lat.metric("Latencia actual", f"{dt * 1000:.0f} ms", border=True)
            m_media.metric("Latencia media", f"{lat_media * 1000:.0f} ms", border=True)
            m_fps.metric("FPS de detección", f"{fps_ef:.1f}", border=True)
            m_rt.metric("Factor tiempo real", f"{factor:.2f}x",
                        "✅ tiempo real" if factor >= 1 else "⚠️ no alcanza", border=True)
            m_proc.metric("Procesados", f"{procesados}", border=True)
            m_desc.metric("Descartados", f"{descartados}", border=True)

            if ultimas:
                dfo = pd.DataFrame({"Objeto": [es(n) for _, n, _ in ultimas],
                                    "Confianza": [s * 100 for _, _, s in ultimas]})
                dfo = dfo.sort_values("Confianza", ascending=False).head(12)
                ph_tabla.dataframe(
                    dfo, hide_index=True, use_container_width=True,
                    column_config={"Confianza": st.column_config.ProgressColumn(
                        "Confianza", min_value=0, max_value=100, format="%.0f%%")})
            else:
                ph_tabla.info("Sin detecciones sobre el umbral.")

            ph_lat.line_chart(pd.Series([x * 1000 for x in latencias], name="ms"), height=220)
            if conteo:
                ph_clases.bar_chart(pd.Series(conteo).sort_values(ascending=False).head(10),
                                    height=220)
        i += 1

    cap.release()
    if out is not None:
        out.release()
    barra.progress(1.0)

    if capturar:
        st.info(f"✍️ Se guardaron {capturados} frames para anotar. "
                f"Ve a la pestaña **Anotar** para crear el ground truth.")

    # ---------------- Evaluación de calidad ----------------
    st.subheader("🎯 Evaluación de calidad")

    gt_video = cargar_gt().get(nombre_video, {})
    if modo_prueba:
        st.info("Modo prueba de cordura: el ground truth se generó con el propio modelo "
                "(confianza ≥ 0.8). Las métricas deben ser cercanas al 100%; no son métricas reales.")
    elif not gt_video:
        st.warning(f"No hay anotaciones para '{nombre_video}' en ground_truth.json. "
                   f"Captura frames, anótalos en la pestaña Anotar y vuelve a procesar.")

    resultados = evaluar_modelo(model, ruta, gt_video, umbral,
                                modo_prueba=modo_prueba, ver_gt=ver_gt)

    if resultados is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{resultados['Precision']:.2%}")
        c2.metric("Recall", f"{resultados['Recall']:.2%}")
        c3.metric("mAP@0.5", f"{resultados['mAP@0.5']:.2%}")
        c4.metric("mAP@0.5:0.95", f"{resultados['mAP@0.5:0.95']:.2%}")
        st.caption(f"Frames evaluados: {resultados['Frames evaluados']} · "
                   f"TP={resultados['TP']} · FP={resultados['FP']} · FN={resultados['FN']}")
    else:
        resultados = {"Precision": None, "Recall": None, "mAP@0.5": None,
                      "mAP@0.5:0.95": None, "Frames evaluados": 0}
        st.info("Sin métricas de calidad para esta corrida (N/D).")

    lat = sum(latencias) / len(latencias)
    res = {"Video": nombre_video, "Modelo": nombre_modelo, "Resolución": f"{w}x{h}",
           "Modo": "Simulación tiempo real" if simular_rt else "Todos los frames",
           "FPS video": round(fps_in, 1), "Latencia (ms)": round(lat * 1000, 1),
           "FPS efectivo": round(procesados / (i / fps_in) if simular_rt else procesados / sum(latencias), 2),
           "Procesados": procesados, "Total": i,
           "% descartados": round(100 * descartados / i, 1),
           "Detec./frame": round(sum(n_dets) / len(n_dets), 1),
           "Factor RT": round((1 / lat) / fps_in, 2)}
    res.update({k: resultados[k] for k in
                ("Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Frames evaluados")})
    res["¿Tiempo real?"] = "Sí" if res["Factor RT"] >= 1 else "No"
    st.session_state.setdefault("historial", []).append(res)
    if guardar:
        st.success(f"Video con cajas guardado en: {ruta_out}")
    st.success(f"Listo: {procesados}/{i} frames procesados · latencia media {res['Latencia (ms)']} ms · "
               f"¿tiempo real? **{res['¿Tiempo real?']}**")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ------------------------- Anotación de ground truth -------------------------
def df_cajas(filas):
    """DataFrame con tipos correctos para st.data_editor (aunque esté vacío)."""
    df = pd.DataFrame(filas, columns=COLS_BOX)
    return df.astype({"x1": "float64", "y1": "float64", "x2": "float64", "y2": "float64",
                      "clase": "object"})


def anotar_ui():
    videos = sorted(p.name for p in COLA.iterdir() if p.is_dir() and any(p.glob("*.jpg")))
    if not videos:
        st.info("Aún no hay frames capturados. Activa **Guardar frames para anotar** en la barra "
                "lateral y procesa un video.")
        return

    cats = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    id_de = {es(n): i for i, n in enumerate(cats) if n not in ("__background__", "N/A")}
    nom_de = {i: n for n, i in id_de.items()}

    c1, c2 = st.columns(2)
    vid = c1.selectbox("Video", videos, key="an_vid")
    frames = sorted((COLA / vid).glob("*.jpg"))
    gt_vid = cargar_gt().get(vid, {})

    fr = c2.selectbox(
        "Frame", frames, key="an_fr",
        format_func=lambda p: f"Frame {int(p.stem)}" + (" ✅" if str(int(p.stem)) in gt_vid else ""))
    idx = str(int(fr.stem))
    img = cv2.imread(str(fr))
    if img is None:
        st.error(f"No se pudo leer {fr}")
        return

    n_anot = sum(str(int(p.stem)) in gt_vid for p in frames)
    st.caption(f"✅ {n_anot} de {len(frames)} frames anotados · coordenadas en píxeles "
               f"(x1, y1, x2, y2) sobre un frame de {img.shape[1]}x{img.shape[0]}")

    # Estado por (video, frame)
    k_df, k_ver = f"an_df_{vid}_{idx}", f"an_ver_{vid}_{idx}"
    if k_df not in st.session_state:
        if idx in gt_vid:
            g = gt_vid[idx]
            filas = [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "clase": nom_de.get(l, "persona")}
                     for b, l in zip(g["boxes"], g["labels"])]
        else:
            filas = []
        st.session_state[k_df] = df_cajas(filas)

    # Pre-anotación opcional con un modelo
    m1, m2 = st.columns([2, 1])
    modelo_sug = m1.selectbox("Modelo para pre-anotar", list(MODELOS), key="an_mod")
    umbral_sug = m2.slider("Umbral", 0.1, 0.95, 0.5, 0.05, key="an_um")
    if st.button("🤖 Pre-anotar con el modelo (reemplaza la tabla)"):
        model, labels = cargar_modelo(modelo_sug)
        with torch.no_grad():
            p = model([to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device)])[0]
        filas = []
        for b, l, s in zip(p["boxes"].cpu().tolist(), p["labels"].cpu().tolist(),
                           p["scores"].cpu().tolist()):
            if s >= umbral_sug and es(labels[l]) in id_de:
                filas.append({"x1": int(b[0]), "y1": int(b[1]), "x2": int(b[2]), "y2": int(b[3]),
                              "clase": es(labels[l])})
        st.session_state[k_df] = df_cajas(filas)
        st.session_state[k_ver] = st.session_state.get(k_ver, 0) + 1
        st.rerun()

    col_img, col_tab = st.columns([3, 2])
    with col_tab:
        st.markdown("**Cajas del frame** (usa `+` para agregar, selecciona una fila para borrarla)")
        editado = st.data_editor(
            st.session_state[k_df], num_rows="dynamic", use_container_width=True,
            key=f"an_ed_{vid}_{idx}_{st.session_state.get(k_ver, 0)}",
            column_config={
                "x1": st.column_config.NumberColumn(step=1),
                "y1": st.column_config.NumberColumn(step=1),
                "x2": st.column_config.NumberColumn(step=1),
                "y2": st.column_config.NumberColumn(step=1),
                "clase": st.column_config.SelectboxColumn("clase", options=list(id_de), required=True),
            })

    # Solo filas completas y con caja válida
    validas = editado.dropna()
    validas = validas[(validas["x2"] > validas["x1"]) & (validas["y2"] > validas["y1"])]
    validas = validas[validas["clase"].isin(id_de)]

    vis = img.copy()
    for n, r in enumerate(validas.itertuples(), 1):
        c = color_de(r.clase)
        cv2.rectangle(vis, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 3)
        cv2.putText(vis, str(n), (int(r.x1) + 4, int(r.y1) + 28), FONT, 1.0, c, 2, cv2.LINE_AA)
    col_img.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    b1, b2 = st.columns(2)
    if b1.button("💾 Guardar anotación de este frame", type="primary", use_container_width=True):
        gt = cargar_gt()
        gt.setdefault(vid, {})[idx] = {
            "boxes": [[float(r.x1), float(r.y1), float(r.x2), float(r.y2)]
                      for r in validas.itertuples()],
            "labels": [id_de[r.clase] for r in validas.itertuples()],
        }
        guardar_gt(gt)
        st.session_state[k_df] = df_cajas(validas.to_dict("records"))
        st.session_state[k_ver] = st.session_state.get(k_ver, 0) + 1
        st.toast(f"Frame {idx} guardado ({len(validas)} objetos) en {RUTA_GT.name}", icon="💾")
        st.rerun()
    if b2.button("🗑 Quitar anotación de este frame", use_container_width=True):
        gt = cargar_gt()
        if gt.get(vid, {}).pop(idx, None) is not None:
            guardar_gt(gt)
            st.toast(f"Frame {idx} eliminado de {RUTA_GT.name}", icon="🗑")
        st.session_state.pop(k_df, None)
        st.session_state[k_ver] = st.session_state.get(k_ver, 0) + 1
        st.rerun()


# ------------------------- Interfaz -------------------------
st.title("🎯 Detección de objetos en video · Faster R-CNN")

with st.sidebar:
    st.header("⚙️ Configuración")
    st.caption(f"Dispositivo: **{device.upper()}**"
               + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else " (será lento)"))
    origen = st.radio("Origen del video", ["Carpeta ./videos", "Subir archivo"], horizontal=True)
    ruta, nombre_video = None, None
    if origen == "Carpeta ./videos":
        exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        archivos = sorted(p for e in exts for p in CARPETA.glob(e))
        if archivos:
            sel = st.selectbox("Video", archivos, format_func=lambda p: p.name)
            ruta, nombre_video = sel, sel.stem
        else:
            st.warning("No hay videos en ./videos. Copia ahí tus .mp4 o usa 'Subir archivo'.")
    else:
        up = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / up.name
            tmp.write_bytes(up.getbuffer())
            ruta, nombre_video = tmp, Path(up.name).stem

    nombre_modelo = st.selectbox("Modelo", list(MODELOS))
    umbral = st.slider("Umbral de confianza", 0.1, 0.95, 0.6, 0.05)
    simular_rt = st.toggle("Simular cámara en vivo (descarta frames)", value=True,
                           help="Activado: si el modelo tarda, los frames que llegan mientras tanto se descartan, "
                                "como en una cámara real. Desactivado: procesa todos los frames.")
    guardar = st.checkbox("Guardar video con cajas en ./resultados", value=False)
    paso_ui = st.select_slider("Refrescar pantalla cada N frames", [1, 2, 3, 5, 10], value=2,
                               help="Más alto = menos carga de interfaz (útil en 4K).")

    st.divider()
    st.subheader("✍️ Anotación")
    capturar = st.checkbox("Guardar frames para anotar durante el procesamiento", value=False,
                           help="Guarda 1 de cada N frames (sin cajas dibujadas) en ./frames_anotar/<video>/ "
                                "para anotarlos en la pestaña 'Anotar'.")
    cada_n = st.number_input("Capturar 1 frame cada N", min_value=5, max_value=500, value=30, step=5)

    st.divider()
    st.subheader("🧪 Validación de métricas")
    ver_gt = st.checkbox("Mostrar GT vs predicción (verde/rojo)", value=False,
                         help="Dibuja las cajas anotadas (verde) y las predichas (rojo) en cada frame evaluado.")
    modo_prueba = st.checkbox("Modo prueba de cordura", value=False,
                              help="Usa como ground truth las detecciones del modelo (≥ 0.8). "
                                   "Si el pipeline es correcto, Precision y Recall deben ser cercanos a 100%.")

    iniciar = st.button("▶ Iniciar procesamiento", type="primary", use_container_width=True,
                        disabled=ruta is None)
    st.button("⏹ Detener", use_container_width=True,
              help="Interrumpe la ejecución actual (el resultado parcial no se guarda en el historial).")

tab_vivo, tab_anot, tab_hist = st.tabs(
    ["📹 Procesamiento en vivo", "✍️ Anotar", "📈 Historial y comparación"])

with tab_vivo:
    if iniciar and ruta is not None:
        procesar(ruta, nombre_video, nombre_modelo, umbral, simular_rt, guardar, paso_ui,
                 modo_prueba, ver_gt, capturar, int(cada_n))
    else:
        st.info("Elige un video y un modelo en la barra lateral y pulsa **▶ Iniciar procesamiento**.")

with tab_anot:
    anotar_ui()

with tab_hist:
    hist = st.session_state.get("historial", [])
    if not hist:
        st.info("Aún no hay corridas. Cada procesamiento terminado se agrega aquí para comparar.")
    else:
        df = pd.DataFrame(hist)
        st.dataframe(df, hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        lat = df.groupby(["Video", "Modelo"])["Latencia (ms)"].mean().unstack("Modelo")
        rt = df.groupby(["Video", "Modelo"])["Factor RT"].mean().unstack("Modelo")
        c1.markdown("**Latencia media (ms/frame)** · menos es mejor")
        c1.bar_chart(lat)
        c2.markdown("**Factor de tiempo real** · ≥ 1 = tiempo real")
        c2.bar_chart(rt)
        b1, b2 = st.columns(2)
        b1.download_button("⬇ Descargar CSV", df.to_csv(index=False).encode("utf-8"),
                           "resultados_frcnn.csv", "text/csv", use_container_width=True)
        if b2.button("🗑 Limpiar historial", use_container_width=True):
            st.session_state["historial"] = []
            st.rerun()
