"""
Item 4 - Detección de objetos con YOLO en video / simulación de tiempo real
==========================================================================
Motor de detección. Se usa de dos formas:
  - Interfaz web:  streamlit run app.py
  - Consola:       python yolo_video.py --source 0            (webcam)
                   python yolo_video.py --source video.mp4 --track --realtime --save

--realtime simula una cámara en vivo: si el modelo es más lento que el video,
se descartan los frames que "ya pasaron" y se reportan como frames perdidos.

Métricas por frame (CSV) y resumen (JSON):
  latencia (ms), FPS, nº de detecciones, confianza media, conteo por clase,
  IDs únicos por clase (ByteTrack), frames descartados, inestabilidad temporal.
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_MODELOS = {}


def cargar_modelo(nombre="yolo11n.pt"):
    """Carga (y cachea) un modelo YOLO."""
    if nombre not in _MODELOS:
        _MODELOS[nombre] = YOLO(nombre)
    return _MODELOS[nombre]


def color_for(cls_id):
    rng = np.random.default_rng(int(cls_id) * 7 + 3)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def draw(frame, boxes, names, ids=None, escala=1.0):
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(boxes):
        c = color_for(cls)
        p1, p2 = (int(x1 * escala), int(y1 * escala)), (int(x2 * escala), int(y2 * escala))
        cv2.rectangle(frame, p1, p2, c, 2)
        label = f"{names[int(cls)]} {conf:.2f}"
        if ids is not None and ids[i] is not None:
            label = f"#{ids[i]} " + label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), c, -1)
        cv2.putText(frame, label, (p1[0] + 2, p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, fps, lat_ms, n_det, dropped):
    lines = [f"FPS: {fps:5.1f}", f"Latencia: {lat_ms:5.1f} ms", f"Detecciones: {n_det}"]
    if dropped is not None:
        lines.append(f"Frames descartados: {dropped}")
    h = 22 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (250, 8 + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (16, 32 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)


def procesar_video(source, modelo="yolo11n.pt", conf=0.35, iou=0.5, imgsz=640,
                   classes=None, track=True, realtime=False, save_path=None,
                   max_frames=0, on_frame=None, show=False, ancho_vis=1280):
    """Procesa un video/webcam con YOLO.

    on_frame(frame_anotado_bgr, fila_metricas, progreso_0_1) se llama en cada
    frame (la interfaz lo usa para la vista en vivo). Si devuelve False, se detiene.

    El modelo siempre recibe el frame ORIGINAL. Solo la visualización (dibujo,
    video guardado, vista en vivo) se reduce a `ancho_vis` px para no gastar CPU
    en dibujar/guardar 4K, lo que distorsionaría las métricas.

    Simulación de tiempo real (realtime=True): se usa un RELOJ VIRTUAL que solo
    avanza con el tiempo de detección. El frame k "llega" en k/fps; al terminar
    de procesar, se toma el último frame que ya llegó y los intermedios se
    descartan. Así el % de descarte depende solo del detector
    (≈ 1 - FPS_inferencia / FPS_video) y no del disco, la interfaz o la decodificación.

    Devuelve (filas, resumen, extras) donde extras tiene:
      'mejor_frame' (frame con más detecciones), 'confianzas' (todas las confianzas),
      'apariciones' {clase: [frames donde aparece]}.
    """
    src = int(source) if str(source).isdigit() else str(source)
    is_cam = isinstance(src, int)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 1 or src_fps > 240:
        src_fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    escala = min(1.0, ancho_vis / W) if W else 1.0
    Wv, Hv = int(W * escala), int(H * escala)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    model = cargar_modelo(modelo)
    if track and hasattr(model, "predictor") and model.predictor is not None:
        # reinicia el tracker para que los IDs empiecen de nuevo en cada video
        for t in getattr(model.predictor, "trackers", []) or []:
            t.reset()
    names = model.names
    # Calentamiento: la 1ra inferencia es lenta (carga de pesos / grafo)
    model.predict(np.zeros((H or 480, W or 640, 3), np.uint8), imgsz=imgsz, verbose=False)

    writer = None
    if save_path:
        writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (Wv, Hv))

    rows, all_confs = [], []
    class_counts = Counter()
    unique_ids = defaultdict(set)
    apariciones = defaultdict(list)
    conf_por_clase = defaultdict(list)
    mejor = (-1, None, -1)          # (n_det, frame, idx)
    dropped = frame_idx = 0
    fps_ema = None
    t_virtual = 0.0                  # reloj virtual de la simulación de tiempo real
    t_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        if track:
            r = model.track(frame, conf=conf, iou=iou, imgsz=imgsz, classes=classes,
                            persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        else:
            r = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz,
                              classes=classes, verbose=False)[0]
        lat = (time.perf_counter() - t0) * 1000.0
        sp = getattr(r, "speed", None) or {}

        b = r.boxes
        boxes, ids = [], None
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            clss = b.cls.cpu().numpy().astype(int)
            boxes = [(*xyxy[i], float(confs[i]), int(clss[i])) for i in range(len(clss))]
            all_confs.extend(confs.tolist())
            if track and b.id is not None:
                ids = b.id.cpu().numpy().astype(int).tolist()
                for c, tid in zip(clss, ids):
                    unique_ids[names[c]].add(tid)
            for c, cf in zip(clss, confs):
                class_counts[names[c]] += 1
                conf_por_clase[names[c]].append(float(cf))
            for c in set(clss.tolist()):
                apariciones[names[c]].append(frame_idx + dropped)

        inst_fps = 1000.0 / lat if lat > 0 else 0
        fps_ema = inst_fps if fps_ema is None else 0.9 * fps_ema + 0.1 * inst_fps

        if escala < 1.0:
            frame = cv2.resize(frame, (Wv, Hv), interpolation=cv2.INTER_AREA)
        draw(frame, boxes, names, ids, escala)
        draw_hud(frame, fps_ema, lat, len(boxes), dropped if realtime else None)

        fila = {
            "frame": frame_idx,
            "frame_video": frame_idx + dropped,
            "tiempo_video_s": round((frame_idx + dropped) / src_fps, 3),
            "latencia_ms": round(lat, 2),
            "fps": round(inst_fps, 2),
            "preproceso_ms": round(sp.get("preprocess", 0.0), 2),
            "inferencia_ms": round(sp.get("inference", 0.0), 2),
            "postproceso_ms": round(sp.get("postprocess", 0.0), 2),
            "n_detecciones": len(boxes),
            "conf_media": round(float(np.mean([x[4] for x in boxes])), 4) if boxes else 0.0,
            "clases": "|".join(sorted({names[x[5]] for x in boxes})),
        }
        rows.append(fila)

        if len(boxes) > mejor[0]:
            mejor = (len(boxes), frame.copy(), fila["frame_video"])
        if writer:
            writer.write(frame)
        if show:
            cv2.imshow("YOLO - video / tiempo real (q para salir)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if on_frame is not None:
            prog = min(1.0, (frame_idx + dropped) / total) if total else 0.0
            if on_frame(frame, fila, prog) is False:
                break
        if max_frames and frame_idx >= max_frames:
            break

        # Simulación de tiempo real con reloj virtual (ver docstring).
        if realtime and not is_cam:
            pos = fila["frame_video"]                       # frame recién procesado
            t_virtual = max(t_virtual, pos / src_fps) + lat / 1000.0
            siguiente = max(pos + 1, int(t_virtual * src_fps))
            for _ in range(siguiente - pos - 1):
                if not cap.grab():
                    break
                dropped += 1

    wall = time.perf_counter() - t_start
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    if not rows:
        raise RuntimeError("No se procesó ningún frame.")

    lat = np.array([r["latencia_ms"] for r in rows])
    ndet = np.array([r["n_detecciones"] for r in rows])
    cm = np.array([r["conf_media"] for r in rows if r["n_detecciones"] > 0])
    # Estabilidad temporal: cambio medio del nº de detecciones entre frames
    flicker = float(np.mean(np.abs(np.diff(ndet)))) if len(ndet) > 1 else 0.0

    resumen = {
        "fuente": str(source),
        "modelo": modelo,
        "resolucion": f"{W}x{H}",
        "fps_fuente": round(src_fps, 2),
        "duracion_video_s": round(total / src_fps, 2) if total else None,
        "frames_procesados": len(rows),
        "frames_descartados": dropped,
        "pct_descartados": round(100 * dropped / (len(rows) + dropped), 1),
        "frames_totales_video": total,
        "tiempo_total_s": round(wall, 2),
        "latencia_ms": {"media": round(float(lat.mean()), 2),
                        "mediana": round(float(np.median(lat)), 2),
                        "p95": round(float(np.percentile(lat, 95)), 2),
                        "min": round(float(lat.min()), 2),
                        "max": round(float(lat.max()), 2)},
        "fps_inferencia_medio": round(1000.0 / float(lat.mean()), 2),
        "fps_efectivo_extremo_a_extremo": round(len(rows) / wall, 2),
        "factor_tiempo_real": round(1000.0 / float(lat.mean()) / src_fps, 2),
        "desglose_ms": {k: round(float(np.mean([r[k + "_ms"] for r in rows])), 2)
                        for k in ("preproceso", "inferencia", "postproceso")},
        "tiempo_real": bool(1000.0 / float(lat.mean()) >= src_fps),
        "detecciones_totales": int(ndet.sum()),
        "detecciones_por_frame_media": round(float(ndet.mean()), 2),
        "max_detecciones_en_un_frame": int(ndet.max()),
        "frames_sin_detecciones_pct": round(100 * float((ndet == 0).mean()), 1),
        "conf_media": round(float(cm.mean()), 4) if len(cm) else 0.0,
        "inestabilidad_detecciones": round(flicker, 3),
        "detecciones_por_clase": dict(class_counts.most_common()),
        "objetos_unicos_por_clase": {k: len(v) for k, v in unique_ids.items()} if track else None,
    }
    extras = {"mejor_frame": mejor[1], "mejor_frame_idx": mejor[2],
              "confianzas": all_confs, "apariciones": dict(apariciones),
              "conf_por_clase": dict(conf_por_clase)}
    return rows, resumen, extras


def generar_hallazgos(res, df=None, ex=None):
    """Frases automáticas con los datos más interesantes del análisis."""
    h = []
    fps_inf, fps_src = res["fps_inferencia_medio"], res["fps_fuente"]
    if res["tiempo_real"]:
        h.append(f"⚡ **Va en tiempo real**: procesa {fps_inf:.1f} FPS frente a los {fps_src:.0f} FPS "
                 f"del video ({fps_inf / fps_src:.1f}× más rápido de lo necesario).")
    else:
        h.append(f"🐢 **No llega a tiempo real**: procesa {fps_inf:.1f} FPS y el video va a {fps_src:.0f} FPS. "
                 f"En una cámara en vivo se perdería ~{max(0, 100 * (1 - fps_inf / fps_src)):.0f}% de los frames.")
    if res["frames_descartados"]:
        h.append(f"🗑️ En la simulación de tiempo real se descartaron **{res['frames_descartados']} frames** "
                 f"({res['pct_descartados']}% del video); factor de tiempo real {res['factor_tiempo_real']}.")
    clases = res["detecciones_por_clase"]
    if clases:
        top = next(iter(clases))
        h.append(f"🏆 La clase más frecuente es **{top}** con {clases[top]} detecciones "
                 f"({100 * clases[top] / res['detecciones_totales']:.0f}% del total).")
    if ex["mejor_frame_idx"] >= 0 and res["max_detecciones_en_un_frame"] > 0:
        seg = ex["mejor_frame_idx"] / fps_src
        h.append(f"👥 El momento con más objetos es el **segundo {seg:.1f}**, con "
                 f"{res['max_detecciones_en_un_frame']} detecciones a la vez.")
    if res["objetos_unicos_por_clase"]:
        tot = sum(res["objetos_unicos_por_clase"].values())
        h.append(f"🔢 El seguimiento (ByteTrack) identificó **{tot} objetos distintos**; "
                 f"sin seguimiento solo se sabría que hubo {res['detecciones_totales']} detecciones.")
    if res["frames_sin_detecciones_pct"] > 0:
        h.append(f"🕳️ En el **{res['frames_sin_detecciones_pct']}%** de los frames no se detectó nada.")
    inest = res["inestabilidad_detecciones"]
    nivel = "baja" if inest < 0.2 else "moderada" if inest < 0.6 else "alta"
    h.append(f"📉 Inestabilidad temporal **{nivel}** ({inest}): en promedio el número de objetos cambia "
             f"{inest} entre un frame y el siguiente (parpadeo de detecciones).")
    cpc = ex["conf_por_clase"]
    if len(cpc) > 1:
        medias = {k: np.mean(v) for k, v in cpc.items()}
        segura, dudosa = max(medias, key=medias.get), min(medias, key=medias.get)
        h.append(f"🎯 El modelo está más seguro con **{segura}** (confianza {medias[segura]:.2f}) "
                 f"y menos con **{dudosa}** ({medias[dudosa]:.2f}), posible falso positivo.")
    lat = res["latencia_ms"]
    if lat["max"] > 2 * lat["mediana"]:
        h.append(f"⏱️ Hubo picos de latencia de hasta **{lat['max']:.0f} ms** (mediana {lat['mediana']:.0f} ms), "
                 f"típicos del primer frame o de carga del sistema.")
    return h


def guardar_resultados(rows, resumen, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metricas_por_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(out / "resumen_metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description="YOLO en video / tiempo real")
    p.add_argument("--source", default="0", help="0 = webcam, o ruta de video")
    p.add_argument("--model", default="yolo11n.pt", help="yolo11n/s/m.pt, yolov8n.pt ...")
    p.add_argument("--conf", type=float, default=0.35, help="umbral de confianza")
    p.add_argument("--iou", type=float, default=0.5, help="umbral IoU para NMS")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--classes", type=int, nargs="*", default=None,
                   help="filtrar clases COCO, ej: --classes 0 2 (persona, auto)")
    p.add_argument("--track", action="store_true", help="seguimiento con ByteTrack")
    p.add_argument("--realtime", action="store_true",
                   help="simular tiempo real descartando frames atrasados")
    p.add_argument("--save", action="store_true", help="guardar video anotado")
    p.add_argument("--no-show", action="store_true", help="sin ventana (servidor/Colab)")
    p.add_argument("--max-frames", type=int, default=0, help="0 = sin límite")
    p.add_argument("--out", default="resultados", help="carpeta de salida")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Fuente: {a.source} | modelo: {a.model}")
    rows, resumen, _ = procesar_video(
        a.source, a.model, a.conf, a.iou, a.imgsz, a.classes, a.track, a.realtime,
        save_path=out / "video_anotado.mp4" if a.save else None,
        max_frames=a.max_frames, show=not a.no_show)
    guardar_resultados(rows, resumen, out)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nResultados en: {out.resolve()}")


if __name__ == "__main__":
    main()
