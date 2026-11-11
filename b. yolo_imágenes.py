#!/usr/bin/env python3
"""
Detección de objetos con YOLO (Ultralytics YOLO11 / YOLO12) preentrenado en COCO (80 clases).
Misma interfaz y MISMO evaluador de métricas que rcnn_detector.py, para poder comparar.

Instalación:
  pip install -U ultralytics pillow numpy customtkinter      (gradio solo si usas --web)

Uso rápido
----------
  python yolo_detector.py                       # ventana de escritorio (por defecto)
  python yolo_detector.py imagen.jpg            # ventana con la imagen ya cargada
  python yolo_detector.py carpeta/ --cli -m yolo12x.pt --imgsz 1280
  python yolo_detector.py foto.jpg --cli --classes person car --json
  python yolo_detector.py coco128/images/train2017 --cli --gt coco128/labels/train2017   # mAP
  python yolo_detector.py --web
  python yolo_detector.py --list-models
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# nombre -> descripción (box AP en COCO val según Ultralytics)
MODELS = {
    "yolo11n.pt": "YOLO11n · nano, el más rápido (box AP 39.5)",
    "yolo11s.pt": "YOLO11s · small (box AP 47.0)",
    "yolo11m.pt": "YOLO11m · medium (box AP 51.5)",
    "yolo11l.pt": "YOLO11l · large (box AP 53.4)",
    "yolo11x.pt": "YOLO11x · extra large (box AP 54.7)",
    "yolo12n.pt": "YOLO12n · nano con atención (box AP 40.6)",
    "yolo12s.pt": "YOLO12s · small con atención (box AP 48.0)",
    "yolo12m.pt": "YOLO12m · medium con atención (box AP 52.5)",
    "yolo12l.pt": "YOLO12l · large con atención (box AP 53.7)",
    "yolo12x.pt": "YOLO12x · el más preciso (box AP 55.2)",
}
DEFAULT_MODEL = "yolo11x.pt"
COMPARE_MODELS = ["yolo11n.pt", "yolo11m.pt", "yolo11x.pt", "yolo12x.pt"]   # los que usa "Comparar modelos"
IMGSZ_OPTIONS = ["640", "960", "1280"]
DEFAULT_SCORE = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_THR = 0.05   # umbral mínimo de detección: igual que en rcnn_detector.py (el mAP necesita toda la curva)
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
              "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
              "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
              "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
              "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
              "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
ALL_CLASSES = sorted(COCO_NAMES)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class YOLODetector:
    """Envoltura sencilla alrededor de un modelo YOLO de Ultralytics."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 score_thr: float = DEFAULT_SCORE, nms_iou: float = 0.45, imgsz: int = 640):
        if model_name not in MODELS:
            raise ValueError(f"Modelo desconocido: {model_name}. Opciones: {list(MODELS)}")
        import torch
        from ultralytics import YOLO
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name)          # se descarga la primera vez
        self.categories = self.model.names     # {0: 'person', ...}
        self.score_thr, self.nms_iou, self.imgsz, self.model_name = score_thr, nms_iou, imgsz, model_name

    def detect(self, image: Image.Image, score_thr: float | None = None,
               classes: set[str] | None = None) -> list[dict]:
        """Devuelve [{label, score, box:[x1,y1,x2,y2]}, ...] ordenado por score."""
        thr = self.score_thr if score_thr is None else score_thr
        r = self.model.predict(image.convert("RGB"), conf=thr, iou=self.nms_iou, imgsz=self.imgsz,
                               device=self.device, max_det=300, verbose=False)[0]
        results = []
        for box, c, s in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist(),
                             r.boxes.conf.cpu().tolist()):
            name = self.categories[int(c)]
            if classes and name not in classes:
                continue
            results.append({"label": name, "score": round(s, 4), "box": [round(v, 1) for v in box]})
        return sorted(results, key=lambda d: -d["score"])


@lru_cache(maxsize=4)
def get_detector(model_name: str, device: str | None) -> YOLODetector:
    return YOLODetector(model_name, device)


# --------------------------------------------------------------------------- #
# Dibujo
# --------------------------------------------------------------------------- #
def _color_for(label: str) -> tuple[int, int, int]:
    h = (sum(ord(c) * (i + 1) for i, c in enumerate(label)) % 360) / 360
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size: int):
    for loader in (lambda: ImageFont.load_default(size=size),
                   lambda: ImageFont.truetype("DejaVuSans.ttf", size),
                   lambda: ImageFont.load_default()):
        try:
            return loader()
        except Exception:
            continue


def draw_detections(image: Image.Image, detections: list[dict],
                    highlight: int | None = None) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    thickness = max(2, round(min(img.size) / 250))
    font = _font(max(12, round(min(img.size) / 40)))
    for i, d in enumerate(detections):
        color = _color_for(d["label"])
        x1, y1, x2, y2 = d["box"]
        if i == highlight:
            draw.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=(255, 255, 255), width=thickness + 2)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        text = f'{d["label"]} {d["score"]:.0%}'
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y1
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2 - t), text, fill=(0, 0, 0), font=font)
    return img


# --------------------------------------------------------------------------- #
# Métricas: mAP (estilo COCO), precisión y recall  (idéntico a rcnn_detector.py)
# --------------------------------------------------------------------------- #
IOU_THRS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
_NAME_MAP = {c.lower().replace("_", " "): c for c in COCO_NAMES}


def _norm_label(name: str) -> str:
    name = str(name).strip()
    return _NAME_MAP.get(name.lower().replace("_", " "), name)


def _iou(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _match(ious, thr: float) -> list[bool]:
    n_d, n_g = ious.shape
    taken, flags = [False] * n_g, []
    for di in range(n_d):
        best, m = min(thr, 1 - 1e-10), -1
        for gj in range(n_g):
            if taken[gj] or ious[di, gj] < best:
                continue
            best, m = ious[di, gj], gj
        if m >= 0:
            taken[m] = True
        flags.append(m >= 0)
    return flags


def _ap(scores, tps, n_gt: int, rec_thrs) -> float:
    import numpy as np
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    tp = np.asarray(tps, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp + np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, rec_thrs, side="left")
    q = np.zeros(len(rec_thrs))
    ok = idx < len(precision)
    q[ok] = precision[idx[ok]]
    return float(q.mean())


def evaluate_map(samples: list) -> dict:
    """samples: lista de (predicciones, anotaciones) por imagen. Devuelve mAP@50, @75, @[.5:.95]."""
    import numpy as np
    rec_thrs = np.linspace(0.0, 1.0, 101)
    classes = sorted({g["label"] for _, gts in samples for g in gts})
    per_class = {}
    for c in classes:
        n_gt, per_img = 0, []
        for preds, gts in samples:
            p = sorted((d for d in preds if d["label"] == c), key=lambda d: -d["score"])[:100]
            g = [x for x in gts if x["label"] == c]
            n_gt += len(g)
            per_img.append(([d["score"] for d in p], _iou([d["box"] for d in p], [x["box"] for x in g])))
        aps = []
        for thr in IOU_THRS:
            scores, tps = [], []
            for sc, ious in per_img:
                scores += sc
                tps += _match(ious, thr)
            aps.append(_ap(scores, tps, n_gt, rec_thrs))
        per_class[c] = {"ap50": aps[0], "ap75": aps[5], "ap": float(np.mean(aps)), "n_gt": n_gt}

    def mean(key):
        return float(np.mean([v[key] for v in per_class.values()])) if per_class else None

    return {"map50": mean("ap50"), "map75": mean("ap75"), "map": mean("ap"), "per_class": per_class,
            "n_images": len(samples), "n_gt": sum(v["n_gt"] for v in per_class.values()),
            "unsupported": [c for c in classes if c not in COCO_NAMES]}


def precision_recall(samples: list, score_thr: float = 0.5, iou_thr: float = 0.5) -> dict:
    tp = fp = fn = 0
    for preds, gts in samples:
        for c in {d["label"] for d in preds} | {g["label"] for g in gts}:
            p = sorted((d for d in preds if d["label"] == c and d["score"] >= score_thr),
                       key=lambda d: -d["score"])
            g = [x for x in gts if x["label"] == c]
            hit = sum(_match(_iou([d["box"] for d in p], [x["box"] for x in g]), iou_thr))
            tp += hit
            fp += len(p) - hit
            fn += len(g) - hit
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "score_thr": score_thr, "iou_thr": iou_thr}


class GroundTruth:
    """Anotaciones reales: JSON COCO, XML Pascal VOC o TXT YOLO (archivo o carpeta).
    En YOLO los nombres se leen de classes.txt / obj.names; si no hay, se asume el orden COCO-80
    (que es el de coco128 de Ultralytics: coco128/labels/train2017)."""

    def __init__(self, source):
        self.src = Path(source)
        if not self.src.exists():
            raise FileNotFoundError(f"No existe: {self.src}")
        self.coco: dict | None = None
        self.names: list[str] | None = None
        if self.src.is_file() and self.src.suffix.lower() == ".json":
            self._load_coco()

    @property
    def is_dataset(self) -> bool:
        return self.coco is not None or self.src.is_dir()

    def _load_coco(self):
        data = json.loads(self.src.read_text(encoding="utf-8"))
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        names = {im["id"]: Path(im["file_name"]).name.lower() for im in data["images"]}
        self.coco = {n: [] for n in names.values()}
        for a in data.get("annotations", []):
            n = names.get(a["image_id"])
            if n is None or a.get("iscrowd"):
                continue
            x, y, w, h = a["bbox"]
            self.coco[n].append({"label": _norm_label(cats.get(a["category_id"], a["category_id"])),
                                 "box": [x, y, x + w, y + h]})

    def for_image(self, image_path, size=None):
        ip = Path(image_path)
        if self.coco is not None:
            return self.coco.get(ip.name.lower())
        f = self.src
        if f.is_dir():
            f = next((f / (ip.stem + e) for e in (".xml", ".txt") if (f / (ip.stem + e)).is_file()), None)
            if f is None:
                return None
        ext = f.suffix.lower()
        if ext == ".xml":
            return self._voc(f)
        if ext == ".txt":
            if size is None:
                with Image.open(ip) as im:
                    size = im.size
            return self._yolo(f, size)
        raise ValueError("Formato de anotaciones no soportado (usa .json, .xml o .txt)")

    @staticmethod
    def _voc(f: Path):
        import xml.etree.ElementTree as ET
        out = []
        for obj in ET.parse(f).getroot().iter("object"):
            bb = obj.find("bndbox")
            if bb is not None:
                box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                out.append({"label": _norm_label(obj.findtext("name", "")), "box": box})
        return out

    def _class_names(self, txt: Path) -> list[str]:
        if self.names is None:
            for d in [txt.parent, txt.parent.parent] + ([self.src] if self.src.is_dir() else []):
                for fn in ("classes.txt", "obj.names", "classes.names", "labels.txt"):
                    if (d / fn).is_file():
                        self.names = [ln.strip() for ln in (d / fn).read_text(encoding="utf-8").splitlines()
                                      if ln.strip()]
                        break
                if self.names:
                    break
            self.names = self.names or COCO_NAMES
        return self.names

    def _yolo(self, f: Path, size):
        W, H = size
        names, out = self._class_names(f), []
        for line in f.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            k = int(float(p[0]))
            cx, cy, w, h = (float(v) for v in p[1:5])
            label = _norm_label(names[k] if 0 <= k < len(names) else f"clase_{k}")
            out.append({"label": label, "box": [(cx - w / 2) * W, (cy - h / 2) * H,
                                                 (cx + w / 2) * W, (cy + h / 2) * H]})
        return out


def evaluate_dataset(detector, images, gt: GroundTruth, classes=None, score_thr: float = 0.5,
                     progress=None, cancel=None) -> dict:
    """Detecta en cada imagen y calcula mAP contra las anotaciones (omite imágenes sin anotación)."""
    samples, times, skipped = [], [], 0
    for i, f in enumerate(images, 1):
        if cancel and cancel():
            break
        with Image.open(f) as im:
            img = im.convert("RGB")
        gts = gt.for_image(f, img.size)
        if gts is None:
            skipped += 1
        else:
            t0 = time.perf_counter()
            dets = detector.detect(img, MIN_THR)
            times.append(time.perf_counter() - t0)
            if classes:
                dets = [d for d in dets if d["label"] in classes]
                gts = [g for g in gts if g["label"] in classes]
            samples.append((dets, gts))
        if progress:
            progress(i, len(images))
    m = evaluate_map(samples)
    m["pr"] = precision_recall(samples, score_thr)
    m["skipped"] = skipped
    m["tiempo_medio_s"] = sum(times) / len(times) if times else None
    return m


def format_metrics(m: dict) -> str:
    if m["map50"] is None:
        return "No hay objetos anotados para evaluar."
    lines = []
    if m.get("modelo"):
        lines.append(f"Modelo: {m['modelo']}   ·   Dispositivo: {m.get('dispositivo', '?')}")
    lines += [f"Imágenes evaluadas: {m['n_images']}   ·   Objetos anotados (GT): {m['n_gt']}",
              f"mAP@0.5       : {m['map50']:.4f}",
              f"mAP@0.75      : {m['map75']:.4f}",
              f"mAP@0.5:0.95  : {m['map']:.4f}"]
    pr = m.get("pr")
    if pr:
        lines.append(f"Precisión {pr['precision']:.3f} · Recall {pr['recall']:.3f} · F1 {pr['f1']:.3f}"
                     f"   (umbral {pr['score_thr']:.2f}, IoU {pr['iou_thr']})")
    t = m.get("tiempo_medio_s")
    if t:
        lines.append(f"Tiempo medio  : {t:.3f} s/imagen  ({1 / t:.1f} FPS)")
    lines += ["", f"{'Clase':<22}{'AP50':>8}{'AP':>8}{'GT':>6}"]
    for k, v in sorted(m["per_class"].items(), key=lambda kv: -kv[1]["ap50"]):
        lines.append(f"{k:<22}{v['ap50']:>8.3f}{v['ap']:>8.3f}{v['n_gt']:>6}")
    if m["unsupported"]:
        lines.append("\nClases anotadas que no existen en COCO (AP = 0): " + ", ".join(m["unsupported"]))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def collect_images(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
        else:
            print(f"[aviso] se ignora: {p}", file=sys.stderr)
    return files


def run_cli(args) -> int:
    files = collect_images(args.inputs)
    if not files:
        print("No se encontraron imágenes válidas.", file=sys.stderr)
        return 1
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cargando {args.model} ...")
    det = get_detector(args.model, args.device)
    det.imgsz = args.imgsz
    print(f"Dispositivo: {det.device} · imgsz {det.imgsz} · umbral {args.score}")
    classes = set(args.classes) if args.classes else None
    gt = GroundTruth(args.gt) if args.gt else None
    if gt is not None and not gt.is_dataset:
        print("--gt en modo consola requiere un JSON COCO o una carpeta de anotaciones.", file=sys.stderr)
        return 1
    samples, times, skipped, all_results = [], [], 0, {}
    det.detect(Image.new("RGB", (640, 640)), 0.5)       # calentamiento (no cuenta en el tiempo)

    for f in files:
        img = Image.open(f)
        t0 = time.perf_counter()
        if gt is None:
            dets = det.detect(img, args.score, classes)
        else:                                            # para el mAP se necesita el umbral mínimo
            raw = det.detect(img, MIN_THR)
            dets = [d for d in raw if d["score"] >= args.score and (not classes or d["label"] in classes)]
        dt = time.perf_counter() - t0
        if gt is not None:
            gts = gt.for_image(f, img.size)
            if gts is None:
                skipped += 1
            else:
                times.append(dt)
                samples.append(([d for d in raw if not classes or d["label"] in classes],
                                [g for g in gts if not classes or g["label"] in classes]))
        dest = out_dir / f"{f.stem}_det.jpg"
        draw_detections(img, dets).save(dest, quality=95)
        all_results[str(f)] = {"tiempo_s": round(dt, 3), "dispositivo": str(det.device), "detecciones": dets}
        print(f"{f.name}: {len(dets)} objetos ({', '.join(sorted({d['label'] for d in dets})) or 'nada'})"
              f" · {dt:.2f}s → {dest}")

    if gt is not None:
        m = evaluate_map(samples)
        m["pr"] = precision_recall(samples, args.score)
        m["skipped"] = skipped
        m["tiempo_medio_s"] = sum(times) / len(times) if times else None
        m["modelo"], m["dispositivo"], m["imgsz"] = args.model, str(det.device), args.imgsz
        print("\n" + format_metrics(m))
        (out_dir / "metricas.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Métricas guardadas en {out_dir / 'metricas.json'}")
    if args.json:
        (out_dir / "detecciones.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"JSON guardado en {out_dir / 'detecciones.json'}")
    return 0


# --------------------------------------------------------------------------- #
# Interfaz web (opcional)
# --------------------------------------------------------------------------- #
def run_web(args) -> int:
    try:
        import gradio as gr
    except ImportError:
        print("Instala gradio:  pip install gradio", file=sys.stderr)
        return 1

    def predict(image, model_name, thr, size, class_text):
        if image is None:
            return None, []
        classes = {c.strip() for c in class_text.split(",") if c.strip()} or None
        pil = Image.fromarray(image)
        det = get_detector(model_name, args.device)
        det.imgsz = int(size)
        dets = det.detect(pil, thr, classes)
        rows = [[d["label"], f'{d["score"]:.3f}', *map(int, d["box"])] for d in dets]
        return draw_detections(pil, dets), rows

    gr.Interface(
        fn=predict,
        inputs=[gr.Image(type="numpy", label="Imagen"),
                gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Modelo"),
                gr.Slider(0.05, 0.99, value=DEFAULT_SCORE, step=0.05, label="Umbral de confianza"),
                gr.Dropdown(IMGSZ_OPTIONS, value="640", label="Tamaño de entrada (imgsz)"),
                gr.Textbox(label="Filtrar clases (coma, opcional)", placeholder="person, car, dog")],
        outputs=[gr.Image(label="Resultado"),
                 gr.Dataframe(headers=["clase", "score", "x1", "y1", "x2", "y2"], label="Detecciones")],
        title="Detección de objetos con YOLO", description="YOLO11 / YOLO12 preentrenado en COCO (Ultralytics).",
        flagging_mode="never").launch(server_port=args.port)
    return 0


# --------------------------------------------------------------------------- #
# Ventana de escritorio moderna (CustomTkinter), tema oscuro
# --------------------------------------------------------------------------- #
def run_gui(args) -> int:
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox

    try:
        import customtkinter as ctk
    except ImportError:
        print("Para la ventana moderna instala:  pip install customtkinter\n"
              "(o usa --cli / --web)", file=sys.stderr)
        return 1
    from PIL import ImageTk

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    BG, PANEL, CARD, FG, MUTED = "#0b0d12", "#12151c", "#1b1f2a", "#e8eaee", "#8a92a1"
    ACCENT, ACCENT_H, GHOST = "#6c8cff", "#5476f0", "#242a38"
    FONT = "Segoe UI"

    root = ctk.CTk()
    root.title("YOLO · Detección de objetos")
    root.geometry("1340x880")
    root.minsize(1120, 760)
    root.configure(fg_color=BG)

    S = dict(path=None, image=None, raw=[], meta=None, gt=None, gt_obj=None, metrics=None,
             busy=False, selected=None, photo=None, t0=0.0)
    model_var = tk.StringVar(value=DEFAULT_MODEL)
    thr_var = tk.DoubleVar(value=DEFAULT_SCORE)
    thr_text = tk.StringVar(value=f"{DEFAULT_SCORE:.0%}")
    cls_var, imgsz_var = tk.StringVar(), tk.StringVar(value="640")
    status, file_var = tk.StringVar(value="Abre una imagen para empezar"), tk.StringVar()
    t_var, n_var, d_var, m50_var, m95_var, pr_var = (tk.StringVar(value="—") for _ in range(6))
    show_gt = tk.BooleanVar(value=True)
    jobs: "queue.Queue" = queue.Queue()

    # ---------- cabecera ----------
    head = ctk.CTkFrame(root, fg_color="transparent")
    head.pack(fill="x", padx=20, pady=(16, 10))
    ctk.CTkLabel(head, text="◉  YOLO Vision", font=(FONT, 20, "bold"), text_color=FG).pack(side="left")
    ctk.CTkLabel(head, textvariable=file_var, font=(FONT, 12), text_color=MUTED).pack(side="left", padx=18)

    def gbtn(parent, text, cmd, accent=False, **kw):
        return ctk.CTkButton(parent, text=text, command=cmd, corner_radius=10, height=36,
                             font=(FONT, 12, "bold"),
                             fg_color=ACCENT if accent else GHOST,
                             hover_color=ACCENT_H if accent else "#2f3646",
                             text_color="white" if accent else FG, **kw)

    # ---------- cuerpo ----------
    body = ctk.CTkFrame(root, fg_color="transparent")
    body.pack(fill="both", expand=True, padx=20)
    stage = ctk.CTkFrame(body, fg_color="#080a0e", corner_radius=16)
    stage.pack(side="left", fill="both", expand=True)
    canvas = tk.Canvas(stage, bg="#080a0e", highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True, padx=8, pady=8)
    side = ctk.CTkScrollableFrame(body, fg_color=PANEL, corner_radius=16, width=350)
    side.pack(side="right", fill="y", padx=(14, 0))

    def cap(text, pady=(14, 4)):
        ctk.CTkLabel(side, text=text, font=(FONT, 10, "bold"), text_color=MUTED).pack(anchor="w", padx=18, pady=pady)

    cap("MODELO", (16, 4))
    desc = ctk.CTkLabel(side, text=MODELS[DEFAULT_MODEL], font=(FONT, 11), text_color=MUTED,
                        wraplength=300, justify="left")

    def on_model(_=None):
        desc.configure(text=MODELS[model_var.get()])
        detect()

    ctk.CTkOptionMenu(side, variable=model_var, values=list(MODELS), command=on_model, height=34,
                      corner_radius=10, fg_color=CARD, button_color=GHOST, button_hover_color="#2f3646",
                      dropdown_fg_color=CARD, font=(FONT, 12)).pack(fill="x", padx=18)
    desc.pack(anchor="w", padx=18, pady=(6, 0))

    cap("TAMAÑO DE ENTRADA (imgsz)")
    ctk.CTkSegmentedButton(side, values=IMGSZ_OPTIONS, variable=imgsz_var, command=lambda v: detect(),
                           selected_color=ACCENT, selected_hover_color=ACCENT_H, unselected_color=CARD,
                           unselected_hover_color=GHOST, font=(FONT, 12)).pack(fill="x", padx=18)

    cap("CONFIANZA MÍNIMA")
    r = ctk.CTkFrame(side, fg_color="transparent")
    r.pack(fill="x", padx=18)
    ctk.CTkSlider(r, from_=MIN_THR, to=0.95, variable=thr_var, button_color=ACCENT,
                  button_hover_color=ACCENT_H, progress_color=ACCENT,
                  command=lambda v: (thr_text.set(f"{float(v):.0%}"), refresh())).pack(side="left", fill="x", expand=True)
    ctk.CTkLabel(r, textvariable=thr_text, width=44, font=(FONT, 12, "bold"), text_color=FG).pack(side="right", padx=(8, 0))

    cap("CLASES (separadas por coma, vacío = todas)")
    ent = ctk.CTkEntry(side, textvariable=cls_var, placeholder_text="person, car, dog", height=34,
                       corner_radius=10, fg_color=CARD, border_width=0, font=(FONT, 12))
    ent.pack(fill="x", padx=18)

    def stat_row(items):
        f = ctk.CTkFrame(side, fg_color="transparent")
        f.pack(fill="x", padx=18, pady=(12, 0))
        for i, (c, v) in enumerate(items):
            f.columnconfigure(i, weight=1, uniform="s")
            card = ctk.CTkFrame(f, fg_color=CARD, corner_radius=12)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            ctk.CTkLabel(card, textvariable=v, font=(FONT, 14, "bold"), text_color=FG).pack(anchor="w", padx=10, pady=(8, 0))
            ctk.CTkLabel(card, text=c, font=(FONT, 9, "bold"), text_color=MUTED).pack(anchor="w", padx=10, pady=(0, 8))

    stat_row([("TIEMPO", t_var), ("OBJETOS", n_var), ("DISPOSITIVO", d_var)])
    stat_row([("mAP@50", m50_var), ("mAP@50:95", m95_var)])
    ctk.CTkLabel(side, textvariable=pr_var, font=(FONT, 11), text_color=MUTED).pack(anchor="w", padx=18, pady=(8, 0))
    ctk.CTkSwitch(side, text="Ver anotaciones (GT) en blanco", variable=show_gt, command=lambda: refresh(),
                  progress_color=ACCENT, font=(FONT, 12)).pack(anchor="w", padx=18, pady=(8, 0))

    # --- botones de acción (arriba, siempre visibles) ---
    gbtn(side, "Detectar", lambda: detect(), accent=True).pack(fill="x", padx=18, pady=(14, 8))
    row = ctk.CTkFrame(side, fg_color="transparent")
    row.pack(fill="x", padx=18, pady=(0, 6))
    row.columnconfigure((0, 1), weight=1, uniform="b")
    gbtn(row, "Anotaciones…", lambda: load_gt()).grid(row=0, column=0, sticky="ew", padx=(0, 4))
    gbtn(row, "Evaluar carpeta…", lambda: eval_folder()).grid(row=0, column=1, sticky="ew", padx=(4, 0))
    gbtn(side, "Comparar modelos", lambda: compare_models()).pack(fill="x", padx=18, pady=(0, 4))

    # --- secciones desplegables ---
    def section(title, open_=True):
        wrap = ctk.CTkFrame(side, fg_color="transparent")
        wrap.pack(fill="x", padx=18, pady=(10, 0))
        inner = ctk.CTkFrame(wrap, fg_color="transparent")
        st = {"open": open_}
        btn = ctk.CTkButton(wrap, anchor="w", height=34, corner_radius=10, fg_color=CARD, hover_color=GHOST,
                            text_color=FG, font=(FONT, 11, "bold"))

        def paint():
            btn.configure(text=("▾   " if st["open"] else "▸   ") + title)
            if st["open"]:
                inner.pack(fill="x", pady=(8, 0))
            else:
                inner.pack_forget()

        def toggle():
            st["open"] = not st["open"]
            paint()

        btn.configure(command=toggle)
        btn.pack(fill="x")
        paint()
        return inner

    sec_det = section("DETECCIONES", True)
    det_card = ctk.CTkFrame(sec_det, fg_color=CARD, corner_radius=10, border_width=1, border_color="#232836")
    det_card.pack(fill="x", pady=(2, 4))
    lb = tk.Listbox(det_card, bg=CARD, fg=FG, selectbackground=ACCENT, selectforeground="white", relief="flat",
                    highlightthickness=0, bd=0, activestyle="none", font=(FONT, 10), height=7)
    lb.pack(fill="x", padx=6, pady=6)

    sec_met = section("MÉTRICAS DETALLADAS", True)
    met_container = ctk.CTkFrame(sec_met, fg_color="transparent")
    met_container.pack(fill="x")

    met_state = {
        "cache_key": None,
        "mode": "img",   # "img" o "folder"
        "dyn_widgets": {},
    }

    def metrics_text():
        parts = []
        m = S["metrics"]
        if m is not None and m["map50"] is not None and S["gt"] is not None:
            vis = visible()
            pr = precision_recall([(vis, S["gt"])], thr_var.get())
            lines = ["── IMAGEN ACTUAL ──",
                     f"{model_var.get()} · imgsz {imgsz_var.get()} · umbral {thr_var.get():.0%}",
                     f"GT: {len(S['gt'])} objetos · detectados: {len(vis)}", "",
                     f"mAP@0.5      {m['map50']:.3f}",
                     f"mAP@0.75     {m['map75']:.3f}",
                     f"mAP@0.5:0.95 {m['map']:.3f}", "",
                     f"Precisión    {pr['precision']:.3f}",
                     f"Recall       {pr['recall']:.3f}",
                     f"F1           {pr['f1']:.3f}",
                     f"TP {pr['tp']} · FP {pr['fp']} · FN {pr['fn']}", "",
                     f"{'Clase':<14}{'AP50':>7}{'AP':>7}{'GT':>4}"]
            for k, v in sorted(m["per_class"].items(), key=lambda kv: -kv[1]["ap50"]):
                lines.append(f"{k[:13]:<14}{v['ap50']:>7.3f}{v['ap']:>7.3f}{v['n_gt']:>4}")
            parts.append("\n".join(lines))
        if S.get("eval"):
            parts.append("── CARPETA COMPLETA ──\n" + format_metrics(S["eval"]))
        return "\n\n".join(parts) or ("Sin métricas todavía.\n\nCarga anotaciones con «Anotaciones…» para ver\n"
                                      "las métricas de esta imagen, o usa\n«Evaluar carpeta…» para todo el conjunto.")

    def set_metrics_box(force=False):
        m = S.get("metrics")
        eval_m = S.get("eval")
        has_img = m is not None and m.get("map50") is not None and S.get("gt") is not None
        has_eval = eval_m is not None and eval_m.get("map50") is not None

        # Determinar modo
        mode = met_state["mode"]
        if not has_img and has_eval:
            mode = "folder"
        elif not has_eval:
            mode = "img"
        met_state["mode"] = mode

        def switch_mode(new_mode):
            met_state["mode"] = new_mode
            set_metrics_box(force=True)

        gt_count = len(S["gt"]) if S.get("gt") else 0
        key = (id(m), has_eval, mode, gt_count, model_var.get(), imgsz_var.get())

        if force or key != met_state["cache_key"]:
            met_state["cache_key"] = key
            met_state["dyn_widgets"].clear()
            for w in met_container.winfo_children():
                w.destroy()

            # Estado vacío
            if not has_img and not has_eval:
                empty = ctk.CTkFrame(met_container, fg_color=CARD, corner_radius=12, border_width=1, border_color="#242a38")
                empty.pack(fill="x", pady=(2, 6))
                ctk.CTkLabel(empty, text="📊", font=(FONT, 22)).pack(pady=(16, 2))
                ctk.CTkLabel(empty, text="Sin métricas todavía", font=(FONT, 12, "bold"), text_color=FG).pack(pady=(0, 4))
                ctk.CTkLabel(empty, text="Carga anotaciones con «Anotaciones…» para calcular métricas en esta imagen, o usa «Evaluar carpeta…» para todo el conjunto.",
                             font=(FONT, 10), text_color=MUTED, wraplength=270, justify="center").pack(padx=14, pady=(0, 16))
                return

            # Selector si hay ambas fuentes (imagen actual y carpeta)
            if has_img and has_eval:
                seg = ctk.CTkSegmentedButton(met_container, values=["Imagen actual", "Carpeta completa"],
                                             command=lambda v: switch_mode("img" if v == "Imagen actual" else "folder"),
                                             selected_color=ACCENT, selected_hover_color=ACCENT_H,
                                             unselected_color=CARD, unselected_hover_color=GHOST, font=(FONT, 11))
                seg.set("Imagen actual" if mode == "img" else "Carpeta completa")
                seg.pack(fill="x", pady=(0, 8))

            # VISTA: Imagen actual
            if mode == "img" and has_img:
                vis = visible()
                pr = precision_recall([(vis, S["gt"])], thr_var.get())

                # Tarjeta Cabecera / Contexto
                hdr = ctk.CTkFrame(met_container, fg_color="#141822", corner_radius=10, border_width=1, border_color="#232836")
                hdr.pack(fill="x", pady=(0, 6))

                top_l = ctk.CTkFrame(hdr, fg_color="transparent")
                top_l.pack(fill="x", padx=10, pady=(8, 2))
                ctk.CTkLabel(top_l, text="● IMAGEN ACTUAL", font=(FONT, 10, "bold"), text_color=ACCENT).pack(side="left")
                ctk.CTkLabel(top_l, text=model_var.get(), font=(FONT, 9, "bold"), fg_color="#1f2638", text_color="#93c5fd",
                             corner_radius=6, padx=6, pady=2).pack(side="right")

                sub_lbl = ctk.CTkLabel(hdr, text=f"imgsz {imgsz_var.get()} · umbral {thr_var.get():.0%}",
                                       font=(FONT, 10), text_color=MUTED)
                sub_lbl.pack(anchor="w", padx=10, pady=(0, 2))
                met_state["dyn_widgets"]["sub_lbl"] = sub_lbl

                cnt_lbl = ctk.CTkLabel(hdr, text=f"GT: {len(S['gt'])} objetos · detectados: {len(vis)}",
                                       font=(FONT, 10, "bold"), text_color=FG)
                cnt_lbl.pack(anchor="w", padx=10, pady=(0, 8))
                met_state["dyn_widgets"]["cnt_lbl"] = cnt_lbl

                # KPI Cards (mAP)
                kpi_row = ctk.CTkFrame(met_container, fg_color="transparent")
                kpi_row.pack(fill="x", pady=(0, 6))
                kpi_row.columnconfigure((0, 1, 2), weight=1, uniform="kpi")

                kpis = [
                    ("mAP@0.5", f"{m['map50']:.3f}", "#38bdf8"),
                    ("mAP@0.75", f"{m['map75']:.3f}", "#818cf8"),
                    ("mAP@50:95", f"{m['map']:.3f}", "#c084fc")
                ]
                for col, (title, val, clr) in enumerate(kpis):
                    c = ctk.CTkFrame(kpi_row, fg_color=CARD, corner_radius=10, border_width=1, border_color="#232836")
                    c.grid(row=0, column=col, padx=(0 if col == 0 else 3, 0 if col == 2 else 3), sticky="nsew")
                    ctk.CTkLabel(c, text=title, font=(FONT, 9, "bold"), text_color=MUTED).pack(pady=(6, 1))
                    ctk.CTkLabel(c, text=val, font=(FONT, 13, "bold"), text_color=clr).pack(pady=(0, 6))

                # Tarjeta Precisión, Recall & F1
                pr_card = ctk.CTkFrame(met_container, fg_color=CARD, corner_radius=10, border_width=1, border_color="#232836")
                pr_card.pack(fill="x", pady=(0, 6))
                ctk.CTkLabel(pr_card, text="RENDIMIENTO DE DETECCIÓN", font=(FONT, 9, "bold"), text_color=MUTED).pack(anchor="w", padx=10, pady=(8, 4))

                p_grid = ctk.CTkFrame(pr_card, fg_color="transparent")
                p_grid.pack(fill="x", padx=8, pady=(0, 6))
                p_grid.columnconfigure((0, 1, 2), weight=1, uniform="pr")

                for col, (label, key, val) in enumerate([("Precisión", "p", pr["precision"]), ("Recall", "r", pr["recall"]), ("F1-Score", "f1", pr["f1"])]):
                    f = ctk.CTkFrame(p_grid, fg_color="#141722", corner_radius=8)
                    f.grid(row=0, column=col, padx=2, sticky="nsew")
                    ctk.CTkLabel(f, text=label, font=(FONT, 9), text_color=MUTED).pack(pady=(4, 0))
                    v_lbl = ctk.CTkLabel(f, text=f"{val:.3f}", font=(FONT, 12, "bold"), text_color=FG)
                    v_lbl.pack(pady=(0, 4))
                    met_state["dyn_widgets"][f"pr_{key}"] = v_lbl

                p_row = ctk.CTkFrame(pr_card, fg_color="transparent")
                p_row.pack(fill="x", padx=8, pady=(0, 8))
                p_row.columnconfigure((0, 1, 2), weight=1, uniform="pill")

                pills = [
                    ("tp", f"TP  {pr['tp']}", "#132a1e", "#4ade80"),
                    ("fp", f"FP  {pr['fp']}", "#2e171b", "#f87171"),
                    ("fn", f"FN  {pr['fn']}", "#2b2214", "#fbbf24")
                ]
                for col, (pkey, txt, bg, fg) in enumerate(pills):
                    lbl = ctk.CTkLabel(p_row, text=txt, font=(FONT, 9, "bold"), fg_color=bg, text_color=fg, corner_radius=6, height=22)
                    lbl.grid(row=0, column=col, padx=2, sticky="ew")
                    met_state["dyn_widgets"][f"pill_{pkey}"] = lbl

                # Tabla por clase
                tbl = ctk.CTkFrame(met_container, fg_color=CARD, corner_radius=10, border_width=1, border_color="#232836")
                tbl.pack(fill="x", pady=(0, 6))

                t_hdr = ctk.CTkFrame(tbl, fg_color="transparent")
                t_hdr.pack(fill="x", padx=10, pady=(8, 4))
                ctk.CTkLabel(t_hdr, text="POR CLASE", font=(FONT, 9, "bold"), text_color=MUTED).pack(side="left")
                ctk.CTkLabel(t_hdr, text=f"({len(m['per_class'])} clases)", font=(FONT, 9), text_color=MUTED).pack(side="right")

                th = ctk.CTkFrame(tbl, fg_color="#141722", corner_radius=6, height=24)
                th.pack(fill="x", padx=8, pady=(0, 4))
                th.columnconfigure(0, weight=5)
                th.columnconfigure((1, 2, 3), weight=2)

                ctk.CTkLabel(th, text="CLASE", font=(FONT, 8, "bold"), text_color=MUTED, anchor="w").grid(row=0, column=0, padx=(8, 0), sticky="w")
                ctk.CTkLabel(th, text="AP50", font=(FONT, 8, "bold"), text_color=MUTED).grid(row=0, column=1)
                ctk.CTkLabel(th, text="AP", font=(FONT, 8, "bold"), text_color=MUTED).grid(row=0, column=2)
                ctk.CTkLabel(th, text="GT", font=(FONT, 8, "bold"), text_color=MUTED).grid(row=0, column=3, padx=(0, 8))

                for k, v in sorted(m["per_class"].items(), key=lambda kv: -kv[1]["ap50"]):
                    row = ctk.CTkFrame(tbl, fg_color="transparent", height=24)
                    row.pack(fill="x", padx=8, pady=1)
                    row.columnconfigure(0, weight=5)
                    row.columnconfigure((1, 2, 3), weight=2)

                    ctk.CTkLabel(row, text=k[:14], font=(FONT, 10), text_color=FG, anchor="w").grid(row=0, column=0, padx=(8, 0), sticky="w")
                    ap50 = v["ap50"]
                    clr = "#4ade80" if ap50 >= 0.8 else ("#60a5fa" if ap50 >= 0.5 else "#fbbf24")
                    ctk.CTkLabel(row, text=f"{ap50:.3f}", font=(FONT, 10, "bold"), text_color=clr).grid(row=0, column=1)
                    ctk.CTkLabel(row, text=f"{v['ap']:.3f}", font=(FONT, 10), text_color=MUTED).grid(row=0, column=2)
                    ctk.CTkLabel(row, text=str(v["n_gt"]), font=(FONT, 10), text_color=FG).grid(row=0, column=3, padx=(0, 8))

                ctk.CTkFrame(tbl, fg_color="transparent", height=4).pack()

            # VISTA: Carpeta completa
            elif mode == "folder" and has_eval:
                hdr = ctk.CTkFrame(met_container, fg_color="#141822", corner_radius=10, border_width=1, border_color="#232836")
                hdr.pack(fill="x", pady=(0, 6))

                top_l = ctk.CTkFrame(hdr, fg_color="transparent")
                top_l.pack(fill="x", padx=10, pady=(8, 2))
                ctk.CTkLabel(top_l, text="● CARPETA COMPLETA", font=(FONT, 10, "bold"), text_color=ACCENT).pack(side="left")
                mod = eval_m.get("modelo", model_var.get())
                ctk.CTkLabel(top_l, text=mod, font=(FONT, 9, "bold"), fg_color="#1f2638", text_color="#93c5fd",
                             corner_radius=6, padx=6, pady=2).pack(side="right")

                info_txt = f"{eval_m.get('n_images', 0)} imágenes  ·  {eval_m.get('n_gt', 0)} objetos GT"
                ctk.CTkLabel(hdr, text=info_txt, font=(FONT, 10, "bold"), text_color=FG).pack(anchor="w", padx=10, pady=(0, 2))

                t = eval_m.get("tiempo_medio_s")
                fps_txt = f"Tiempo: {t:.3f} s ({1/t:.1f} FPS) · {eval_m.get('dispositivo', '?')}" if t else f"Dispositivo: {eval_m.get('dispositivo', '?')}"
                ctk.CTkLabel(hdr, text=fps_txt, font=(FONT, 9), text_color=MUTED).pack(anchor="w", padx=10, pady=(0, 8))

                # KPI Cards (mAP carpeta)
                kpi_row = ctk.CTkFrame(met_container, fg_color="transparent")
                kpi_row.pack(fill="x", pady=(0, 6))
                kpi_row.columnconfigure((0, 1, 2), weight=1, uniform="kpi")

                kpis = [
                    ("mAP@0.5", f"{eval_m['map50']:.3f}", "#38bdf8"),
                    ("mAP@0.75", f"{eval_m['map75']:.3f}", "#818cf8"),
                    ("mAP@50:95", f"{eval_m['map']:.3f}", "#c084fc")
                ]
                for col, (title, val, clr) in enumerate(kpis):
                    c = ctk.CTkFrame(kpi_row, fg_color=CARD, corner_radius=10, border_width=1, border_color="#232836")
                    c.grid(row=0, column=col, padx=(0 if col == 0 else 3, 0 if col == 2 else 3), sticky="nsew")
                    ctk.CTkLabel(c, text=title, font=(FONT, 9, "bold"), text_color=MUTED).pack(pady=(6, 1))
                    ctk.CTkLabel(c, text=val, font=(FONT, 13, "bold"), text_color=clr).pack(pady=(0, 6))

                # PR si está disponible
                pr_eval = eval_m.get("pr")
                if pr_eval:
                    pr_card = ctk.CTkFrame(met_container, fg_color=CARD, corner_radius=10, border_width=1, border_color="#232836")
                    pr_card.pack(fill="x", pady=(0, 6))
                    ctk.CTkLabel(pr_card, text=f"PRECISIÓN GLOBAL (umbral {pr_eval.get('score_thr', 0.25):.2f})",
                                 font=(FONT, 9, "bold"), text_color=MUTED).pack(anchor="w", padx=10, pady=(8, 4))

                    p_grid = ctk.CTkFrame(pr_card, fg_color="transparent")
                    p_grid.pack(fill="x", padx=8, pady=(0, 8))
                    p_grid.columnconfigure((0, 1, 2), weight=1, uniform="pr")

                    for col, (label, val) in enumerate([("Precisión", pr_eval["precision"]), ("Recall", pr_eval["recall"]), ("F1-Score", pr_eval["f1"])]):
                        f = ctk.CTkFrame(p_grid, fg_color="#141722", corner_radius=8)
                        f.grid(row=0, column=col, padx=2, sticky="nsew")
                        ctk.CTkLabel(f, text=label, font=(FONT, 9), text_color=MUTED).pack(pady=(4, 0))
                        ctk.CTkLabel(f, text=f"{val:.3f}", font=(FONT, 12, "bold"), text_color=FG).pack(pady=(0, 4))

                # Tabla por clase de la carpeta
                if eval_m.get("per_class"):
                    tbl = ctk.CTkFrame(met_container, fg_color=CARD, corner_radius=10, border_width=1, border_color="#232836")
                    tbl.pack(fill="x", pady=(0, 6))

                    t_hdr = ctk.CTkFrame(tbl, fg_color="transparent")
                    t_hdr.pack(fill="x", padx=10, pady=(8, 4))
                    ctk.CTkLabel(t_hdr, text="POR CLASE (CARPETA)", font=(FONT, 9, "bold"), text_color=MUTED).pack(side="left")
                    ctk.CTkLabel(t_hdr, text=f"({len(eval_m['per_class'])} clases)", font=(FONT, 9), text_color=MUTED).pack(side="right")

                    th = ctk.CTkFrame(tbl, fg_color="#141722", corner_radius=6, height=24)
                    th.pack(fill="x", padx=8, pady=(0, 4))
                    th.columnconfigure(0, weight=5)
                    th.columnconfigure((1, 2, 3), weight=2)

                    ctk.CTkLabel(th, text="CLASE", font=(FONT, 8, "bold"), text_color=MUTED, anchor="w").grid(row=0, column=0, padx=(8, 0), sticky="w")
                    ctk.CTkLabel(th, text="AP50", font=(FONT, 8, "bold"), text_color=MUTED).grid(row=0, column=1)
                    ctk.CTkLabel(th, text="AP", font=(FONT, 8, "bold"), text_color=MUTED).grid(row=0, column=2)
                    ctk.CTkLabel(th, text="GT", font=(FONT, 8, "bold"), text_color=MUTED).grid(row=0, column=3, padx=(0, 8))

                    for k, v in sorted(eval_m["per_class"].items(), key=lambda kv: -kv[1]["ap50"]):
                        row = ctk.CTkFrame(tbl, fg_color="transparent", height=24)
                        row.pack(fill="x", padx=8, pady=1)
                        row.columnconfigure(0, weight=5)
                        row.columnconfigure((1, 2, 3), weight=2)

                        ctk.CTkLabel(row, text=k[:14], font=(FONT, 10), text_color=FG, anchor="w").grid(row=0, column=0, padx=(8, 0), sticky="w")
                        ap50 = v["ap50"]
                        clr = "#4ade80" if ap50 >= 0.8 else ("#60a5fa" if ap50 >= 0.5 else "#fbbf24")
                        ctk.CTkLabel(row, text=f"{ap50:.3f}", font=(FONT, 10, "bold"), text_color=clr).grid(row=0, column=1)
                        ctk.CTkLabel(row, text=f"{v['ap']:.3f}", font=(FONT, 10), text_color=MUTED).grid(row=0, column=2)
                        ctk.CTkLabel(row, text=str(v["n_gt"]), font=(FONT, 10), text_color=FG).grid(row=0, column=3, padx=(0, 8))

                    ctk.CTkFrame(tbl, fg_color="transparent", height=4).pack()
        else:
            # Actualización ultra-rápida (al arrastrar el slider de umbral)
            if mode == "img" and has_img:
                vis = visible()
                pr = precision_recall([(vis, S["gt"])], thr_var.get())
                dw = met_state["dyn_widgets"]
                if "sub_lbl" in dw:
                    dw["sub_lbl"].configure(text=f"imgsz {imgsz_var.get()} · umbral {thr_var.get():.0%}")
                if "cnt_lbl" in dw:
                    dw["cnt_lbl"].configure(text=f"GT: {len(S['gt'])} objetos · detectados: {len(vis)}")
                if "pr_p" in dw:
                    dw["pr_p"].configure(text=f"{pr['precision']:.3f}")
                if "pr_r" in dw:
                    dw["pr_r"].configure(text=f"{pr['recall']:.3f}")
                if "pr_f1" in dw:
                    dw["pr_f1"].configure(text=f"{pr['f1']:.3f}")
                if "pill_tp" in dw:
                    dw["pill_tp"].configure(text=f"TP  {pr['tp']}")
                if "pill_fp" in dw:
                    dw["pill_fp"].configure(text=f"FP  {pr['fp']}")
                if "pill_fn" in dw:
                    dw["pill_fn"].configure(text=f"FN  {pr['fn']}")

    def copy_metrics():
        root.clipboard_clear()
        root.clipboard_append(metrics_text())
        status.set("Métricas copiadas al portapapeles")

    gbtn(sec_met, "📋  Copiar reporte de métricas", copy_metrics).pack(fill="x", pady=(8, 0))
    ctk.CTkFrame(side, fg_color="transparent", height=16).pack()

    # ---------- barra inferior ----------
    foot = ctk.CTkFrame(root, fg_color="transparent")
    foot.pack(fill="x", padx=20, pady=(10, 12))
    prog = ctk.CTkProgressBar(foot, height=6, progress_color=ACCENT, fg_color=CARD)
    prog.pack(fill="x")
    prog.set(0)
    ctk.CTkLabel(foot, textvariable=status, font=(FONT, 11), text_color=MUTED, anchor="w").pack(fill="x", pady=(6, 0))

    def prog_busy(on: bool):
        if on:
            prog.configure(mode="indeterminate")
            prog.start()
        else:
            prog.stop()
            prog.configure(mode="determinate")
            prog.set(0)

    # ---------- lógica ----------
    def visible():
        cl = {c.strip() for c in cls_var.get().split(",") if c.strip()}
        return [d for d in S["raw"] if d["score"] >= thr_var.get() and (not cl or d["label"] in cl)]

    def render():
        canvas.delete("all")
        img, cw, ch = S["image"], canvas.winfo_width(), canvas.winfo_height()
        if img is None:
            canvas.create_text(cw / 2, ch / 2, fill=MUTED, font=(FONT, 14), justify="center",
                               text="Haz clic aquí o pulsa Ctrl + O\npara abrir una imagen")
            return
        if cw < 40 or ch < 40:
            return
        s = min((cw - 16) / img.width, (ch - 16) / img.height)
        base = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)
        if show_gt.get() and S["gt"]:
            dg = ImageDraw.Draw(base)
            for g in S["gt"]:
                dg.rectangle([v * s for v in g["box"]], outline=(255, 255, 255), width=1)
        vis = visible()
        sc = [{**d, "box": [v * s for v in d["box"]]} for d in vis]
        hl = next((i for i, d in enumerate(vis) if d.get("id") == S["selected"]), None)
        S["photo"] = ImageTk.PhotoImage(draw_detections(base, sc, hl) if sc else base)
        canvas.create_image(cw // 2, ch // 2, image=S["photo"])

    def refresh(*_):
        render()
        vis = visible()
        lb.delete(0, "end")
        for d in vis:
            lb.insert("end", f"  ●  {d['label']:<16} {d['score']:.0%}")
        if S["meta"]:
            t_var.set(f"{S['meta'][0]:.2f} s")
            d_var.set(S["meta"][1].upper())
            n_var.set(str(len(vis)))
        m = S["metrics"]
        m50_var.set(f"{m['map50']:.1%}" if m and m["map50"] is not None else "—")
        m95_var.set(f"{m['map']:.1%}" if m and m["map"] is not None else "—")
        if S["gt"] is not None and S["raw"]:
            pr = precision_recall([(vis, S["gt"])], thr_var.get())
            pr_var.set(f"Precisión {pr['precision']:.0%} · Recall {pr['recall']:.0%} · F1 {pr['f1']:.0%}")
        else:
            pr_var.set("")
        set_metrics_box()

    def update_metrics():
        S["metrics"] = evaluate_map([(S["raw"], S["gt"])]) if S["gt"] is not None and S["raw"] else None

    def open_image(path=None):
        if not isinstance(path, str):
            path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return
        S.update(path=path, image=img, raw=[], meta=None, gt=None, metrics=None, selected=None)
        if S["gt_obj"] is not None:
            try:
                S["gt"] = S["gt_obj"].for_image(path, img.size)
            except Exception:
                S["gt"] = None
        file_var.set(f"{Path(path).name}  ·  {img.width}×{img.height}")
        refresh()
        detect()

    def tick():
        if S["busy"]:
            t_var.set(f"{time.perf_counter() - S['t0']:.1f} s")
            root.after(100, tick)

    def detect(*_):
        if S["image"] is None or S["busy"]:
            return
        model, img, sz = model_var.get(), S["image"], int(imgsz_var.get())
        S.update(busy=True, raw=[], meta=None, metrics=None, t0=time.perf_counter())
        tick()
        prog_busy(True)
        status.set("Analizando… (la primera vez se descarga el modelo)")

        def work():
            try:
                det = get_detector(model, args.device)
                det.imgsz = sz
                if not getattr(det, "_warm", False):
                    det.detect(Image.new("RGB", (64, 64)), 0.5)
                    det._warm = True
                t0 = time.perf_counter()
                dets = det.detect(img, MIN_THR)
                jobs.put(lambda: done(img, dets, time.perf_counter() - t0, str(det.device)))
            except Exception as e:
                jobs.put(lambda err=e: fail(err))

        threading.Thread(target=work, daemon=True).start()

    def done(img, dets, dt, dev):
        prog_busy(False)
        S["busy"] = False
        if img is not S["image"]:
            return detect()
        for i, d in enumerate(dets):
            d["id"] = i
        S.update(raw=dets, meta=(dt, dev))
        update_metrics()
        refresh()
        status.set(f"{len(visible())} objetos visibles de {len(dets)} · {model_var.get()} · "
                   f"imgsz {imgsz_var.get()} · {dt:.3f} s · {dev}")

    def fail(err):
        prog_busy(False)
        S["busy"] = False
        status.set("Error")
        messagebox.showerror("Error", str(err))

    def save_image():
        if S["raw"]:
            p = filedialog.asksaveasfilename(defaultextension=".jpg", initialfile=f"{Path(S['path']).stem}_det.jpg")
            if p:
                draw_detections(S["image"], visible()).save(p)
                status.set(f"Guardado en {p}")

    def export_json():
        if S["raw"]:
            p = filedialog.asksaveasfilename(defaultextension=".json", initialfile=f"{Path(S['path']).stem}_det.json")
            if p:
                data = {"imagen": S["path"], "modelo": model_var.get(), "imgsz": int(imgsz_var.get()),
                        "umbral": round(thr_var.get(), 2),
                        "tiempo_deteccion_s": round(S["meta"][0], 3), "dispositivo": S["meta"][1],
                        "detecciones": [{k: v for k, v in d.items() if k != "id"} for d in visible()]}
                if S["metrics"]:
                    data["metricas"] = S["metrics"]
                Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                status.set(f"Guardado en {p}")

    def load_gt(path=None):
        if not isinstance(path, str):
            path = filedialog.askopenfilename(title="Anotaciones (COCO JSON / VOC XML / YOLO TXT)",
                                              filetypes=[("Anotaciones", "*.json *.xml *.txt"), ("Todos", "*.*")])
        if not path:
            return
        try:
            g = GroundTruth(path)
            if S["image"] is None:
                if not g.is_dataset:
                    raise ValueError("Abre primero una imagen.")
                S["gt_obj"] = g
                return status.set("Anotaciones cargadas: se aplicarán al abrir cada imagen")
            gts = g.for_image(S["path"], S["image"].size)
            if gts is None:
                raise ValueError("Esta imagen no aparece en las anotaciones.")
        except Exception as e:
            return messagebox.showerror("Anotaciones", str(e))
        S["gt_obj"] = g if g.is_dataset else None
        S["gt"] = gts
        update_metrics()
        refresh()
        status.set(f"{len(gts)} objetos anotados cargados desde {Path(path).name}")

    def text_window(title, content, size="660x720", export_data=None):
        w = ctk.CTkToplevel(root)
        w.title(title)
        w.geometry(size)
        w.configure(fg_color=PANEL)
        txt = ctk.CTkTextbox(w, fg_color=BG, text_color=FG, font=("Consolas", 12), corner_radius=12)
        txt.pack(fill="both", expand=True, padx=14, pady=(14, 8))
        txt.insert("end", content)
        txt.configure(state="disabled")
        if export_data is not None:
            def export():
                p = filedialog.asksaveasfilename(parent=w, defaultextension=".json", initialfile="metricas.json")
                if p:
                    Path(p).write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")
            gbtn(w, "Exportar JSON", export).pack(pady=(0, 14))
        w.after(150, w.lift)

    def eval_folder():
        folder = filedialog.askdirectory(title="Carpeta de imágenes")
        if not folder:
            return
        images = collect_images([folder])
        if not images:
            return messagebox.showerror("Imágenes", "La carpeta no tiene imágenes.")
        if messagebox.askyesno("Anotaciones", "¿Las anotaciones están en un archivo JSON COCO?\n(No = carpeta con .txt/.xml)"):
            ann = filedialog.askopenfilename(filetypes=[("COCO JSON", "*.json")])
        else:
            ann = filedialog.askdirectory(title="Carpeta de anotaciones")
        if not ann:
            return
        try:
            g = GroundTruth(ann)
        except Exception as e:
            return messagebox.showerror("Anotaciones", str(e))
        model, thr, sz = model_var.get(), thr_var.get(), int(imgsz_var.get())
        prog_busy(False)

        def work():
            try:
                det = get_detector(model, args.device)
                det.imgsz = sz
                m = evaluate_dataset(det, images, g, score_thr=thr,
                                     progress=lambda i, n: jobs.put(lambda: (prog.set(i / n),
                                                                             status.set(f"Evaluando {i}/{n}…"))))
                m["modelo"], m["dispositivo"], m["imgsz"] = model, str(det.device), sz
                jobs.put(lambda: show_eval(m))
            except Exception as e:
                jobs.put(lambda err=e: fail(err))

        threading.Thread(target=work, daemon=True).start()

    def show_eval(m):
        prog.set(0)
        S["eval"] = m
        refresh()
        status.set("Evaluación terminada")
        text_window("Evaluación · mAP", format_metrics(m), export_data=m)

    def compare_models():
        if S["image"] is None:
            return messagebox.showinfo("Comparar", "Abre una imagen primero.")
        if S["busy"]:
            return
        img, thr, sz = S["image"], thr_var.get(), int(imgsz_var.get())
        S["busy"] = True
        prog_busy(True)
        status.set("Comparando modelos… (la primera vez se descargan los pesos)")

        def work():
            rows = []
            try:
                for name in COMPARE_MODELS:
                    jobs.put(lambda n=name: status.set(f"Probando {n}…"))
                    det = get_detector(name, args.device)
                    det.imgsz = sz
                    det.detect(Image.new("RGB", (64, 64)), 0.5)          # calentamiento
                    t0 = time.perf_counter()
                    dets = det.detect(img, thr)
                    dt = time.perf_counter() - t0
                    rows.append((name, len(dets), sum(d["score"] for d in dets) / len(dets) if dets else 0.0, dt))
                jobs.put(lambda: show_compare(rows, thr, sz))
            except Exception as e:
                jobs.put(lambda err=e: fail(err))

        threading.Thread(target=work, daemon=True).start()

    def show_compare(rows, thr, sz):
        prog_busy(False)
        S["busy"] = False
        status.set("Comparación terminada")
        txt = (f"Umbral {thr:.0%} · imgsz {sz} · imagen: {Path(S['path']).name}\n\n"
               f"{'Modelo':<14}{'Objetos':>9}{'Conf. media':>13}{'Tiempo (s)':>12}\n" + "─" * 48 + "\n")
        for name, n, c, t in rows:
            txt += f"{name:<14}{n:>9}{c:>13.2f}{t:>12.3f}\n"
        text_window("Comparación de modelos YOLO", txt, "560x300",
                    export_data={"umbral": thr, "imgsz": sz,
                                 "resultados": [{"modelo": a, "objetos": b, "conf_media": c, "tiempo_s": d}
                                                for a, b, c, d in rows]})

    # ---------- botones ----------
    gbtn(head, "Exportar JSON", export_json).pack(side="right", padx=(8, 0))
    gbtn(head, "Guardar imagen", save_image).pack(side="right", padx=(8, 0))
    gbtn(head, "Abrir imagen", open_image, accent=True).pack(side="right", padx=(8, 0))


    # ---------- eventos ----------
    ent.bind("<KeyRelease>", refresh)
    ent.bind("<Return>", refresh)

    def on_select(_):
        sel = lb.curselection()
        vis = visible()
        if sel and sel[0] < len(vis):
            S["selected"] = vis[sel[0]].get("id")
        render()

    lb.bind("<<ListboxSelect>>", on_select)
    canvas.bind("<Configure>", lambda e: render())
    canvas.bind("<Button-1>", lambda e: S["image"] is None and open_image())
    root.bind("<Control-o>", lambda e: open_image())
    root.bind("<F5>", lambda e: detect())

    def poll():
        try:
            while True:
                jobs.get_nowait()()
        except queue.Empty:
            pass
        root.after(80, poll)

    poll()
    refresh()
    if args.inputs:
        root.after(300, lambda: (open_image(args.inputs[0]), args.gt and load_gt(args.gt)))
    root.mainloop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Detección de objetos con YOLO (Ultralytics YOLO11 / YOLO12)")
    ap.add_argument("inputs", nargs="*", help="imágenes o carpetas")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("-s", "--score", type=float, default=DEFAULT_SCORE, help="umbral de confianza")
    ap.add_argument("-c", "--classes", nargs="+", help="solo estas clases (p.ej. person car)")
    ap.add_argument("-o", "--output", default="salida", help="carpeta de salida")
    ap.add_argument("--imgsz", type=int, default=640, help="tamaño de entrada de YOLO (640, 960, 1280)")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (auto por defecto)")
    ap.add_argument("--json", action="store_true", help="guardar detecciones en JSON")
    ap.add_argument("--cli", action="store_true", help="modo consola (sin ventana)")
    ap.add_argument("--gt", help="anotaciones reales para calcular mAP: JSON COCO, carpeta o archivo VOC/YOLO")
    ap.add_argument("--web", action="store_true", help="interfaz web (Gradio)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for k, desc in MODELS.items():
            print(f"{k:16s} {desc}")
        return 0
    if args.web:
        return run_web(args)
    if args.cli:
        if not args.inputs:
            ap.error("--cli requiere al menos una imagen o carpeta")
        return run_cli(args)
    return run_gui(args)


if __name__ == "__main__":
    sys.exit(main())