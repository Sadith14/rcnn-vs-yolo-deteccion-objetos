#!/usr/bin/env python3
"""
Detección de objetos con la familia R-CNN (Faster R-CNN y Mask R-CNN) usando
torchvision con pesos preentrenados en COCO (91 ids / 80 clases reales).

Uso rápido
----------
  python rcnn_detector.py                  # ventana de escritorio (por defecto)
  python rcnn_detector.py imagen.jpg       # ventana con la imagen ya cargada
  python rcnn_detector.py carpeta/ --cli --score 0.6          # modo consola, lote
  python rcnn_detector.py foto.jpg --cli --classes person car --json
  python rcnn_detector.py --web            # interfaz web con Gradio
  python rcnn_detector.py --list-models
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms import functional as TF

# nombre -> (constructor, pesos, descripción)
MODELS = {
    "fasterrcnn_resnet50_fpn_v2": (
        fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
        "Faster R-CNN R50-FPN v2 (más preciso, COCO box AP 46.7)"),
    "fasterrcnn_resnet50_fpn": (
        fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights,
        "Faster R-CNN R50-FPN v1 (clásico, box AP 37.0)"),
    "fasterrcnn_mobilenet_v3_large_fpn": (
        fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        "Faster R-CNN MobileNetV3-FPN (rápido, box AP 32.8)"),
    "fasterrcnn_mobilenet_v3_large_320_fpn": (
        fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        "Faster R-CNN MobileNetV3-320 (muy ligero, ideal CPU)"),
    "maskrcnn_resnet50_fpn_v2": (
        maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights,
        "Mask R-CNN R50-FPN v2 (se usan solo las cajas)"),
}
DEFAULT_MODEL = "fasterrcnn_resnet50_fpn_v2"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIN_THR = 0.05   # umbral mínimo de detección: el mAP necesita toda la curva precisión-recall
COCO_NAMES = [c for c in FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
              if c not in ("__background__", "N/A")]     # las 80 clases COCO en orden estándar
ALL_CLASSES = sorted(COCO_NAMES)

_INFER_LOCK = threading.Lock()   # evita inferencias simultáneas sobre el mismo modelo


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class RCNNDetector:
    """Envoltura sencilla alrededor de un modelo R-CNN de torchvision."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 score_thr: float = 0.5, nms_iou: float = 0.5):
        if model_name not in MODELS:
            raise ValueError(f"Modelo desconocido: {model_name}. Opciones: {list(MODELS)}")
        builder, weights_enum, _ = MODELS[model_name]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.weights = weights_enum.DEFAULT
        self.categories: list[str] = self.weights.meta["categories"]
        self.model = builder(weights=self.weights, box_score_thresh=score_thr,
                             box_nms_thresh=nms_iou)
        self.model.eval().to(self.device)
        self.model_name = model_name

    @torch.inference_mode()
    def detect(self, image: Image.Image, score_thr: float | None = None,
               classes: set[str] | None = None) -> list[dict]:
        """Devuelve [{label, score, box:[x1,y1,x2,y2]}, ...] ordenado por score."""
        with _INFER_LOCK:
            if score_thr is not None:
                self.model.roi_heads.score_thresh = score_thr
            tensor = TF.to_tensor(image.convert("RGB")).to(self.device)
            out = self.model([tensor])[0]

        results = []
        for box, label, score in zip(out["boxes"].cpu().tolist(),
                                     out["labels"].cpu().tolist(),
                                     out["scores"].cpu().tolist()):
            name = self.categories[label]
            if classes and name not in classes:
                continue
            results.append({"label": name, "score": round(score, 4),
                            "box": [round(v, 1) for v in box]})
        return sorted(results, key=lambda d: -d["score"])


@lru_cache(maxsize=8)
def get_detector(model_name: str, device: str | None) -> RCNNDetector:
    return RCNNDetector(model_name, device)


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
            draw.rectangle([x1 - 3, y1 - 3, x2 + 3, y2 + 3], outline=(255, 255, 255),
                           width=thickness + 2)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)
        text = f'{d["label"]} {d["score"]:.0%}'
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        tw, th = r - l, b - t
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y1
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 2 - t), text, fill=(0, 0, 0), font=font)
    return img


# --------------------------------------------------------------------------- #
# Métricas: mAP (estilo COCO), precisión y recall
# --------------------------------------------------------------------------- #
IOU_THRS = [round(0.5 + 0.05 * i, 2) for i in range(10)]   # 0.50 : 0.05 : 0.95
_NAME_MAP = {c.lower().replace("_", " "): c for c in COCO_NAMES}


def _norm_label(name: str) -> str:
    """Normaliza el nombre de una clase anotada a la nomenclatura COCO del modelo."""
    name = str(name).strip()
    return _NAME_MAP.get(name.lower().replace("_", " "), name)


def _iou(a, b):
    """Matriz de IoU entre cajas [x1,y1,x2,y2]: (len(a), len(b))."""
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
    """Asignación voraz estilo COCO (detecciones ya ordenadas por score desc).
    Devuelve, por detección, True si es verdadero positivo."""
    n_d, n_g = ious.shape
    taken = [False] * n_g
    flags = []
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
    """AP con interpolación de 101 puntos (definición COCO)."""
    import numpy as np
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")
    tp = np.asarray(tps, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = ctp / (ctp + cfp + np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]      # envolvente monótona
    idx = np.searchsorted(recall, rec_thrs, side="left")
    q = np.zeros(len(rec_thrs))
    ok = idx < len(precision)
    q[ok] = precision[idx[ok]]
    return float(q.mean())


def evaluate_map(samples: list) -> dict:
    """mAP sobre un conjunto de imágenes.

    samples: lista de (predicciones, anotaciones) por imagen, con
      predicciones = [{label, score, box}] (usar un umbral bajo, p. ej. 0.05)
      anotaciones  = [{label, box}]
    Devuelve mAP@0.5, mAP@0.75, mAP@[0.5:0.95] y el AP por clase.
    Solo cuentan las clases que tienen al menos un objeto anotado.
    """
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

    return {"map50": mean("ap50"), "map75": mean("ap75"), "map": mean("ap"),
            "per_class": per_class, "n_images": len(samples),
            "n_gt": sum(v["n_gt"] for v in per_class.values()),
            "unsupported": [c for c in classes if c not in COCO_NAMES]}


def precision_recall(samples: list, score_thr: float = 0.5, iou_thr: float = 0.5) -> dict:
    """Precisión, recall y F1 para un umbral de confianza y un IoU dados."""
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
    """Anotaciones reales. Formatos: JSON COCO, XML Pascal VOC o TXT YOLO.

    `source` puede ser un .json (COCO, con todas las imágenes), un archivo .xml/.txt
    (una sola imagen) o una carpeta con un .xml/.txt por imagen (mismo nombre que la imagen).
    En YOLO los nombres de clase se leen de classes.txt / obj.names (en la misma carpeta o en
    la superior); si no existe se asume el orden estándar de COCO (80 clases).
    """

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
        """True si cubre varias imágenes (JSON COCO o carpeta)."""
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
            label = _norm_label(cats.get(a["category_id"], a["category_id"]))
            self.coco[n].append({"label": label, "box": [x, y, x + w, y + h]})

    def for_image(self, image_path, size=None):
        """Anotaciones de una imagen, o None si no hay anotación para ella."""
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
            if bb is None:
                continue
            box = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            out.append({"label": _norm_label(obj.findtext("name", "")), "box": box})
        return out

    def _class_names(self, txt: Path) -> list[str]:
        if self.names is None:
            folders = [txt.parent, txt.parent.parent] + ([self.src] if self.src.is_dir() else [])
            for d in folders:
                for fn in ("classes.txt", "obj.names", "classes.names", "labels.txt"):
                    if (d / fn).is_file():
                        lines = (d / fn).read_text(encoding="utf-8").splitlines()
                        self.names = [ln.strip() for ln in lines if ln.strip()]
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
            out.append({"label": label,
                        "box": [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]})
        return out


def evaluate_dataset(detector, images, gt: "GroundTruth", classes=None, score_thr: float = 0.5,
                     progress=None, cancel=None) -> dict:
    """Detecta en cada imagen y calcula mAP contra las anotaciones.
    Las imágenes sin anotación se omiten (se cuentan en `skipped`)."""
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


def compare_models(image, gt=None, score_thr=0.5, classes=None, models=None, device=None,
                   progress=None, on_row=None, cancel=None) -> list[dict]:
    """Ejecuta todos los modelos sobre una imagen y devuelve una fila de métricas por modelo.
    gt: lista de anotaciones [{label, box}] de ESA imagen (o None)."""
    names = list(models or MODELS)
    rows = []
    for i, name in enumerate(names, 1):
        if cancel and cancel():
            break
        if progress:
            progress(i, len(names), name)
        det = get_detector(name, device)
        if det.device.type == "cuda":                    # calentamiento: no penalizar al primero
            det.detect(image.resize((64, 64)), MIN_THR)
        t0 = time.perf_counter()
        raw = det.detect(image, MIN_THR)
        dt = time.perf_counter() - t0

        raw_f = [d for d in raw if d["label"] in classes] if classes else raw
        vis = [d for d in raw_f if d["score"] >= score_thr]
        row = {"modelo": name, "tiempo_s": dt, "fps": 1 / dt if dt else None,
               "n_det": len(vis),
               "score_medio": sum(d["score"] for d in vis) / len(vis) if vis else None,
               "raw": raw, "device": str(det.device)}
        if gt is not None:
            gts = [g for g in gt if not classes or g["label"] in classes]
            s = [(raw_f, gts)]
            m, pr = evaluate_map(s), precision_recall(s, score_thr)
            row.update(map50=m["map50"], map=m["map"], precision=pr["precision"],
                       recall=pr["recall"], f1=pr["f1"])
        rows.append(row)
        if on_row:
            on_row(row)
    return rows


def format_metrics(m: dict) -> str:
    """Resumen de métricas en texto plano (para la consola)."""
    if m["map50"] is None:
        return "No hay objetos anotados para evaluar."
    lines = [f"Imágenes evaluadas: {m['n_images']}   ·   Objetos anotados (GT): {m['n_gt']}",
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
    print(f"Dispositivo: {det.device}")
    classes = set(args.classes) if args.classes else None

    gt = GroundTruth(args.gt) if args.gt else None
    if gt is not None and not gt.is_dataset:
        print("--gt en modo consola requiere un JSON COCO o una carpeta de anotaciones.",
              file=sys.stderr)
        return 1
    samples, times, skipped = [], [], 0

    all_results = {}
    for f in files:
        img = Image.open(f)
        t0 = time.perf_counter()
        if gt is None:
            dets = det.detect(img, args.score, classes)
        else:   # para el mAP se necesita el umbral mínimo; luego se filtra para mostrar
            raw = det.detect(img, MIN_THR)
            dets = [d for d in raw if d["score"] >= args.score
                    and (not classes or d["label"] in classes)]
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
        all_results[str(f)] = {"tiempo_s": round(dt, 3), "dispositivo": str(det.device),
                               "detecciones": dets}
        resumen = ", ".join(sorted({d["label"] for d in dets})) or "nada"
        print(f"{f.name}: {len(dets)} objetos ({resumen}) · {dt:.2f}s → {dest}")

    if gt is not None:
        m = evaluate_map(samples)
        m["pr"] = precision_recall(samples, args.score)
        m["skipped"] = skipped
        m["tiempo_medio_s"] = sum(times) / len(times) if times else None
        print("\n" + format_metrics(m))
        if skipped:
            print(f"({skipped} imágenes sin anotación, omitidas)")
        mpath = out_dir / "metricas.json"
        mpath.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Métricas guardadas en {mpath}")

    if args.json:
        jpath = out_dir / "detecciones.json"
        jpath.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"JSON guardado en {jpath}")
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

    def predict(image, model_name, thr, class_text):
        if image is None:
            return None, []
        classes = {c.strip() for c in class_text.split(",") if c.strip()} or None
        det = get_detector(model_name, args.device)
        pil = Image.fromarray(image)
        dets = det.detect(pil, thr, classes)
        rows = [[d["label"], f'{d["score"]:.3f}', *map(int, d["box"])] for d in dets]
        return draw_detections(pil, dets), rows

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="numpy", label="Imagen"),
            gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Modelo"),
            gr.Slider(0.05, 0.99, value=0.5, step=0.05, label="Umbral de confianza"),
            gr.Textbox(label="Filtrar clases (separadas por coma, opcional)",
                       placeholder="person, car, dog"),
        ],
        outputs=[gr.Image(label="Resultado"),
                 gr.Dataframe(headers=["clase", "score", "x1", "y1", "x2", "y2"],
                              label="Detecciones")],
        title="Detección de objetos con R-CNN",
        description="Faster R-CNN / Mask R-CNN preentrenados en COCO (torchvision).",
        flagging_mode="never",
    )
    demo.launch(server_port=args.port)
    return 0

# --------------------------------------------------------------------------- #
# Ventana de escritorio (Tkinter) - tema oscuro minimalista
# --------------------------------------------------------------------------- #
def run_gui(args) -> int:
    import queue
    import threading
    import tkinter as tk
    import tkinter.font as tkfont
    from collections import Counter
    from tkinter import filedialog, messagebox, ttk

    from PIL import ImageTk

    try:  # nitidez en Windows con pantallas HiDPI
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    BG, PANEL, LINE, TRACK = "#0e1014", "#161920", "#262b36", "#272c37"
    FG, MUTED, ACCENT, ACCENT_H, SEL = "#e8eaee", "#8a92a1", "#4f8cff", "#6aa0ff", "#1e2430"

    root = tk.Tk()
    root.title("R-CNN · Detección de objetos")
    root.geometry("1260x860")
    root.minsize(980, 700)
    root.configure(bg=BG)

    fams = set(tkfont.families())
    FAMILY = next((f for f in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu",
                               "DejaVu Sans") if f in fams), "Helvetica")

    def F(size, bold=False):
        return (FAMILY, size, "bold") if bold else (FAMILY, size)

    def hexcolor(label):
        return "#%02x%02x%02x" % _color_for(label)

    # ---- estilos ----
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=BG, foreground=FG, font=F(10), borderwidth=0, focuscolor=BG)
    st.configure("TLabel", background=BG, foreground=FG)
    st.configure("Muted.TLabel", foreground=MUTED)
    st.configure("Cap.TLabel", background=PANEL, foreground=MUTED, font=F(8, True))
    st.configure("Side.TLabel", background=PANEL, foreground=FG)
    st.configure("SideMuted.TLabel", background=PANEL, foreground=MUTED, font=F(9))
    st.configure("TButton", background="#20242d", foreground=FG, padding=(14, 8), relief="flat")
    st.map("TButton", background=[("active", "#2a2f3b"), ("disabled", "#181b22")],
           foreground=[("disabled", "#566070")])
    st.configure("Accent.TButton", background=ACCENT, foreground="white", font=F(10, True))
    st.map("Accent.TButton", background=[("active", ACCENT_H), ("disabled", "#26344f")],
           foreground=[("disabled", "#7d8aa6")])
    st.configure("TCombobox", fieldbackground=BG, background="#20242d", foreground=FG,
                 arrowcolor=MUTED, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, padding=6)
    st.map("TCombobox", fieldbackground=[("readonly", BG)], foreground=[("readonly", FG)],
           selectbackground=[("readonly", BG)], selectforeground=[("readonly", FG)])
    root.option_add("*TCombobox*Listbox.background", BG)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "white")
    st.configure("Horizontal.TScale", background=ACCENT, troughcolor=TRACK, bordercolor=PANEL,
                 lightcolor=ACCENT, darkcolor=ACCENT)
    st.configure("Thin.Horizontal.TProgressbar", troughcolor=BG, background=ACCENT,
                 bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT, thickness=3)
    st.configure("Side.TCheckbutton", background=PANEL, foreground=MUTED, font=F(9))
    st.map("Side.TCheckbutton", background=[("active", PANEL)], foreground=[("active", FG)])
    st.configure("Treeview", background=BG, fieldbackground=BG, foreground=FG,
                 rowheight=30, borderwidth=0, font=F(10))
    st.configure("Treeview.Heading", background=PANEL, foreground=MUTED, font=F(9, True),
                 relief="flat", padding=6)
    st.map("Treeview", background=[("selected", ACCENT)], foreground=[("selected", "white")])
    st.map("Treeview.Heading", background=[("active", PANEL)])

    # ---- estado ----
    state = dict(path=None, image=None, raw=[], classes=set(), selected=None, photo=None,
                 view_key=None, base=None, busy=False, meta=None, gt=None, gt_obj=None,
                 metrics=None)
    show_gt = tk.BooleanVar(value=True)
    auto_cmp = tk.BooleanVar(value=False)   # pon True para comparar siempre al abrir una imagen
    cmp = {"busy": False, "dlg": None, "tree": None, "msg": None, "pbar": None,
           "rows": [], "cancel": False, "img": None}
    model_var = tk.StringVar(value=DEFAULT_MODEL)
    thr_var = tk.DoubleVar(value=0.5)
    status = tk.StringVar(value="Abre una imagen para empezar")
    file_var = tk.StringVar(value="")
    classes_var = tk.StringVar(value="Todas")
    thr_text = tk.StringVar(value="50%")
    desc_var = tk.StringVar(value=MODELS[DEFAULT_MODEL][2])
    jobs: "queue.Queue" = queue.Queue()
    pending = {"id": None}

    # ---- cabecera ----
    header = tk.Frame(root, bg=BG)
    header.pack(fill="x", padx=20, pady=(14, 10))
    tk.Label(header, text="◉  R-CNN", bg=BG, fg=FG, font=F(14, True)).pack(side="left")
    ttk.Label(header, textvariable=file_var, style="Muted.TLabel").pack(side="left", padx=18)
    btn_json = ttk.Button(header, text="Exportar JSON", command=lambda: export_json())
    btn_json.pack(side="right")
    btn_save = ttk.Button(header, text="Guardar imagen", command=lambda: save_image())
    btn_save.pack(side="right", padx=8)
    btn_open = ttk.Button(header, text="Abrir imagen", style="Accent.TButton",
                          command=lambda: open_image())
    btn_open.pack(side="right")

    # ---- cuerpo ----
    body = tk.Frame(root, bg=BG)
    body.pack(fill="both", expand=True, padx=20)

    canvas = tk.Canvas(body, bg="#0a0c10", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    side = tk.Frame(body, bg=PANEL, width=340)
    side.pack(side="right", fill="y", padx=(14, 0))
    side.pack_propagate(False)

    def caption(text, top=16):
        ttk.Label(side, text=text, style="Cap.TLabel").pack(anchor="w", padx=18, pady=(top, 6))

    caption("MODELO", 18)
    combo = ttk.Combobox(side, textvariable=model_var, values=list(MODELS), state="readonly")
    combo.pack(fill="x", padx=18)
    ttk.Label(side, textvariable=desc_var, style="SideMuted.TLabel", wraplength=300,
              justify="left").pack(anchor="w", padx=18, pady=(6, 0))

    caption("CONFIANZA MÍNIMA")
    row = tk.Frame(side, bg=PANEL)
    row.pack(fill="x", padx=18)
    ttk.Scale(row, from_=MIN_THR, to=0.95, variable=thr_var,
              command=lambda v: (thr_text.set(f"{float(v):.0%}"), schedule())
              ).pack(side="left", fill="x", expand=True)
    ttk.Label(row, textvariable=thr_text, style="Side.TLabel", width=5,
              anchor="e").pack(side="right")

    caption("CLASES")
    row2 = tk.Frame(side, bg=PANEL)
    row2.pack(fill="x", padx=18)
    ttk.Button(row2, text="Filtrar…", command=lambda: class_dialog()).pack(side="left")
    ttk.Label(row2, textvariable=classes_var, style="SideMuted.TLabel").pack(side="left", padx=10)

    btn_detect = ttk.Button(side, text="Detectar", style="Accent.TButton", state="disabled",
                            command=lambda: detect())
    btn_detect.pack(fill="x", padx=18, pady=(20, 8))

    cmp_row = tk.Frame(side, bg=PANEL)
    cmp_row.pack(fill="x", padx=18, pady=(0, 12))
    btn_cmp = ttk.Button(cmp_row, text="Comparar modelos", command=lambda: compare_all())
    btn_cmp.pack(side="left")
    ttk.Checkbutton(cmp_row, text="Auto al abrir", variable=auto_cmp,
                    style="Side.TCheckbutton").pack(side="right")

    time_var, count_var, dev_var = tk.StringVar(value="—"), tk.StringVar(value="—"), \
        tk.StringVar(value="—")
    stats = tk.Frame(side, bg=PANEL)
    stats.pack(fill="x", padx=18, pady=(0, 14))
    for i, (cap, var) in enumerate((("TIEMPO", time_var), ("OBJETOS", count_var),
                                    ("DISPOSITIVO", dev_var))):
        stats.columnconfigure(i, weight=1, uniform="stat")
        card = tk.Frame(stats, bg="#1b1f28")
        card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
        tk.Label(card, textvariable=var, bg="#1b1f28", fg=FG,
                 font=F(13, True)).pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(card, text=cap, bg="#1b1f28", fg=MUTED,
                 font=F(7, True)).pack(anchor="w", padx=10, pady=(0, 8))

    # ---- evaluación mAP ----
    map50_var, map_var = tk.StringVar(value="—"), tk.StringVar(value="—")
    head = tk.Frame(side, bg=PANEL)
    head.pack(fill="x", padx=18, pady=(0, 6))
    ttk.Label(head, text="EVALUACIÓN (mAP)", style="Cap.TLabel").pack(side="left")
    ttk.Checkbutton(head, text="Ver GT", variable=show_gt, style="Side.TCheckbutton",
                    command=lambda: schedule(1)).pack(side="right")
    mrow = tk.Frame(side, bg=PANEL)
    mrow.pack(fill="x", padx=18)
    for i, (cap, var) in enumerate((("mAP@50", map50_var), ("mAP@50:95", map_var))):
        mrow.columnconfigure(i, weight=1, uniform="map")
        card = tk.Frame(mrow, bg="#1b1f28")
        card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
        tk.Label(card, textvariable=var, bg="#1b1f28", fg=FG,
                 font=F(13, True)).pack(anchor="w", padx=10, pady=(8, 0))
        tk.Label(card, text=cap, bg="#1b1f28", fg=MUTED,
                 font=F(7, True)).pack(anchor="w", padx=10, pady=(0, 8))
    brow = tk.Frame(side, bg=PANEL)
    brow.pack(fill="x", padx=18, pady=(8, 14))
    ttk.Button(brow, text="Anotaciones…", command=lambda: load_gt()).pack(
        side="left", fill="x", expand=True)
    ttk.Button(brow, text="Evaluar carpeta…", command=lambda: eval_dialog()).pack(
        side="left", fill="x", expand=True, padx=(6, 0))

    tk.Frame(side, bg=LINE, height=1).pack(fill="x")
    panel = tk.Canvas(side, bg=PANEL, highlightthickness=0, yscrollincrement=20)
    panel.pack(fill="both", expand=True)

    # ---- pie ----
    footer = tk.Frame(root, bg=BG)
    footer.pack(fill="x", padx=20, pady=(8, 10))
    progress = ttk.Progressbar(footer, mode="indeterminate", style="Thin.Horizontal.TProgressbar")
    progress.pack(fill="x", pady=(0, 6))
    ttk.Label(footer, textvariable=status, style="Muted.TLabel").pack(anchor="w")

    # ------------------------------------------------------------------ lógica
    def visible():
        thr = thr_var.get()
        cl = state["classes"]
        return [d for d in state["raw"]
                if d["score"] >= thr and (not cl or d["label"] in cl)]

    def schedule(delay=25):
        if pending["id"]:
            root.after_cancel(pending["id"])
        pending["id"] = root.after(delay, refresh)

    def render():
        canvas.delete("all")
        img = state["image"]
        cw, ch = canvas.winfo_width(), canvas.winfo_height()
        if img is None:
            canvas.create_text(cw / 2, ch / 2, fill=MUTED, font=F(13), justify="center",
                               text="Abre una imagen\nCtrl + O")
            return
        if cw < 40 or ch < 40:
            return
        s = min((cw - 24) / img.width, (ch - 24) / img.height)
        size = (max(1, int(img.width * s)), max(1, int(img.height * s)))
        key = (id(img), size)
        if state["view_key"] != key:
            state["base"] = img.resize(size, Image.LANCZOS)
            state["view_key"] = key
        vis = visible()
        scaled = [{**d, "box": [v * s for v in d["box"]]} for d in vis]
        hl = next((i for i, d in enumerate(vis) if d["id"] == state["selected"]), None)
        base = state["base"]
        if show_gt.get() and state["gt"]:
            base = base.copy()
            dg = ImageDraw.Draw(base)
            for g in state["gt"]:
                dg.rectangle([v * s for v in g["box"]], outline=(255, 255, 255), width=1)
        shown = draw_detections(base, scaled, hl) if scaled else base
        state["photo"] = ImageTk.PhotoImage(shown)
        canvas.create_image(cw // 2, ch // 2, image=state["photo"])

    def bars(c, W, y, title_text, rows):
        """Bloque con título y filas (etiqueta, fracción 0-1, texto, color) con barra."""
        pad = 18
        c.create_text(pad, y, anchor="nw", text=title_text, fill=MUTED, font=F(8, True))
        y += 24
        for label, frac, text, col in rows:
            c.create_text(pad, y, anchor="nw", text=label, fill=FG, font=F(10))
            c.create_text(W - pad, y, anchor="ne", text=text, fill=FG, font=F(10, True))
            by = y + 23
            c.create_rectangle(pad, by, W - pad, by + 4, fill=TRACK, outline="")
            c.create_rectangle(pad, by, pad + (W - 2 * pad) * max(0.0, min(1.0, frac)), by + 4,
                               fill=col, outline="")
            y += 40
        return y + 10

    def draw_panel():
        c = panel
        c.delete("all")
        W = max(c.winfo_width(), 300)
        pad = 18
        y = 14

        def title(text, y):
            c.create_text(pad, y, anchor="nw", text=text, fill=MUTED, font=F(8, True))
            return y + 24

        if state["image"] is None:
            c.create_text(W / 2, 50, text="Sin resultados", fill=MUTED, font=F(10))
            return
        if state["busy"]:
            c.create_text(W / 2, 50, text="Analizando…", fill=MUTED, font=F(10))
            return

        vis = visible()
        m = state["metrics"]
        if state["gt"] is not None and m is not None:
            cl = state["classes"]
            gts = [g for g in state["gt"] if not cl or g["label"] in cl]
            pr = precision_recall([(vis, gts)], thr_var.get())
            y = bars(c, W, y, f"PRECISIÓN / RECALL  ·  umbral {thr_var.get():.0%}  ·  IoU 0.5", [
                ("Precisión", pr["precision"], f"{pr['precision']:.0%}", ACCENT),
                ("Recall", pr["recall"], f"{pr['recall']:.0%}", "#34d399"),
                ("F1", pr["f1"], f"{pr['f1']:.0%}", "#fbbf24")])
            if m["per_class"]:
                rows = [(k, v["ap50"], f"{v['ap50']:.0%}  ·  GT {v['n_gt']}", hexcolor(k))
                        for k, v in sorted(m["per_class"].items(), key=lambda kv: -kv[1]["ap50"])]
                y = bars(c, W, y, "AP@50 POR CLASE", rows)
        counts = Counter(d["label"] for d in vis)
        maxc = max(counts.values(), default=1)

        y = title(f"CLASES DETECTADAS  ·  {len(counts)}", y)
        if not counts:
            c.create_text(pad, y, anchor="nw", text="Ninguna con estos filtros", fill=MUTED,
                          font=F(10))
            y += 28
        for label, n in counts.most_common():
            col = hexcolor(label)
            c.create_oval(pad, y + 5, pad + 8, y + 13, fill=col, outline="")
            c.create_text(pad + 16, y, anchor="nw", text=label, fill=FG, font=F(10))
            c.create_text(W - pad, y, anchor="ne", text=f"×{n}", fill=MUTED, font=F(10))
            by = y + 23
            c.create_rectangle(pad, by, W - pad, by + 4, fill=TRACK, outline="")
            c.create_rectangle(pad, by, pad + (W - 2 * pad) * n / maxc, by + 4, fill=col,
                               outline="")
            y += 40

        y += 10
        y = title(f"DETECCIONES  ·  {len(vis)}", y)
        for d in vis:
            tag = f"det{d['id']}"
            col = hexcolor(d["label"])
            fill = SEL if d["id"] == state["selected"] else PANEL
            c.create_rectangle(6, y - 6, W - 6, y + 38, fill=fill, outline="", tags=(tag,))
            c.create_text(pad, y, anchor="nw", text=d["label"], fill=FG, font=F(10), tags=(tag,))
            c.create_text(W - pad, y, anchor="ne", text=f"{d['score']:.0%}", fill=FG,
                          font=F(10, True), tags=(tag,))
            by = y + 23
            c.create_rectangle(pad, by, W - pad, by + 4, fill=TRACK, outline="", tags=(tag,))
            c.create_rectangle(pad, by, pad + (W - 2 * pad) * d["score"], by + 4, fill=col,
                               outline="", tags=(tag,))
            c.tag_bind(tag, "<Button-1>", lambda e, k=d["id"]: select(k))
            c.tag_bind(tag, "<Enter>", lambda e: c.config(cursor="hand2"))
            c.tag_bind(tag, "<Leave>", lambda e: c.config(cursor=""))
            y += 48
        c.configure(scrollregion=(0, 0, W, y + 10))

    def refresh():
        pending["id"] = None
        render()
        draw_panel()
        has = bool(state["raw"]) and not state["busy"]
        for b in (btn_save, btn_json):
            b.state(["!disabled"] if has else ["disabled"])
        btn_detect.state(["disabled"] if (state["busy"] or state["image"] is None)
                         else ["!disabled"])
        if state["meta"] is not None and not state["busy"]:
            dt, dev = state["meta"]
            time_var.set(f"{dt:.2f} s")
            count_var.set(str(len(visible())))
            dev_var.set(dev.upper())
            status.set(f"{len(visible())} objetos visibles de {len(state['raw'])} "
                       f"· detección en {dt:.2f} s · {dev}")
        elif not state["busy"]:
            time_var.set("—")
            count_var.set("—")
            dev_var.set("—")
        else:
            count_var.set("—")
            dev_var.set("—")
        mm = state["metrics"]
        if mm and mm["map50"] is not None:
            map50_var.set(f"{mm['map50']:.1%}")
            map_var.set(f"{mm['map']:.1%}")
        else:
            map50_var.set("—")
            map_var.set("—")

    def select(det_id):
        state["selected"] = None if state["selected"] == det_id else det_id
        refresh()

    def open_image(path=None):
        if not isinstance(path, str):
            path = filedialog.askopenfilename(
                title="Selecciona una imagen",
                filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
                           ("Todos", "*.*")])
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return
        state.update(path=path, image=img, raw=[], selected=None, view_key=None, meta=None,
                     gt=None, metrics=None)
        if state["gt_obj"] is not None:      # anotaciones tipo dataset: se aplican a cada imagen
            try:
                state["gt"] = state["gt_obj"].for_image(path, img.size)
            except Exception:
                state["gt"] = None
        file_var.set(f"{Path(path).name}  ·  {img.width}×{img.height}")
        state["cmp_pending"] = auto_cmp.get()
        refresh()
        detect()

    def tick():  # cronómetro en vivo mientras se analiza
        if state["busy"]:
            time_var.set(f"{time.perf_counter() - state['t0']:.1f} s")
            root.after(100, tick)

    def detect():
        if state["image"] is None:
            return
        if state["busy"]:
            return
        model, img = model_var.get(), state["image"]
        state.update(busy=True, raw=[], selected=None, meta=None, metrics=None,
                     t0=time.perf_counter())
        tick()
        progress.configure(mode="indeterminate")
        progress.start(12)
        status.set("Analizando… (la primera vez se descarga el modelo)")
        refresh()

        def work():
            try:
                det = get_detector(model, args.device)
                t0 = time.perf_counter()
                dets = det.detect(img, MIN_THR, None)
                dt = time.perf_counter() - t0
                jobs.put(lambda: done(img, dets, dt, str(det.device)))
            except Exception as e:
                jobs.put(lambda err=e: fail(err))

        threading.Thread(target=work, daemon=True).start()

    def stop_progress():
        progress.stop()
        progress.configure(mode="determinate", value=0)

    def done(img, dets, dt, device):
        stop_progress()
        state["busy"] = False
        if img is not state["image"]:      # se abrió otra imagen mientras tanto
            detect()
            return
        for i, d in enumerate(dets):
            d["id"] = i
        state.update(raw=dets, meta=(dt, device))
        update_metrics()
        refresh()
        if state.pop("cmp_pending", False):
            compare_all()

    def fail(err):
        stop_progress()
        state["busy"] = False
        status.set("Error")
        refresh()
        messagebox.showerror("Error", str(err))

    def save_image():
        if not state["raw"]:
            return
        stem = Path(state["path"]).stem
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", initialfile=f"{stem}_det.jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if path:
            draw_detections(state["image"], visible()).save(path)
            status.set(f"Guardado en {path}")

    def export_json():
        if not state["raw"]:
            return
        stem = Path(state["path"]).stem
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialfile=f"{stem}_det.json",
            filetypes=[("JSON", "*.json")])
        if path:
            data = {"imagen": state["path"], "modelo": model_var.get(),
                    "umbral": round(thr_var.get(), 2),
                    "tiempo_deteccion_s": round(state["meta"][0], 3),
                    "dispositivo": state["meta"][1],
                    "detecciones": [{k: v for k, v in d.items() if k != "id"}
                                    for d in visible()]}
            if state["metrics"] is not None:
                data["metricas"] = state["metrics"]
            Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
            status.set(f"Guardado en {path}")

    # ---------------------------------------------------------------- mAP
    def update_metrics():
        if state["gt"] is None or state["busy"] or state["meta"] is None:
            state["metrics"] = None
        else:   # mAP con todas las detecciones (umbral mínimo), independiente del deslizador
            state["metrics"] = evaluate_map([(state["raw"], state["gt"])])

    def load_gt(path=None):
        if not isinstance(path, str):
            path = filedialog.askopenfilename(
                title="Anotaciones reales (ground truth)",
                filetypes=[("COCO JSON / VOC XML / YOLO TXT", "*.json *.xml *.txt"),
                           ("Todos", "*.*")])
        if not path:
            return
        try:
            gt_obj = GroundTruth(path)
            if state["image"] is None:
                if not gt_obj.is_dataset:
                    raise ValueError("Abre primero una imagen.")
                state["gt_obj"] = gt_obj
                status.set("Anotaciones cargadas: se aplicarán al abrir cada imagen")
                return
            gts = gt_obj.for_image(state["path"], state["image"].size)
            if gts is None:
                raise ValueError("Esta imagen no aparece en las anotaciones.")
        except Exception as e:
            messagebox.showerror("Anotaciones", str(e))
            return
        state["gt_obj"] = gt_obj if gt_obj.is_dataset else None
        state["gt"] = gts
        update_metrics()
        refresh()
        status.set(f"{len(gts)} objetos anotados cargados desde {Path(path).name}")

    def eval_dialog():
        dlg = tk.Toplevel(root)
        dlg.title("Evaluar carpeta · mAP")
        dlg.configure(bg=PANEL)
        dlg.geometry(f"700x760+{root.winfo_rootx() + 90}+{root.winfo_rooty() + 30}")
        dlg.transient(root)
        dlg.grab_set()
        img_var, gt_var = tk.StringVar(), tk.StringVar()
        msg = tk.StringVar(value="Elige la carpeta de imágenes y sus anotaciones.")
        cancel, result = {"v": False}, {"m": None}

        def field(label, var, buttons):
            ttk.Label(dlg, text=label, style="Cap.TLabel").pack(anchor="w", padx=20, pady=(16, 6))
            r = tk.Frame(dlg, bg=PANEL)
            r.pack(fill="x", padx=20)
            tk.Entry(r, textvariable=var, bg=BG, fg=FG, insertbackground=FG, relief="flat",
                     font=F(10), highlightthickness=1, highlightbackground=LINE,
                     highlightcolor=ACCENT).pack(side="left", fill="x", expand=True, ipady=6)
            for text, cmd in buttons:
                ttk.Button(r, text=text, command=cmd).pack(side="left", padx=(6, 0))

        field("CARPETA DE IMÁGENES", img_var, [
            ("Examinar…", lambda: img_var.set(filedialog.askdirectory(parent=dlg) or img_var.get()))])
        field("ANOTACIONES  ·  JSON COCO, o carpeta con un .xml (VOC) / .txt (YOLO) por imagen", gt_var, [
            ("JSON…", lambda: gt_var.set(filedialog.askopenfilename(
                parent=dlg, filetypes=[("COCO JSON", "*.json")]) or gt_var.get())),
            ("Carpeta…", lambda: gt_var.set(filedialog.askdirectory(parent=dlg) or gt_var.get()))])

        bar = tk.Frame(dlg, bg=PANEL)
        bar.pack(fill="x", padx=20, pady=(18, 8))
        btn_run = ttk.Button(bar, text="Evaluar", style="Accent.TButton", command=lambda: run())
        btn_run.pack(side="left")
        btn_exp = ttk.Button(bar, text="Exportar JSON", state="disabled", command=lambda: export())
        btn_exp.pack(side="left", padx=8)
        pbar = ttk.Progressbar(dlg, style="Thin.Horizontal.TProgressbar", maximum=1, value=0)
        pbar.pack(fill="x", padx=20)
        ttk.Label(dlg, textvariable=msg, style="SideMuted.TLabel", wraplength=640,
                  justify="left").pack(anchor="w", padx=20, pady=(8, 4))
        rc = tk.Canvas(dlg, bg=PANEL, highlightthickness=0, yscrollincrement=20)
        rc.pack(fill="both", expand=True, pady=(4, 0))
        rc.bind("<MouseWheel>", lambda e: rc.yview_scroll(-2 if e.delta > 0 else 2, "units"))
        rc.bind("<Button-4>", lambda e: rc.yview_scroll(-2, "units"))
        rc.bind("<Button-5>", lambda e: rc.yview_scroll(2, "units"))

        def draw_eval(m):
            rc.delete("all")
            W, pad, y = max(rc.winfo_width(), 640), 18, 12
            if m["map50"] is None:
                rc.create_text(pad, y, anchor="nw", fill=MUTED, font=F(10),
                               text="Ninguna imagen con objetos anotados para evaluar.")
                return
            pr, t = m["pr"], m.get("tiempo_medio_s")
            items = [(f"{m['map50']:.1%}", "mAP@50"), (f"{m['map']:.1%}", "mAP@50:95"),
                     (f"{m['map75']:.1%}", "mAP@75"), (f"{pr['precision']:.1%}", "PRECISIÓN"),
                     (f"{pr['recall']:.1%}", "RECALL"),
                     (f"{t * 1000:.0f} ms" if t else "—", "TIEMPO / IMAGEN")]
            cw = (W - 2 * pad - 20) / 3
            for i, (val, cap) in enumerate(items):
                x, yy = pad + (i % 3) * (cw + 10), y + (i // 3) * 72
                rc.create_rectangle(x, yy, x + cw, yy + 62, fill="#1b1f28", outline="")
                rc.create_text(x + 12, yy + 10, anchor="nw", text=val, fill=FG, font=F(15, True))
                rc.create_text(x + 12, yy + 40, anchor="nw", text=cap, fill=MUTED, font=F(7, True))
            y += 2 * 72 + 4
            info = (f"{m['n_images']} imágenes evaluadas  ·  {m['n_gt']} objetos anotados"
                    + (f"  ·  {m['skipped']} sin anotación (omitidas)" if m["skipped"] else "")
                    + f"  ·  precisión/recall con umbral {pr['score_thr']:.0%}, IoU 0.5")
            rc.create_text(pad, y, anchor="nw", text=info, fill=MUTED, font=F(9), width=W - 2 * pad)
            y += 40
            if m["unsupported"]:
                rc.create_text(pad, y, anchor="nw", fill="#fbbf24", font=F(9), width=W - 2 * pad,
                               text="Clases que no existen en COCO (AP = 0): " + ", ".join(m["unsupported"]))
                y += 40
            rows = [(k, v["ap50"], f"AP50 {v['ap50']:.0%}  ·  AP {v['ap']:.0%}  ·  GT {v['n_gt']}", hexcolor(k))
                    for k, v in sorted(m["per_class"].items(), key=lambda kv: -kv[1]["ap50"])]
            y = bars(rc, W, y, f"AP POR CLASE  ·  {len(rows)}", rows)
            rc.configure(scrollregion=(0, 0, W, y + 10))

        def prog(i, n):
            if dlg.winfo_exists():
                pbar.configure(value=i)
                msg.set(f"Analizando imágenes… {i}/{n}")

        def finished(m):
            if not dlg.winfo_exists():
                return
            result["m"] = m
            btn_run.state(["!disabled"])
            btn_exp.state(["!disabled"])
            msg.set("Evaluación terminada." if not cancel["v"] else "Evaluación cancelada (parcial).")
            draw_eval(m)

        def failed(err):
            if dlg.winfo_exists():
                btn_run.state(["!disabled"])
                msg.set("Error durante la evaluación.")
                messagebox.showerror("Error", str(err), parent=dlg)

        def run():
            folder, ann = img_var.get().strip(), gt_var.get().strip()
            images = collect_images([folder]) if folder and Path(folder).is_dir() else []
            if not images:
                messagebox.showerror("Imágenes", "Elige una carpeta con imágenes.", parent=dlg)
                return
            try:
                if not ann:
                    raise ValueError("Elige el archivo o carpeta de anotaciones.")
                gt = GroundTruth(ann)
                if not gt.is_dataset:
                    raise ValueError("Elige un JSON COCO o una carpeta con un archivo de "
                                     "anotaciones por imagen.")
            except Exception as e:
                messagebox.showerror("Anotaciones", str(e), parent=dlg)
                return
            cancel["v"] = False
            btn_run.state(["disabled"])
            btn_exp.state(["disabled"])
            pbar.configure(maximum=len(images), value=0)
            msg.set("Cargando modelo… (la primera vez se descarga)")
            model, thr = model_var.get(), thr_var.get()

            def work():
                try:
                    det = get_detector(model, args.device)
                    m = evaluate_dataset(
                        det, images, gt, score_thr=thr,
                        progress=lambda i, n: jobs.put(lambda: prog(i, n)),
                        cancel=lambda: cancel["v"])
                    m["modelo"], m["dispositivo"] = model, str(det.device)
                    jobs.put(lambda: finished(m))
                except Exception as e:
                    jobs.put(lambda err=e: failed(err))

            threading.Thread(target=work, daemon=True).start()

        def export():
            path = filedialog.asksaveasfilename(parent=dlg, defaultextension=".json",
                                                initialfile="metricas_map.json",
                                                filetypes=[("JSON", "*.json")])
            if path and result["m"]:
                Path(path).write_text(json.dumps(result["m"], indent=2, ensure_ascii=False),
                                      encoding="utf-8")
                msg.set(f"Guardado en {path}")

        def close():
            cancel["v"] = True
            dlg.destroy()

        dlg.protocol("WM_DELETE_WINDOW", close)

    # ------------------------------------------------------- comparación de modelos
    CMP_COLS = [("modelo", "Modelo", 250, "w"), ("tiempo_s", "Tiempo", 80, "e"),
                ("fps", "FPS", 60, "e"), ("n_det", "Objetos", 70, "e"),
                ("score_medio", "Conf. media", 95, "e"), ("map50", "mAP@50", 80, "e"),
                ("map", "mAP@50:95", 95, "e"), ("precision", "Precisión", 85, "e"),
                ("recall", "Recall", 70, "e"), ("f1", "F1", 60, "e")]
    CMP_FMT = {"tiempo_s": lambda v: f"{v:.2f} s", "fps": lambda v: f"{v:.1f}",
               "n_det": str, "score_medio": lambda v: f"{v:.0%}", "map50": lambda v: f"{v:.1%}",
               "map": lambda v: f"{v:.1%}", "precision": lambda v: f"{v:.1%}",
               "recall": lambda v: f"{v:.1%}", "f1": lambda v: f"{v:.1%}"}
    CMP_BEST = ("tiempo_s", "fps", "map50", "map", "precision", "recall", "f1")

    def cmp_fill():
        tree = cmp["tree"]
        tree.delete(*tree.get_children())
        rows = cmp["rows"]
        best = {}
        for k in CMP_BEST:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                best[k] = min(vals) if k == "tiempo_s" else max(vals)
        for r in rows:
            cells = [r["modelo"]]
            for k, *_ in CMP_COLS[1:]:
                v = r.get(k)
                txt = "—" if v is None else CMP_FMT[k](v)
                if len(rows) > 1 and k in best and v == best[k]:
                    txt += " ★"
                cells.append(txt)
            tree.insert("", "end", iid=r["modelo"], values=cells)

    def cmp_apply(_e=None):
        """Doble clic: muestra en la imagen principal las detecciones de ese modelo."""
        name = cmp["tree"].focus()
        row = next((r for r in cmp["rows"] if r["modelo"] == name), None)
        if row is None or cmp["img"] is not state["image"] or state["busy"]:
            return
        dets = [{**d, "id": i} for i, d in enumerate(row["raw"])]
        model_var.set(name)
        desc_var.set(MODELS[name][2])
        state.update(raw=dets, selected=None, meta=(row["tiempo_s"], row["device"]))
        update_metrics()
        refresh()

    def cmp_build():
        dlg = tk.Toplevel(root)
        dlg.configure(bg=PANEL)
        dlg.geometry(f"1000x420+{root.winfo_rootx() + 60}+{root.winfo_rooty() + 80}")
        dlg.transient(root)
        tree = ttk.Treeview(dlg, columns=[c[0] for c in CMP_COLS], show="headings",
                            selectmode="browse")
        for k, title_, w, anchor in CMP_COLS:
            tree.heading(k, text=title_)
            tree.column(k, width=w, anchor=anchor, stretch=(k == "modelo"))
        tree.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        tree.bind("<Double-1>", cmp_apply)
        pbar = ttk.Progressbar(dlg, style="Thin.Horizontal.TProgressbar", maximum=1, value=0)
        pbar.pack(fill="x", padx=16)
        msg = tk.StringVar()
        ttk.Label(dlg, textvariable=msg, style="SideMuted.TLabel", wraplength=960,
                  justify="left").pack(anchor="w", padx=16, pady=(6, 12))

        def close():
            cmp["cancel"] = True
            dlg.destroy()
        dlg.protocol("WM_DELETE_WINDOW", close)
        cmp.update(dlg=dlg, tree=tree, msg=msg, pbar=pbar)

    def compare_all():
        img = state["image"]
        if img is None or cmp["busy"]:
            return
        if cmp["dlg"] is None or not cmp["dlg"].winfo_exists():
            cmp_build()
        dlg, pbar, msg = cmp["dlg"], cmp["pbar"], cmp["msg"]
        dlg.lift()
        dlg.title(f"Comparación de modelos · {Path(state['path']).name}")
        cmp.update(rows=[], cancel=False, busy=True, img=img)
        cmp_fill()
        btn_cmp.state(["disabled"])
        pbar.configure(maximum=len(MODELS), value=0)
        thr, cl, gts = thr_var.get(), set(state["classes"]), state["gt"]
        msg.set("Cargando modelos… (la primera vez se descargan los pesos)")

        def prog(i, n, name):
            if dlg.winfo_exists():
                pbar.configure(value=i - 1)
                msg.set(f"Analizando con {name}… ({i}/{n})")

        def add_row(r):
            if dlg.winfo_exists() and cmp["img"] is img:
                cmp["rows"].append(r)
                cmp_fill()

        def finished(err=None):
            cmp["busy"] = False
            btn_cmp.state(["!disabled"])
            if not dlg.winfo_exists():
                return
            pbar.configure(value=len(cmp["rows"]))
            if err:
                msg.set("Error durante la comparación.")
                messagebox.showerror("Error", str(err), parent=dlg)
                return
            nota = "" if gts is not None else "  Carga anotaciones (GT) para ver mAP/precisión/recall."
            msg.set(f"Listo · umbral {thr:.0%} · ★ = mejor valor · doble clic en una fila "
                    f"para verla en la imagen.{nota}")

        def work():
            try:
                compare_models(img, gts, thr, cl or None, device=args.device,
                               progress=lambda i, n, nm: jobs.put(lambda: prog(i, n, nm)),
                               on_row=lambda r: jobs.put(lambda: add_row(r)),
                               cancel=lambda: cmp["cancel"])
                jobs.put(lambda: finished())
            except Exception as e:
                jobs.put(lambda err=e: finished(err))

        threading.Thread(target=work, daemon=True).start()

    def class_dialog():
        dlg = tk.Toplevel(root)
        dlg.title("Filtrar clases")
        dlg.configure(bg=PANEL)
        dlg.geometry(f"340x480+{root.winfo_rootx() + 120}+{root.winfo_rooty() + 80}")
        dlg.transient(root)
        dlg.grab_set()
        sel = set(state["classes"])
        q = tk.StringVar()
        items: list[str] = []
        ent = tk.Entry(dlg, textvariable=q, bg=BG, fg=FG, insertbackground=FG, relief="flat",
                       font=F(11), highlightthickness=1, highlightbackground=LINE,
                       highlightcolor=ACCENT)
        ent.pack(fill="x", padx=18, pady=(18, 10), ipady=6)
        lb = tk.Listbox(dlg, selectmode="multiple", exportselection=False, bg=BG, fg=FG,
                        selectbackground=ACCENT, selectforeground="white", relief="flat",
                        highlightthickness=0, activestyle="none", font=F(11))
        lb.pack(fill="both", expand=True, padx=18)

        def fill(*_):
            items[:] = [c for c in ALL_CLASSES if q.get().strip().lower() in c.lower()]
            lb.delete(0, "end")
            for i, c in enumerate(items):
                lb.insert("end", c)
                if c in sel:
                    lb.selection_set(i)

        def on_select(_e=None):
            sel.difference_update(items)
            sel.update(items[i] for i in lb.curselection())

        def apply():
            state["classes"] = set(sel)
            n = len(sel)
            classes_var.set("Todas" if n == 0 else ", ".join(sorted(sel)) if n <= 2
                            else f"{n} seleccionadas")
            state["selected"] = None
            dlg.destroy()
            refresh()

        def clear():
            sel.clear()
            fill()

        lb.bind("<<ListboxSelect>>", on_select)
        q.trace_add("write", fill)
        bar = tk.Frame(dlg, bg=PANEL)
        bar.pack(fill="x", padx=18, pady=14)
        ttk.Button(bar, text="Limpiar", command=clear).pack(side="left")
        ttk.Button(bar, text="Aplicar", style="Accent.TButton", command=apply).pack(side="right")
        fill()
        ent.focus_set()

    # ---- eventos ----
    def on_wheel(e):
        up = getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0
        panel.yview_scroll(-2 if up else 2, "units")

    def bind_wheel(_e=None):
        root.bind_all("<MouseWheel>", on_wheel)
        root.bind_all("<Button-4>", on_wheel)
        root.bind_all("<Button-5>", on_wheel)

    def unbind_wheel(_e=None):
        for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            root.unbind_all(ev)

    panel.bind("<Enter>", bind_wheel)
    panel.bind("<Leave>", unbind_wheel)
    panel.bind("<Configure>", lambda e: schedule(60))
    canvas.bind("<Configure>", lambda e: schedule(60))
    canvas.bind("<Button-1>", lambda e: state["image"] is None and open_image())
    combo.bind("<<ComboboxSelected>>", lambda e: (
        desc_var.set(MODELS[model_var.get()][2]), combo.selection_clear(),
        state["image"] is not None and not state["busy"] and detect()))
    root.bind("<Control-o>", lambda e: open_image())
    root.bind("<Control-s>", lambda e: save_image())
    root.bind("<Return>", lambda e: detect())
    root.bind("<F5>", lambda e: detect())

    def poll():  # ejecuta en el hilo principal lo que piden los hilos de trabajo
        try:
            while True:
                jobs.get_nowait()()
        except queue.Empty:
            pass
        root.after(80, poll)

    poll()
    refresh()
    if args.inputs:
        def first_load():
            open_image(args.inputs[0])
            if args.gt:
                load_gt(args.gt)
        root.after(300, first_load)
    elif args.gt:
        root.after(300, lambda: load_gt(args.gt))
    root.mainloop()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Detección de objetos con R-CNN / Faster R-CNN")
    ap.add_argument("inputs", nargs="*", help="imágenes o carpetas")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("-s", "--score", type=float, default=0.5, help="umbral de confianza")
    ap.add_argument("-c", "--classes", nargs="+", help="solo estas clases (p.ej. person car)")
    ap.add_argument("-o", "--output", default="salida", help="carpeta de salida")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (auto por defecto)")
    ap.add_argument("--json", action="store_true", help="guardar detecciones en JSON")
    ap.add_argument("--cli", action="store_true", help="modo consola (sin ventana): procesa y guarda en --output")
    ap.add_argument("--gt", help="anotaciones reales para calcular mAP: JSON COCO, carpeta o "
                                 "archivo VOC (.xml) / YOLO (.txt)")
    ap.add_argument("--web", action="store_true", help="lanzar interfaz web (Gradio)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    if args.list_models:
        for k, (_, _, desc) in MODELS.items():
            print(f"{k:42s} {desc}")
        return 0
    if args.web:
        return run_web(args)
    if args.cli:
        if not args.inputs:
            ap.error("--cli requiere al menos una imagen o carpeta")
        return run_cli(args)
    return run_gui(args)   # por defecto: ventana de escritorio


if __name__ == "__main__":
    sys.exit(main())