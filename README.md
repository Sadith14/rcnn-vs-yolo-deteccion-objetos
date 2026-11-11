# Detección de objetos con R-CNN y YOLO en imágenes y video

**Universidad Nacional del Altiplano · Ingeniería de Sistemas · Visión Artificial**
Actividad 06 · Segunda unidad · Docente: Ing. Fernández Chambi, Mayenka

Cuatro aplicaciones que comparan la detección de objetos con **Faster R-CNN** (dos etapas) y **YOLO** (una etapa), sobre imágenes y sobre video o simulación de tiempo real. Todos los modelos usan pesos preentrenados en **COCO (80 clases)**.

| Ítem | Enfoque | Medio | Aplicación | Interfaz |
|---|---|---|---|---|
| [1](#ítem-1--faster-r-cnn-en-imágenes) | Faster R-CNN / Mask R-CNN | Imágenes | `rcnn_detector.py` | Tkinter · consola · Gradio |
| [2](#ítem-2--yolo-en-imágenes) | YOLO11 / YOLO12 | Imágenes | `yolo_detector.py` | CustomTkinter · consola · Gradio |
| [3](#ítem-3--faster-r-cnn-en-video--tiempo-real) | Faster R-CNN / Mask R-CNN | Video / tiempo real | `app_frcnn.py` | Streamlit |
| [4](#ítem-4--yolo-en-video--tiempo-real) | YOLO11n | Video / tiempo real | `gui.py`, `yolo_video.py` | PySide6 · Streamlit · consola |

**Índice:** [Marco teórico](#marco-teórico) · [Metodología](#metodología-de-evaluación) · [Ítem 1](#ítem-1--faster-r-cnn-en-imágenes) · [Ítem 2](#ítem-2--yolo-en-imágenes) · [Ítem 3](#ítem-3--faster-r-cnn-en-video--tiempo-real) · [Ítem 4](#ítem-4--yolo-en-video--tiempo-real) · [Imágenes vs video](#imágenes-vs-video) · [Conclusiones](#conclusiones) · [Limitaciones y pendientes](#limitaciones-y-pendientes)

---

## Marco teórico

| Aspecto | R-CNN / Faster R-CNN | YOLO |
|---|---|---|
| Etapas | Dos: propuesta de regiones (RPN) + clasificación | Una: cajas y clases en una sola pasada |
| Velocidad | Más lento | Muy rápido (cercano a tiempo real) |
| Objetos pequeños | Tradicionalmente buena | Mejora con mayor `imgsz` y modelos grandes |
| Complejidad | Mayor | Menor |

**Evolución de la familia R-CNN:** R-CNN usa Selective Search (~2000 regiones) y una CNN por región (muy lento); Fast R-CNN comparte una CNN por imagen con RoI Pooling (medio); **Faster R-CNN** reemplaza Selective Search por una red aprendida (RPN) y es el más rápido de la familia. Mask R-CNN añade máscaras, pero aquí solo se usan sus cajas.

---

## Metodología de evaluación

Se definen las mismas métricas en todas las aplicaciones:

- **IoU:** solape entre la caja predicha y la real.
- **mAP@0.5 / @0.75 / @[0.5:0.95]:** promedio de AP por clase con distintos umbrales de IoU; el último es el estándar de COCO.
- **Precisión** = TP / (TP + FP) · **Recall** = TP / (TP + FN) · **F1:** media armónica de ambos.
- **Latencia, tiempo por imagen y FPS**, tras un calentamiento previo del modelo.
- **Factor de tiempo real (video):** `FPS de inferencia / FPS del video`; **≥ 1 = tiempo real**.
- **% descartados:** frames que llegan mientras el modelo está ocupado y se pierden, como en una cámara en vivo.

**Para que la comparación sea válida:** mismo equipo, mismo conjunto de imágenes o video y mismo umbral de confianza. Los umbrales por defecto difieren entre apps (R-CNN imágenes 0.5 en consola, R-CNN video 0.6, YOLO imágenes 0.25; YOLO video 0.6 en GPU y 0.35 en CPU), lo que altera precisión y recall sin cambiar la calidad del modelo. Las apps de imágenes calculan el mAP con un evaluador propio (sin `pycocotools`) y la de R-CNN en video usa `torchmetrics`: ambos siguen el estilo COCO pero pueden diferir ligeramente.

---

## Ítem 1 — Faster R-CNN en imágenes

Aplicación Python con tres modos: **ventana de escritorio** (Tkinter, tema oscuro; por defecto), **consola** (`--cli`) y **web** (`--web`, Gradio, sin métricas). Usa GPU si hay CUDA; también se puede forzar `cpu`, `cuda` o `mps` con `--device`.

**Funcionalidades**

- 5 modelos seleccionables, del más preciso al más ligero para CPU.
- Tiempo de detección en pantalla con cronómetro en vivo.
- Métricas contra anotaciones reales: mAP@50, mAP@50:95, mAP@75, precisión, recall, F1 y AP por clase.
- **Comparación de los 5 modelos** sobre la misma imagen (tiempo, FPS, objetos, confianza media y métricas), con ★ en el mejor valor de cada columna.
- Umbral de confianza y filtro de clases **instantáneos**, sin volver a ejecutar el modelo.
- Evaluación por carpeta con anotaciones en **JSON COCO, XML Pascal VOC o TXT YOLO**.
- Exportación de imagen anotada, detecciones (JSON) y métricas (`metricas.json`).
- Atajos: `Ctrl+O` abrir · `Ctrl+S` guardar · `Enter`/`F5` detectar.

**Modelos**

| Modelo | box AP (COCO val) | Descripción |
|---|---|---|
| `fasterrcnn_resnet50_fpn_v2` | 46.7 | Máxima precisión (por defecto) |
| `fasterrcnn_resnet50_fpn` | 37.0 | Versión clásica |
| `fasterrcnn_mobilenet_v3_large_fpn` | 32.8 | Equilibrio velocidad/precisión |
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 22.8 | El más rápido, ideal sin GPU |
| `maskrcnn_resnet50_fpn_v2` | 47.4 | Se usan solo las cajas |

**Instalación y uso**

```bash
pip install -r requirements.txt            # Python 3.9+, torch, torchvision, Pillow, numpy
python rcnn_detector.py                    # ventana
python rcnn_detector.py foto.jpg --gt anotaciones.json              # ventana con mAP
python rcnn_detector.py imagenes/ --cli --gt anotaciones.json       # consola con mAP
python rcnn_detector.py foto.jpg --cli -m fasterrcnn_mobilenet_v3_large_320_fpn -s 0.7 -c person car
python rcnn_detector.py --web                                       # http://127.0.0.1:7860
```

**Detalle técnico.** Se detecta siempre con umbral mínimo **0.05** para conservar toda la curva precisión-recall; el deslizador solo filtra lo mostrado, por eso no afecta al mAP. Precisión, recall y F1 sí dependen del umbral (IoU fijo 0.5). El mAP sigue el protocolo COCO (asignación voraz por confianza y AP con interpolación de 101 puntos) y difiere ligeramente de `pycocotools`: no separa por tamaño, descarta `iscrowd` y limita a 100 detecciones por clase e imagen. El tiempo mide solo la inferencia; la primera detección de cada sesión es más lenta (calentamiento).

**Resultados** (CPU, umbral 50 %, una sola imagen):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | F1 | s/imagen | FPS |
|---|---|---|---|---|---|---|---|
| `fasterrcnn_resnet50_fpn_v2` | 80.0 % | 66.0 % | 75.0 % | 75.0 % | 75.0 % | 6.58 | 0.2 |
| `fasterrcnn_resnet50_fpn` | 77.3 % | 54.8 % | 75.0 % | 75.0 % | 75.0 % | 4.49 | 0.2 |
| `fasterrcnn_mobilenet_v3_large_fpn` | **93.2 %** | 65.7 % | **81.8 %** | 75.0 % | 78.3 % | 0.98 | 1.0 |
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 57.1 % | 36.7 % | 75.0 % | 25.0 % | 37.5 % | **0.33** | **3.0** |
| `maskrcnn_resnet50_fpn_v2` | 91.1 % | **74.9 %** | 76.9 % | **83.3 %** | **80.0 %** | 7.28 | 0.1 |

Lectura: el modelo más ligero (`mobilenet_320`) es ~20 veces más rápido que ResNet50 v2 pero pierde mucho recall (25 %). `mobilenet_v3_large_fpn` ofrece el mejor equilibrio en esta imagen.

---

## Ítem 2 — YOLO en imágenes

Aplicación con los mismos tres modos (ventana CustomTkinter, consola `--cli`, web `--web`) y **el mismo evaluador de métricas** que el ítem 1, para comparar con criterios idénticos.

**Funcionalidades**

- **10 modelos** Ultralytics: YOLO11 y YOLO12 en tamaños n, s, m, l, x.
- Parámetros: modelo, umbral de confianza (25 % por defecto), tamaño de entrada `imgsz` (640 / 960 / 1280) y filtro de clases.
- Métricas: mAP@0.5, mAP@0.75, mAP@[0.5:0.95], precisión, recall, F1, tiempo por imagen y FPS.
- Comparación de modelos sobre una misma imagen y evaluación por carpeta (COCO JSON, Pascal VOC, YOLO TXT).
- Exportación de imagen anotada, detecciones y métricas en JSON. Atajos: `Ctrl+O` y `F5`.

**Modelos** (AP publicado por Ultralytics en COCO val)

| Modelo | AP | Uso recomendado |
|---|---|---|
| `yolo11n` | 39.5 | El más rápido, demos en CPU |
| `yolo11s` | 47.0 | Equilibrio velocidad/precisión |
| `yolo11m` | 51.5 | Buen punto medio sin GPU |
| `yolo11l` | 53.4 | Alta precisión |
| `yolo11x` | 54.7 | Máxima precisión de YOLO11 (por defecto) |
| `yolo12n` / `s` / `m` / `l` | 40.6 / 48.0 / 52.5 / 53.7 | YOLO12, con atención |
| `yolo12x` | 55.2 | Máxima precisión disponible |

**Instalación y uso**

```bash
pip install -U ultralytics pillow numpy customtkinter   # actualizar ultralytics para yolo12*
pip install gradio                                      # solo para --web
python yolo_detector.py                                 # ventana
python yolo_detector.py carpeta/ --cli -m yolo12x.pt --imgsz 1280
python yolo_detector.py coco128/images/train2017 --cli \
    --gt coco128/labels/train2017 -m yolo11x.pt -o salida_11x     # evaluación con mAP
```

Salidas: `*_det.jpg`, `detecciones.json` (`--json`) y `metricas.json` (`--gt`). Para el mAP se detecta con umbral 0.05; precisión y recall usan el umbral elegido.

**Resultados** (mismo conjunto de imágenes, `imgsz` 640 salvo indicación):

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Precisión | Recall | F1 | s/imagen | FPS |
|---|---|---|---|---|---|---|---|
| `yolo11n` | 78.0 % | 50.5 % | 80.0 % | 33.3 % | 47.1 % | **0.011** | **91.0** |
| `yolo11m` | 75.0 % | 62.1 % | **90.9 %** | **83.3 %** | **87.0 %** | 0.028 | 35.5 |
| `yolo11x` | **83.3 %** | 72.8 % | 88.9 % | 66.7 % | 76.2 % | 0.052 | 19.1 |
| `yolo12x` | **83.3 %** | **75.2 %** | 90.0 % | 75.0 % | 81.8 % | 0.064 | 15.7 |
| `yolo11x` (1280) | N/D | N/D | N/D | N/D | N/D | N/D | N/D |

mAP@0.75 aparece como N/D en todas las filas. Lectura: al subir de `n` a `x` el mAP@0.5:0.95 pasa de 50.5 % a 72.8 %, a costa de ~5× más tiempo; YOLO12x mejora ligeramente a YOLO11x en mAP@0.5:0.95 pero es algo más lento.

### Comparación en imágenes: Faster R-CNN vs YOLO

| Métrica | Faster R-CNN (`mobilenet_v3_large_fpn`) | YOLO (`yolo12x`) |
|---|---|---|
| mAP@0.5 | **93.2 %** | 83.3 % |
| mAP@0.5:0.95 | 65.7 % | **75.2 %** |
| Precisión | 81.8 % | **90.0 %** |
| Recall | 75.0 % | 75.0 % |
| Tiempo por imagen | 0.98 s | **0.064 s** |
| FPS | 1.0 | **15.7** |

En esta imagen YOLO12x localiza mejor (mAP@0.5:0.95 y precisión) y es ~15× más rápido; Faster R-CNN gana en mAP@0.5. Con una sola imagen y dispositivos posiblemente distintos, es una referencia y no un veredicto.

---

## Ítem 3 — Faster R-CNN en video / tiempo real

App **Streamlit** que procesa video con modelos de `torchvision`, mide **rendimiento** y, una vez anotados los frames, **calidad** contra un *ground truth* creado desde la propia interfaz.

**Funcionalidades**

- Procesamiento en vivo con cajas de colores, etiquetas en español (`persona 87%`) e insignia PROCESADO / DESCARTADO.
- **Simulación de cámara en vivo:** si el modelo tarda, los frames que llegan mientras procesa se descartan; desactivada, procesa todos.
- Métricas en tiempo real: latencia actual y media, FPS de detección, factor RT, procesados y descartados.
- **Pestaña Anotar:** tabla editable de cajas, vista previa y pre-anotación con un modelo; se guarda en `ground_truth.json`.
- Captura de frames (1 de cada N, sin cajas) para anotarlos después.
- Validación: **modo prueba de cordura** (GT = detecciones con confianza ≥ 0.8; solo valida el cálculo, no son métricas reales) y vista GT vs predicción (verde / rojo).
- Historial de corridas con comparación entre modelos y exportación a CSV. Una corrida procesa un modelo.

**Instalación y uso**

```bash
pip install torch torchvision                     # según tu equipo: https://pytorch.org
pip install -U streamlit opencv-python pandas torchmetrics pycocotools   # o faster-coco-eval
streamlit run app_frcnn.py                        # http://localhost:8501
```

Estructura: `app_frcnn.py`, `ground_truth.json`, `videos/`, `resultados/`, `frames_anotar/`.

**Flujo:** (1) procesar y capturar frames → (2) anotar en **✍️ Anotar** → (3) volver a procesar con cada modelo para evaluar → (4) comparar en **📈 Historial y comparación**.

**Métricas**

| Métrica | Significado |
|---|---|
| Latencia (ms) | Tiempo medio de inferencia por frame (sincronizado en GPU). |
| FPS efectivo | Frames procesados por segundo de video (simulación) o velocidad real del modelo (todos los frames). |
| Factor RT | `(1 / latencia) / FPS del video`; ≥ 1 = tiempo real. |
| % descartados | Frames omitidos en la simulación. |
| Detec./frame | Promedio de objetos por frame procesado. |
| Precision / Recall | IoU ≥ 0.5, solo detecciones sobre el umbral (0.6 por defecto). |
| mAP@0.5 / @0.5:0.95 | Con todas las predicciones, vía `torchmetrics`. |

Cada modelo hace 3 inferencias de calentamiento antes de medir.

**Estado:** rendimiento ✅ disponible · calidad (Precision, Recall, mAP) 

**Resultados en GPU** (Colab T4, umbral 0.6, simulación de tiempo real):

| Video | Modelo | Latencia | Factor RT | Descartados | Detec./frame | ¿Tiempo real? |
|---|---|---|---|---|---|---|
| video1 (Full HD) | ResNet50-FPN | 122.4 ms | 0.33 | 74.3 % | 16.5 | No |
| video1 (Full HD) | MobileNetV3 | 43.0 ms | 0.93 | 49.2 % | 13.2 | No |
| video2 (4K) | ResNet50-FPN | 134.9 ms | 0.31 | 75.0 % | 18.1 | No |
| video2 (4K) | MobileNetV3 | 49.2 ms | 0.85 | 49.9 % | 11.6 | No |
| video3 (480p) | ResNet50-FPN | 136.3 ms | 0.29 | 75.1 % | 4.4 | No |
| video3 (480p) | MobileNetV3 | 30.9 ms | 1.29 | 2.6 % | 3.6 | Sí |

**Resultados en ejecución local** (app Streamlit, leídos de capturas):

| Video | Modelo | Latencia | FPS efectivo | Procesados / total | Descartados | Detec./frame |
|---|---|---|---|---|---|---|
| video3_fix (1080p, 24 FPS) | ResNet50-FPN | 3341.2 ms | 0.3 | 17 / 1346 | 98.7 % | 4.5 |
| video3_fix (1080p, 24 FPS) | MobileNetV3 | 758.7 ms | 1.28 | 72 / 1346 | 94.7 % | 3.8 |


**Consideraciones al anotar:** idealmente 30–50 frames variados; si pre-anotas con un modelo y corriges poco, el GT queda sesgado a su favor; los frames se identifican por índice y nombre de video, así que recodificar el video invalida las anotaciones.

---

## Ítem 4 — YOLO en video / tiempo real

Detecta con **YOLO11n** sobre **webcam** o **archivo de video**, con seguimiento **ByteTrack** (cuenta objetos distintos) y un modo que **simula tiempo real** descartando frames cuando el modelo no alcanza la velocidad del video.

**Estructura**

```
yolo_video/
├── gui.py                 # aplicación de escritorio (PySide6), principal
├── app.py                 # interfaz web (Streamlit), opcional
├── yolo_video.py          # motor de detección + tracking + métricas (consola)
├── graficar_metricas.py   # gráficas a partir del CSV
├── video_prueba.mp4       # video de prueba (180 frames @ 30 fps)
├── res_full/  res_rt/     # ejecuciones de ejemplo (todos los frames / tiempo real)
└── resultados/AAAAMMDD_HHMMSS/   # video anotado, CSV, JSON y 3 gráficas PNG por ejecución
```

**Uso**

```bash
pip install -r requirements.txt
python gui.py                                                    # ventana (fuente, configuración, ▶ Detectar)
streamlit run app.py                                             # versión web opcional
python yolo_video.py --source 0 --track                          # webcam (q para salir)
python yolo_video.py --source video_prueba.mp4 --track --realtime --save --out res_rt
python graficar_metricas.py res_rt
```

Opciones útiles: `--model yolo11s.pt`, `--conf 0.5`, `--classes 0 2`, `--no-show`. Al terminar, la ventana habilita las pestañas **Resumen** (6 indicadores y datos interesantes automáticos), **Rendimiento**, **Objetos** (detecciones frente a objetos distintos y cuándo aparece cada clase), **Confianza** y **Datos**.

**Cómo funciona**

1. **Captura** con OpenCV, frame a frame.
2. **Inferencia** en una sola pasada (rejilla → cajas + clases), sin la etapa de propuestas de Faster R-CNN.
3. **Post-proceso:** filtro de confianza y NMS.
4. **ByteTrack:** asigna IDs entre frames (`#3 person`) para contar objetos distintos.
5. **Reloj virtual:** el frame *k* "llega" en *k/fps* y el reloj avanza solo con el tiempo de detección; al terminar un frame se toma el último que ya llegó y se descartan los intermedios. Así el % de descarte depende solo del detector (≈ 1 − FPS de inferencia / FPS del video), no del disco ni de la interfaz, lo que permite comparar con Faster R-CNN de forma justa.
6. **Visualización ≤ 1280 px:** el modelo recibe el frame original (incluso 4K); dibujo, guardado y vista en vivo usan una versión reducida.

**Métricas registradas:** latencia (media, mediana, **p95**), FPS de inferencia y efectivo, factor RT, % descartados, detecciones por frame, confianza media, **inestabilidad** (cambio medio del nº de detecciones entre frames; parpadeo) y objetos únicos por clase. La p95 es el techo del comportamiento normal: la media esconde los picos. Límite de latencia para tiempo real = 1000 / FPS del video (40 ms a 25 FPS).

**Videos de prueba** (los mismos que el ítem 3):

| Video | Resolución | FPS | Frames | Duración | Contenido |
|---|---|---|---|---|---|
| video1 | 1920×1080 | 25 | 447 | 17.9 s | Personas caminando en una calle |
| video2 | 3840×2160 (4K) | 24 | 619 | 25.8 s | Avenida con buses, autos y motos |
| video3 | 854×480 | 25 | 973 | 38.9 s | Primeros planos de personas |

**Resultados en GPU** (Colab T4, yolo11n, umbral 0.6) — **tiempo real en los 6 casos**:

| Video | Entrada | Latencia | FPS inf. | FPS efectivo | Descartados | Detec./frame | Conf. media | Factor RT |
|---|---|---|---|---|---|---|---|---|
| video1 | 640 | 18.9 ms | 52.9 | 24.3 | 2.7 % | 5.6 | 0.80 | **2.12** |
| video1 | 1280 | 21.3 ms | 46.9 | 24.9 | 0.4 % | 6.5 | 0.80 | **1.88** |
| video2 | 640 | 17.6 ms | 56.8 | 23.6 | 1.6 % | 4.2 | 0.76 | **2.37** |
| video2 | 1280 | 26.7 ms | 37.5 | 22.7 | 5.2 % | 5.8 | 0.79 | **1.56** |
| video3 | 640 | 11.8 ms | 84.5 | 25.0 | 0 % | 1.5 | 0.82 | **3.38** |
| video3 | 1280 | 13.8 ms | 72.3 | 25.0 | 0 % | 1.4 | 0.78 | **2.89** |

Los pocos descartes (0.4–5.2 %) se deben a frames sueltos con picos de latencia mayores al límite de 40 ms.

**Resultados en CPU** (laptop, umbral 0.35, entrada 640) — **ningún caso llega a tiempo real**:

| Video · modelo | FPS inf. | Latencia | p95 | Descartados | Detec./frame | Objetos distintos | Conf. | Inestab. | Factor RT |
|---|---|---|---|---|---|---|---|---|---|
| video1 · yolo11n | 13.7 | 73 ms | 95 ms | 45.2 % | 8.27 | 93 | 0.70 | 0.82 | 0.55 |
| video2 · yolo11n | 9.7 | 103 ms | 186 ms | 59.3 % | 5.73 | 107 | 0.68 | 1.08 | 0.41 |
| video2 · yolo11s | 7.5 | 132 ms | 208 ms | 68.3 % | 7.81 | 111 | 0.72 | 1.26 | 0.32 |
| video3 · yolo11n | 11.7 | 86 ms | 114 ms | 53.3 % | 2.10 | 51 | 0.72 | 0.29 | 0.47 |

---

## Imágenes vs video

**Comparación YOLO11n-640 vs Faster R-CNN** (misma T4, umbral 0.6):

| Video | ResNet50-FPN | MobileNetV3 | YOLO11n-640 |
|---|---|---|---|
| video1 | 122.4 ms · RT 0.33 · 74.3 % desc. | 43.0 ms · RT 0.93 · 49.2 % desc. | **18.9 ms · RT 2.12 · 2.7 % desc.** |
| video2 | 134.9 ms · RT 0.31 · 75.0 % desc. | 49.2 ms · RT 0.85 · 49.9 % desc. | **17.6 ms · RT 2.37 · 1.6 % desc.** |
| video3 | 136.3 ms · RT 0.29 · 75.1 % desc. | 30.9 ms · RT 1.29 · 2.6 % desc. | **11.8 ms · RT 3.38 · 0 % desc.** |

YOLO11n es **6.5, 7.7 y 11.5 veces** más rápido que ResNet50-FPN (video1, 2 y 3) y **2.3, 2.8 y 2.6 veces** más rápido que MobileNetV3.

**Hallazgos principales**

1. **Qué cambia en video:** además de la calidad (mAP, precisión, recall) aparece una restricción temporal: cada frame debe procesarse antes de que llegue el siguiente. Las métricas clave pasan a ser latencia, FPS y factor RT. Un modelo con buen mAP puede ser inútil en vivo si no llega a tiempo real.
2. **Compromiso velocidad-precisión:** los modelos grandes suelen dar mayor mAP con más latencia; en video esa latencia se traduce en frames descartados. Los ligeros sacrifican algo de precisión para acercarse al tiempo real.
3. **Arquitectura:** YOLO mira la imagen una vez; Faster R-CNN propone regiones y clasifica cada una. Por eso ResNet50-FPN nunca llegó a tiempo real y MobileNetV3 solo en 480p.
4. **Detecciones por frame:** con el mismo umbral, Faster R-CNN detecta más (video1: 16.5 frente a 5.6), sobre todo personas pequeñas o lejanas. **No se puede concluir cuál es más preciso** en video: faltan anotaciones y cada modelo calibra distinto su confianza.
5. **Hardware:** con YOLO11n, la GPU T4 fue 4–7 veces más rápida que la CPU (video1: 73 → 18.9 ms; video2: 103 → 17.6 ms). Los umbrales fueron distintos (0.35 en CPU, 0.6 en GPU), por lo que solo es comparable la latencia.
6. **Resolución del video vs tamaño de entrada:** YOLO reduce el frame antes de procesarlo, así que el 4K tarda casi lo mismo que el Full HD en GPU. Entrada 1280 detecta más objetos pequeños (video2: 5.8 frente a 4.2 por frame) pero es más lenta (26.7 frente a 17.6 ms); en 480p no aporta nada.
7. **Modelo más grande:** en CPU con 4K, `yolo11s` es 28 % más lento que `yolo11n`, detecta más (7.81 frente a 5.73) y con algo más de confianza, pero descarta más frames (68 % frente a 59 %).
8. **Problemas propios del video:** parpadeo del número de objetos (inestabilidad 0.82–1.26 con mucha gente), cambios de clase de un mismo vehículo (`bus` ↔ `train`) con IDs distintos en ByteTrack, objetos rápidos que cruzan sin ser detectados por los frames descartados, y picos de latencia en la primera inferencia (300–380 ms en CPU), por lo que se hace calentamiento y se reporta la p95.
9. **Umbral de confianza:** uno alto sube la precisión y baja el recall; uno bajo hace lo contrario. No afecta al mAP.

---

## Conclusiones

- Se desarrollaron cuatro aplicaciones (Faster R-CNN y YOLO, en imágenes y en video) con métricas comparables de rendimiento y, en imágenes, de calidad.
- **En video, YOLO11n en GPU es la opción práctica:** tiempo real en los tres videos (factor 1.56–3.38) y entre 2.3 y 11.5 veces más rápido que Faster R-CNN. Faster R-CNN ResNet50-FPN no llegó a tiempo real en ninguno y perdió cerca del 75 % de los frames.
- **En imágenes,** YOLO12x igualó o superó a Faster R-CNN en mAP@0.5:0.95 y precisión y fue mucho más rápido; Faster R-CNN MobileNetV3 tuvo mayor mAP@0.5 en la imagen evaluada.
- En video, el factor RT y el % de frames descartados son tan importantes como el mAP, porque determinan si el sistema es utilizable en vivo.
- **El hardware condiciona los resultados.** En CPU ningún modelo llegó a tiempo real; la alternativa es bajar el tamaño de entrada o procesar solo algunos frames.

---

## Limitaciones y pendientes

**Limitaciones**

- Modelos preentrenados en COCO (80 clases): lo que esté fuera no se detecta (AP = 0).
- El mAP de una sola imagen no es representativo; hay que evaluar un conjunto.
- La calidad en video (Precision, Recall, mAP) sigue sin obtenerse; no concluir nada sobre calidad entre modelos en video.
- Los resultados de Colab y los locales no son comparables; los de CPU dependen del procesador de la laptop.
- En video con YOLO solo se probó `yolo11n` en GPU; `yolo11s` únicamente en CPU con el video 4K.
- Ultralytics YOLO se distribuye bajo licencia AGPL-3.0; revisar sus condiciones fuera del contexto académico.

---

## Equipo y referencias

- Centeno Mamani, Mirian Lucero
- Colquehuanca Colquehuanca, Jhosemy Brissett
- Hancco Vilca, Yennybel Rocio
- Mamani Apaza, Sadith Leidy

Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection* (2016) · Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks* (2015) · Lin et al., *Microsoft COCO: Common Objects in Context* (2014) · Ultralytics, documentación de YOLO11 y YOLO12: https://docs.ultralytics.com

Proyecto con fines académicos.
