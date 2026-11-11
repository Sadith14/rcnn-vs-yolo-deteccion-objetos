"""Genera gráficas a partir de resultados/metricas_por_frame.csv
Uso: python graficar_metricas.py [carpeta_resultados]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = Path(sys.argv[1] if len(sys.argv) > 1 else "resultados")
df = pd.read_csv(d / "metricas_por_frame.csv")
res = json.loads((d / "resumen_metricas.json").read_text(encoding="utf-8"))

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"YOLO en video — {res['modelo']} ({res['resolucion']})", fontsize=13)

ax[0, 0].plot(df.frame, df.latencia_ms, lw=0.8, color="#3b6ea5")
ax[0, 0].axhline(df.latencia_ms.mean(), ls="--", color="#c0392b",
                 label=f"media {df.latencia_ms.mean():.1f} ms")
ax[0, 0].axhline(1000 / res["fps_fuente"], ls=":", color="#555",
                 label=f"presupuesto tiempo real ({1000 / res['fps_fuente']:.1f} ms)")
ax[0, 0].set(title="Latencia de inferencia por frame", xlabel="frame", ylabel="ms")
ax[0, 0].legend()

ax[0, 1].plot(df.frame, df.fps.rolling(15, min_periods=1).mean(), color="#2e8b57")
ax[0, 1].axhline(res["fps_fuente"], ls=":", color="#555", label="FPS del video")
ax[0, 1].set(title="FPS (media móvil 15 frames)", xlabel="frame", ylabel="FPS")
ax[0, 1].legend()

ax[1, 0].plot(df.frame, df.n_detecciones, lw=0.8, color="#8e44ad")
ax[1, 0].set(title=f"Detecciones por frame (inestabilidad = {res['inestabilidad_detecciones']})",
             xlabel="frame", ylabel="nº objetos")

cls = pd.Series(res["detecciones_por_clase"]).head(10)
ax[1, 1].barh(cls.index[::-1], cls.values[::-1], color="#d68910")
ax[1, 1].set(title="Detecciones por clase (top 10)", xlabel="nº detecciones")

plt.tight_layout()
plt.savefig(d / "graficas_metricas.png", dpi=130)
print("Guardado:", d / "graficas_metricas.png")
