# -*- coding: utf-8 -*-
"""Fig.1：16×16 场 + 传感器/源点（坐标取自 13.pdf p10 的矢量标记，字号加大加粗）。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.font_manager as fm

for f in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"):
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

WS = r"C:\Users\Dawn\Documents\deepseek-harness\default-workspace"

# 由 13.pdf p10 图内标记反算（以 Injection(5,5)、Production(11,11) 两点定标）
sensors = [(8, 13), (3, 12), (10, 12), (6, 11), (2, 10), (11, 9),
           (7, 8), (10, 7), (3, 6), (6, 5), (2, 4), (9, 4), (2, 2), (5, 2)]
disps = [(9, 13), (3, 11), (12, 11), (8, 9), (4, 7), (7, 5), (12, 4), (3, 3)]
inj, prod = (5, 5), (11, 11)

field = Image.open(WS + r"\fig1_field_15.png")
fig, ax = plt.subplots(figsize=(10.6, 9.9), dpi=300)
fig.subplots_adjust(left=0.125, right=0.845, top=0.930, bottom=0.115)
ax.imshow(field, extent=(0, 16, 0, 15), origin="upper", aspect="auto",
          interpolation="nearest", zorder=1)
for g in range(17):
    ax.axvline(g, color="#FFFFFF", lw=0.5, alpha=0.35, zorder=2)
for g in range(16):
    ax.axhline(g, color="#FFFFFF", lw=0.5, alpha=0.35, zorder=2)

ax.scatter([p[0] for p in sensors], [p[1] for p in sensors], s=280,
           marker="o", facecolor="#2E75B6", edgecolor="white", linewidth=1.8, zorder=6)
ax.scatter([p[0] for p in disps], [p[1] for p in disps], s=260,
           marker="s", facecolor="#7030A0", edgecolor="white", linewidth=1.8, zorder=6)
ax.plot([inj[0]], [inj[1]], marker="*", ms=40, mfc="#FFD700", mec="black", mew=1.6, zorder=7)
ax.plot([prod[0]], [prod[1]], marker="*", ms=40, mfc="#8B1A1A", mec="black", mew=1.6, zorder=7)

ax.set_xlim(0, 16)
ax.set_ylim(0, 15)
ax.set_xticks(range(0, 17))
ax.set_yticks(range(0, 16))
ax.set_xticklabels([str(i) for i in range(17)], fontsize=18, fontweight="bold")
ax.set_yticklabels([str(i) for i in range(16)], fontsize=18, fontweight="bold")
ax.set_xlabel("Grid-column index", fontsize=22, fontweight="bold", labelpad=12)
ax.set_ylabel("Grid-row index", fontsize=22, fontweight="bold", labelpad=12)
ax.set_title("(a)", fontsize=20,
             fontweight="bold", pad=16)
for s in ax.spines.values():
    s.set_linewidth(1.3)

sm = ScalarMappable(norm=Normalize(vmin=0.30, vmax=0.65), cmap="RdYlBu_r")
cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02,
                    ticks=np.arange(0.30, 0.66, 0.05))
cbar.set_label("Normalized Physical Field", fontsize=18, fontweight="bold", labelpad=10)
cbar.ax.tick_params(labelsize=18)
for t in cbar.ax.get_yticklabels():
    t.set_fontweight("bold")

leg = [Line2D([0], [0], marker="o", color="none", mfc="#2E75B6", mec="white",
              ms=16, label="Pressure sensors"),
       Line2D([0], [0], marker="s", color="none", mfc="#7030A0", mec="white",
              ms=16, label="Displacement sensors"),
       Line2D([0], [0], marker="*", color="none", mfc="#FFD700", mec="black",
              ms=22, label="Injection (5, 5)"),
       Line2D([0], [0], marker="*", color="none", mfc="#8B1A1A", mec="black",
              ms=22, label="Production (11, 11)")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.11),
          ncol=2, frameon=False,
          prop={"family": "Arial", "size": 18, "weight": "bold"},
          handletextpad=0.5, columnspacing=1.8)

fig.savefig(WS + r"\Fig1.pdf", facecolor="white", bbox_inches="tight", pad_inches=0.15)
fig.savefig(WS + r"\Fig1.png", facecolor="white", dpi=300, bbox_inches="tight", pad_inches=0.15)
print("saved fig1_sensors (fonts enlarged, markers recalibrated)")
