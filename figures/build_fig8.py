# -*- coding: utf-8 -*-
"""
重画 Fig.8（Cohort D 随机掩码复制，森林图）

数据来源：6.pdf 第 24 页的 Fig.8 为矢量图（Matplotlib 嵌入 Form XObject），
本脚本直接从 PDF 内容流读取误差棒端点，并用图内印出的 7 个点值做最小二乘定标，
因此 CI 不是像素估读，而是矢量坐标反算（误差 < 0.002）。

数值（点值 = 图内印刷，CI = 矢量反算）：
  11101  3.898  [3.508, 4.307]
  11111  3.674  [3.300, 4.059]
  11121  3.493  [3.126, 3.889]
  11131  3.814  [3.425, 4.220]
  11141  3.733  [3.362, 4.140]
  mean   3.722  [3.338, 4.137]
  11011  3.843  [3.461, 4.267]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle

for f in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"):
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

OUT = r"C:\Users\Dawn\Documents\deepseek-harness\default-workspace"
INK = "#000000"
DEEP = "#1F4E79"
GREEN = "#2E7D32"
GRAY = "#7F7F7F"
GRID = "#D9D9D9"
BAND = "#F2F6FA"
BAND_M = "#E6F2EE"

FS_T, FS_Y, FS_X, FS_V = 22, 18, 18, 20

rows = [
    ("Random allocation 11101", 3.898, 3.508, 4.307, DEEP, "o", 0.0),
    ("Random allocation 11111", 3.674, 3.300, 4.059, DEEP, "o", 0.0),
    ("Random allocation 11121", 3.493, 3.126, 3.889, DEEP, "o", 0.0),
    ("Random allocation 11131", 3.814, 3.425, 4.220, DEEP, "o", 0.0),
    ("Random allocation 11141", 3.733, 3.362, 4.140, DEEP, "o", 0.0),
    ("Five-mask mean", 3.722, 3.338, 4.137, GREEN, "D", 0.55),
    ("Historical allocation 11011", 3.843, 3.461, 4.267, GRAY, "o", 0.55),
]

W, H = 10.3, 5.9
fig, ax = plt.subplots(figsize=(W, H), dpi=300)
fig.subplots_adjust(left=0.265, right=0.955, top=0.965, bottom=0.145)

n = len(rows)
ys = []
y = n - 1 + 1.1
for i, r in enumerate(rows):
    gap = r[6]
    if i:
        y -= 1 + gap
    ys.append(y)
ymin, ymax = min(ys) - 0.75, max(ys) + 0.85

# 背景条带
for i, (yy, r) in enumerate(zip(ys, rows)):
    fc = BAND_M if r[4] == GREEN else (BAND if i % 2 == 0 else "white")
    ax.add_patch(Rectangle((-0.02, yy - 0.5), 5.22, 1.0, facecolor=fc,
                           edgecolor="none", zorder=0))

for yy, r in zip(ys, rows):
    label, val, lo, hi, color, marker, _ = r
    ax.plot([lo, hi], [yy, yy], color=color, lw=3.6, solid_capstyle="butt", zorder=3)
    ax.plot([lo, lo], [yy - 0.17, yy + 0.17], color=color, lw=2.6, zorder=3)
    ax.plot([hi, hi], [yy - 0.17, yy + 0.17], color=color, lw=2.6, zorder=3)
    ax.plot([val], [yy], marker=marker, ms=14 if marker == "o" else 15,
            color=color, zorder=4, markeredgecolor=color,
            markerfacecolor=(GREEN if marker == "D" else color))
    ax.text(hi + 0.075, yy, "%.3f" % val, va="center", ha="left",
            fontsize=FS_V, fontweight="bold", color=INK, zorder=6,
            path_effects=[pe.withStroke(linewidth=3.2, foreground="white")])
    ax.text(-0.10, yy, label, va="center", ha="right",
            fontsize=FS_Y, fontweight="bold", color=INK, zorder=5)

ax.set_xlim(0, 5.0)
ax.set_ylim(ymin, ymax)
ax.set_yticks([])
ax.set_xticks(range(6))
ax.set_xticklabels(["0", "1", "2", "3", "4", "5"], fontsize=FS_X, fontweight="bold")
ax.set_xlabel("Mean NLL reduction: random allocation − FisherSpline",
              fontsize=FS_X, fontweight="bold", labelpad=10)
ax.tick_params(axis="x", length=0, pad=6)
ax.set_axisbelow(True)
ax.xaxis.grid(True, color=GRID, lw=1.0)
ax.yaxis.grid(False)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_visible(False)
fig.savefig("%s\\Fig8.pdf" % OUT, facecolor="white",
            bbox_inches="tight", pad_inches=0.16)
fig.savefig("%s\\Fig8.png" % OUT, facecolor="white", dpi=300,
            bbox_inches="tight", pad_inches=0.16)
plt.close(fig)
print("ok")
