# -*- coding: utf-8 -*-
"""Fig.6 重画：development-validation NLL vs. training epoch（数值由 obj300 位图量取）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"):
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

OUT = r"C:\Users\Dawn\Documents\deepseek-harness\default-workspace"
INK = "#000000"
GRID = "#D9D9D9"

SERIES = [
    ("FisherSpline 8e-4", "#1F4E79", "o",
     [239.03, 238.23, 243.03, 246.58, 251.04, 255.65, 261.44, 267.86]),
    ("Matched 8e-4", "#E8720C", "s",
     [245.02, 241.69, 244.03, 247.04]),
    ("Matched 2e-4", "#2E7D32", "^",
     [247.49, 240.98, 240.26, 240.99, 241.99, 242.79, 243.30, 243.99]),
    ("Wide 2e-4", "#B03A2E", "D",
     [243.53, 240.03, 241.04, 243.63, 246.53, 249.04]),
    ("Wide 1e-4", "#6C5B9E", "v",
     [241.54, 238.57, 238.67, 239.07, 240.07, 241.06, 242.08, 243.06]),
]

fig, ax = plt.subplots(figsize=(9.8, 6.2), dpi=300)
fig.subplots_adjust(left=0.115, right=0.975, top=0.965, bottom=0.155)

for name, color, mk, vals in SERIES:
    xs = list(range(2, 2 + 2 * len(vals), 2))
    ax.plot(xs, vals, color=color, lw=2.9, marker=mk, ms=9.8,
            mfc="white" if mk == "o" else color, mec=color, mew=2.4,
            label=name, zorder=3, clip_on=False)

ax.set_xlim(1.2, 16.8)
ax.set_ylim(237.5, 270.5)
ax.set_xticks(range(2, 17, 2))
ax.set_yticks(range(240, 271, 5))
ax.tick_params(labelsize=17, length=0, pad=7, colors=INK)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight("bold")
ax.set_xlabel("Training epoch", fontsize=19, fontweight="bold", color=INK, labelpad=11)
ax.set_ylabel("Development-validation NLL", fontsize=19, fontweight="bold", color=INK, labelpad=10)
ax.set_axisbelow(True)
ax.yaxis.grid(True, color=GRID, lw=1.0, ls=(0, (6, 4)))
ax.xaxis.grid(False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#9E9E9E")
    ax.spines[s].set_linewidth(1.0)
leg = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0,
                prop={"family": "Arial", "size": 17, "weight": "bold"},
                frameon=False, handlelength=2.6, labelspacing=0.5)

# 让图例第一行（FisherSpline）与蓝线最高点精确齐平
anchor = [0.02, 0.98]
for _ in range(4):
    fig.canvas.draw()
    bb = leg.get_texts()[0].get_window_extent(fig.canvas.get_renderer())
    cy = (bb.y0 + bb.y1) / 2.0
    target = ax.transData.transform((16, SERIES[0][3][-1]))[1]
    if abs(target - cy) < 0.5:
        break
    anchor[1] += (target - cy) / ax.bbox.height
    leg.set_bbox_to_anchor((anchor[0], anchor[1]), transform=ax.transAxes)

fig.savefig("%s\\Fig6.pdf" % OUT, facecolor="white",
            bbox_inches="tight", pad_inches=0.16)
fig.savefig("%s\\Fig6.png" % OUT, facecolor="white", dpi=300,
            bbox_inches="tight", pad_inches=0.16)
plt.close(fig)
print("ok")
