# -*- coding: utf-8 -*-
"""
重画 4.pdf 中的三张条形图（Fig. 4 / Fig. 5 / Fig. 10）
本轮改动：
  1. 字号整体加大：16 -> 18（标题 20）
  2. Fig.4 / Fig.5 的基线柱由灰(#D6DCE4)改为与 Fig.10 相同的浅蓝(#8FAADC)
  3. Fig.5 加宽图幅、收窄柱宽以拉大间距，横轴标签改回水平（字号单独降到 13）
输出：仅 PDF（矢量）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.font_manager as fm

for _f in (r"C:\Windows\Fontsrial.ttf", r"C:\Windows\Fontsrialbd.ttf"):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

BASE_C = "#8FAADC"   # 浅蓝：基线 / 对照（本轮由灰改来，与 Fig.10 的 Cohort B 同色）
DEEP   = "#1F4E79"
BAND   = "#1F3864"
ORANGE = "#E8720C"
GRID   = "#D9D9D9"
BASE_L = "#7F7F7F"

TICK_FS, LABEL_FS, VALUE_FS, LEG_FS, TITLE_FS = 17, 17, 17, 16, 20
X5_FS = 16

OUT = r"C:\Users\Dawn\Documents\deepseek-harness\default-workspace"


def style_axes(ax, ylabel, ymin, ymax, yticks, xticklabels, xfs=None):
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, labelpad=12)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, fontsize=xfs or TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS, length=0)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.1)
    ax.xaxis.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BFBFBF")
        ax.spines[s].set_linewidth(1.1)


def ribbon(fig, text):
    fig.text(0.040, 0.948, text, fontsize=TITLE_FS, fontweight="bold",
             color="white", ha="left", va="center", zorder=10,
             bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.12",
                       facecolor=BAND, edgecolor="none"))


def legend_top(ax, handles, y=1.13):
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y),
              ncol=len(handles), frameon=False, fontsize=LEG_FS,
              handlelength=2.0, handleheight=1.1,
              columnspacing=2.4, handletextpad=0.7)


def draw(ax, vals, baseline, colors, digits, bottom, label_pad, width):
    ax.bar(range(len(vals)), [v - bottom for v in vals], bottom=bottom,
           width=width, color=colors, zorder=3, edgecolor="white", linewidth=1.0)
    if baseline is not None:
        ax.axhline(baseline, color=BASE_L, linewidth=1.7,
                   linestyle=(0, (5, 3)), zorder=2)
    texts = []
    for i, v in enumerate(vals):
        texts.append(ax.text(i, v + label_pad, f"{v:.{digits}f}",
                             ha="center", va="bottom", fontsize=VALUE_FS,
                             fontweight="bold", color="#1A1A1A", zorder=9))
    return texts


def box_around(ax, fig, idx, text_artist, bottom, half, top_margin):
    fig.canvas.draw()
    bb = text_artist.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (_, y0), (_, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    ax.add_patch(Rectangle((idx - half, bottom), 2 * half,
                           (y1 + top_margin) - bottom, fill=False,
                           edgecolor=ORANGE, linewidth=3.2,
                           linestyle=(0, (4, 2.6)), zorder=8))


# ===================== Fig. 4 =====================
labels4 = ["Nominal tangent\nGaussian", "Random\n24-spline", "Dual-geometry\naffine",
           "Rank-balanced\naffine", "FisherSpline", "HMInfoSpline"]
vals4 = [255.709, 241.535, 240.399, 240.087, 237.811, 237.802]
cols4 = [BASE_C] * 4 + [DEEP] * 2
fig, ax = plt.subplots(figsize=(9.9, 4.95), dpi=300)
fig.subplots_adjust(left=0.010, right=0.985, top=0.86, bottom=0.17)
style_axes(ax, "Held-out conditional NLL\n(lower is better)", 230, 260,
           [230, 234, 238, 242, 246, 250, 254, 258], labels4)
t4 = draw(ax, vals4, 240.399, cols4, 3, 230, 0.55, 0.50)
for lbl in ax.get_xticklabels():
    lbl.set_rotation(28); lbl.set_ha("right"); lbl.set_rotation_mode("anchor")
fig.subplots_adjust(bottom=0.31)
box_around(ax, fig, 5, t4[5], 230, half=0.37, top_margin=0.40)
legend_top(ax, [
    Line2D([0], [0], color=BASE_L, linewidth=1.8, linestyle=(0, (5, 3)),
           label="Frozen affine baseline"),
    Rectangle((0, 0), 1, 1, facecolor=BASE_C, edgecolor="none",
              label="Baseline models"),
    Rectangle((0, 0), 1, 1, facecolor=DEEP, edgecolor="none", label="Proposed")])
fig.savefig(f"{OUT}\\Fig4.pdf", facecolor="white",
            bbox_inches="tight", pad_inches=0.16)
fig.savefig(f"{OUT}\\Fig4.png", facecolor="white", dpi=300,
            bbox_inches="tight", pad_inches=0.16)
plt.close(fig)

# ===================== Fig. 5 =====================
labels5 = ["HMInfoSpline", "FisherSpline", "FullSplineWide (tuned)",
           "FullSplineMatched", "Dual-geometry affine", "WeakSpline", "RandomSpline"]
vals5 = [238.149, 238.162, 239.997, 240.040, 240.560, 241.607, 241.758]
cols5 = [DEEP] * 2 + [BASE_C] * 5
fig, ax = plt.subplots(figsize=(9.9, 5.2), dpi=300)
fig.subplots_adjust(left=0.010, right=0.988, top=0.86, bottom=0.15)
style_axes(ax, "Held-out conditional NLL\n(lower is better)", 236, 243.6,
           [236, 237, 238, 239, 240, 241, 242, 243], labels5, xfs=X5_FS)
t5 = draw(ax, vals5, 240.560, cols5, 3, 236, 0.045, 0.38)
for lbl in ax.get_xticklabels():
    lbl.set_rotation(28); lbl.set_ha("right"); lbl.set_rotation_mode("anchor")
fig.subplots_adjust(bottom=0.27)
box_around(ax, fig, 0, t5[0], 236, half=0.30, top_margin=0.075)
legend_top(ax, [
    Line2D([0], [0], color=BASE_L, linewidth=1.8, linestyle=(0, (5, 3)),
           label="Frozen affine baseline"),
    Rectangle((0, 0), 1, 1, facecolor=DEEP, edgecolor="none", label="Proposed"),
    Rectangle((0, 0), 1, 1, facecolor=BASE_C, edgecolor="none", label="Controls")])
fig.savefig(f"{OUT}\\Fig5.pdf", facecolor="white",
            bbox_inches="tight", pad_inches=0.16)
fig.savefig(f"{OUT}\\Fig5.png", facecolor="white", dpi=300,
            bbox_inches="tight", pad_inches=0.16)
plt.close(fig)

# ===================== Fig. 10 =====================
labels10 = ["A: affine", "A: random\nspline", "B: matched\nfull spline",
            "B: tuned wider\nfull spline"]
vals10 = [1412 / 2048, 1646 / 2048, 806 / 1024, 778 / 1024]
cols10 = [DEEP, DEEP, BASE_C, BASE_C]
fig, ax = plt.subplots(figsize=(10.6, 5.6), dpi=300)
fig.subplots_adjust(left=0.100, right=0.985, top=0.86, bottom=0.17)
style_axes(ax, "Fraction with lower NLL", 0, 1.0,
           [0, 0.2, 0.4, 0.6, 0.8, 1.0], labels10)
t10 = draw(ax, vals10, 0.5, cols10, 4, 0.0, 0.018, 0.50)
box_around(ax, fig, 1, t10[1], 0.0, half=0.37, top_margin=0.035)
legend_top(ax, [
    Line2D([0], [0], color=BASE_L, linewidth=1.8, linestyle=(0, (5, 3)),
           label="50% random level"),
    Rectangle((0, 0), 1, 1, facecolor=DEEP, edgecolor="none", label="Cohort A"),
    Rectangle((0, 0), 1, 1, facecolor=BASE_C, edgecolor="none", label="Cohort B")])
fig.savefig(f"{OUT}\\Fig10.pdf", facecolor="white",
            bbox_inches="tight", pad_inches=0.16)
fig.savefig(f"{OUT}\\Fig10.png", facecolor="white", dpi=300,
            bbox_inches="tight", pad_inches=0.16)
plt.close(fig)

print("OK: font 18 / baseline bars -> light blue / Fig.5 x labels horizontal")
