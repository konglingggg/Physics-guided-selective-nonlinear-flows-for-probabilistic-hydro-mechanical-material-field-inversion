# -*- coding: utf-8 -*-
"""
Fig.7 / Fig.9 按 15.pdf 图内排版重排，只放大字号。
Fig.7：数值+CI 紧贴误差棒右侧；Fig.9：数值紧贴误差棒右侧。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"):
    try: fm.fontManager.addfont(f)
    except Exception: pass
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "axes.unicode_minus": False})

OUT = r"C:\Users\Dawn\Documents\deepseek-harness\default-workspace"
INK = "#000000"
DEEP = "#1F4E79"
GRID = "#D9D9D9"

FS_LAB, FS_VAL, FS_TICK, FS_AX = 13, 12, 12, 12.5


def forest(fname, rows, xlim, xticks, xlabel, w, h, show_ci):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(w, h), dpi=300)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.96, bottom=0.20)

    for i, (label, val, lo, hi) in enumerate(rows):
        y = n - 1 - i
        ax.plot([lo, hi], [y, y], color=DEEP, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.plot([lo, lo], [y - 0.13, y + 0.13], color=DEEP, lw=1.8, zorder=3)
        ax.plot([hi, hi], [y - 0.13, y + 0.13], color=DEEP, lw=1.8, zorder=3)
        ax.plot([val], [y], marker="o", ms=8.5, color=DEEP, zorder=4)
        txt = ("%.3f [%.3f, %.3f]" % (val, lo, hi)) if show_ci else ("%.3f" % val)
        ax.text(hi + (xlim[1] - xlim[0]) * 0.012, y, txt,
                va="center", ha="left", fontsize=FS_VAL, color=INK)
        ax.text(xlim[0] - (xlim[1] - xlim[0]) * 0.015, y, label,
                va="center", ha="right", fontsize=FS_LAB, color=INK)

    ax.set_xlim(xlim)
    ax.set_ylim(-0.65, n - 0.35)
    ax.set_yticks([])
    ax.set_xticks(xticks)
    ax.set_xticklabels(["%g" % t for t in xticks], fontsize=FS_TICK)
    ax.set_xlabel(xlabel, fontsize=FS_AX, labelpad=8)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, lw=0.9)
    ax.yaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#BFBFBF")
    ax.spines["bottom"].set_linewidth(1.0)

    fig.savefig("%s\\%s.pdf" % (OUT, fname), facecolor="white",
                bbox_inches="tight", pad_inches=0.12)
    fig.savefig("%s\\%s.png" % (OUT, fname), facecolor="white", dpi=300,
                bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("ok", fname)


rows7 = [
    ("Matched full spline", 1.872, 1.714, 2.034),
    ("Validation-tuned\nwide full spline", 1.827, 1.660, 2.006),
    ("Frozen affine flow", 2.487, 2.131, 2.947),
]
forest("Fig7", rows7, (0.0, 4.0),
       [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
       "Conditional NLL reduction (comparator − FisherSpline)", 7.8, 3.0, True)

rows9 = [
    ("Dual-geometry affine", 2.398, 2.086, 2.703),
    ("Matched full spline", 1.878, 1.726, 2.036),
    ("Tuned wider full spline", 1.835, 1.670, 2.003),
    ("WeakSpline", 3.446, 3.114, 3.783),
    ("RandomSpline", 3.596, 3.251, 3.947),
]
forest("Fig9", rows9, (1.5, 4.0),
       [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
       "Paired NLL reduction (comparator − FisherSpline)", 7.8, 3.9, False)
