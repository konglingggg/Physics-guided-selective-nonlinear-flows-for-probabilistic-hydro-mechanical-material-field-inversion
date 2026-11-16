# -*- coding: utf-8 -*-
"""
Fig.2 重画：maximum relative directional error vs. centered finite-difference step
数据来源：6.pdf p11 的位图 figsrc/obj98.jpg（2070×1770），log-log 轴。

定标：
  纵轴 5 条 decade 网格线（x=1500 列测得）：y = 385.5, 658.5, 930.5, 1202.5, 1475.5
  对应 1e-6 … 1e-10，间距 272.5 px/decade
  => log10(v) = -6 + (385.5 - y)/272.4
  用正文锚点“Biot α 首点 = 1.30e-8”反查：预测 y=899.5，实测 899.0，偏差 0.5 px

数值（标记中心，开符号取上下两段中点）：
  512D material directions: (1e-4, 1.22e-9), (1e-3, 2.49e-10), (2e-3, 1.01e-9)
  Biot α direction:         (1e-4, 1.30e-8), (1e-3, 1.21e-6), (2e-3, 4.29e-6)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, NullFormatter

for f in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"):
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                     "axes.unicode_minus": False, "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "mathtext.default": "bf"})

OUT = r"C:\Users\Dawn\Documents\deepseek-harness\default-workspace"
INK = "#000000"
DEEP = "#1F4E79"
RED = "#B03A2E"
GRID = "#D9D9D9"

x1 = [1e-4, 1e-3, 2e-3]
y_512 = [1.22e-9, 2.49e-10, 1.01e-9]
y_biot = [1.30e-8, 1.21e-6, 4.29e-6]

fig, ax = plt.subplots(figsize=(8.6, 6.7), dpi=300)
fig.subplots_adjust(left=0.165, right=0.975, top=0.975, bottom=0.155)

ax.set_xscale("log")
ax.set_yscale("log")

# 装饰性红色阴影（按作者要求保留）：沿 α 折线向下渐隐，不参与数据
xg = np.logspace(np.log10(1e-4), np.log10(2e-3), 260)
logy = np.interp(np.log10(xg), np.log10(x1), np.log10(y_biot))
yg = np.logspace(np.log10(2e-11), np.log10(2e-5), 260)
X, Y = np.meshgrid(xg, yg)
top = 10 ** logy[None, :]
dlog = np.log10(top) - np.log10(Y)
alpha = np.where((Y < top) & (dlog < 3.0), np.exp(-dlog / 1.1) * 0.20, 0.0)
band_cmap = LinearSegmentedColormap.from_list(
    "biot_band", [(0.86, 0.42, 0.38, 0.0), (0.86, 0.42, 0.38, 1.0)])
ax.pcolormesh(X, Y, alpha, cmap=band_cmap, vmin=0.0, vmax=1.0,
              shading="auto", zorder=0.4, rasterized=True)

ax.plot(x1, y_512, color=DEEP, lw=2.9, marker="o", ms=10.5,
        mfc="white", mec=DEEP, mew=2.4, label="512D material directions", zorder=4,
        clip_on=False)
ax.plot(x1, y_biot, color=RED, lw=2.9, marker="s", ms=10.0,
        mfc="white", mec=RED, mew=2.4, label="Biot $\\alpha$ direction", zorder=4,
        clip_on=False)

ax.set_xlim(7.0e-5, 2.8e-3)
ax.set_ylim(3.0e-11, 1.2e-5)

ax.set_xticks(x1)
ax.set_xticklabels([r"$10^{-4}$", r"$10^{-3}$", r"$2\times10^{-3}$"], fontsize=14)
ax.set_yticks([1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
ax.set_yticklabels([r"$10^{-10}$", r"$10^{-9}$", r"$10^{-8}$",
                    r"$10^{-7}$", r"$10^{-6}$"], fontsize=18, fontweight="bold")

ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.tick_params(which="major", length=5.5, width=1.1, colors=INK, pad=7)
ax.tick_params(which="minor", length=2.6, width=0.8, colors=INK)

ax.set_xlabel("Centered finite-difference step", fontsize=18, fontweight="bold", color=INK, labelpad=11)
ax.set_ylabel("Maximum relative directional error", fontsize=18, fontweight="bold", color=INK, labelpad=12)
ax.set_axisbelow(True)
ax.grid(True, which="major", color=GRID, lw=1.0, ls=(0, (6, 4)))
ax.grid(False, which="minor")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#333333")
    ax.spines[s].set_linewidth(1.2)

ax.legend(loc="upper left", prop={"family": "Arial", "size": 16, "weight": "bold"},
          frameon=False, handlelength=2.8, labelspacing=0.55, borderaxespad=0.35)


fig.savefig("%s\\Fig2.pdf" % OUT, facecolor="white",
            bbox_inches="tight", pad_inches=0.16)
fig.savefig("%s\\Fig2.png" % OUT, facecolor="white", dpi=300,
            bbox_inches="tight", pad_inches=0.16)
plt.close(fig)
print("ok")
