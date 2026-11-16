# -*- coding: utf-8 -*-
"""Fig.3 journal version: 800 x 1000 pt, Arial only, plain-text formulas (no mathtext)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.font_manager as fm
import numpy as np

for _f in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"):
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42,
                     "axes.unicode_minus": False})

W, H = 800, 1000
fig = plt.figure(figsize=(W/72, H/72), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

INK = "#16212B"
G_D, G_L = "#3E5C33", "#EEF4E9"
O_D, O_L = "#A9601F", "#FBF0E4"
B_D, B_L = "#1F4E79", "#E9F2FB"
Y_D, Y_L = "#7F6000", "#FFF7DD"

def panel(x, y, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                                facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=1))
def card(x, y, w, h, ec, fc="#FFFFFF", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
def T(x, y, s, fs=16, bold=False, ha="left", c=INK, z=7):
    ax.text(x, y, s, fontsize=fs, fontweight="bold" if bold else "normal",
            ha=ha, va="center", color=c, zorder=z)
def num(x, y, n):
    ax.add_patch(plt.Circle((x, y), 11, facecolor="#FFFFFF", edgecolor=INK, lw=1.2, zorder=8))
    T(x, y-0.5, str(n), 16, True, "center", z=9)

panel(14, 520, 378, 466, G_L, G_D)
panel(408, 520, 378, 466, B_L, B_D)
panel(14, 14, 378, 494, O_L, O_D)
panel(408, 14, 378, 494, Y_L, Y_D)

# ---------------- 1 Inputs ----------------
num(34, 958, 1)
T(52, 958, "Inputs and Physical Setup", 18, True)
T(52, 936, "(Sections 3.1–3.2)", 16, False, c="#444444")
T(28, 906, "Unknown material field", 16, True)
rng = np.random.default_rng(5)
for x0, lab in [(32, "log k(x)"), (142, "log E(x)")]:
    ax.imshow(rng.random((8, 8)), extent=(x0, x0+86, 830, 894), origin="lower",
              cmap="RdYlBu_r", aspect="auto", zorder=5, interpolation="nearest")
    ax.add_patch(Rectangle((x0, 830), 86, 64, fill=False, ec="#7F7F7F", lw=0.8, zorder=6))
    T(x0+43, 816, lab, 16, ha="center")
T(28, 786, "m = [log k, log E],  m in R⁵¹²", 16)
T(28, 764, "two 16 × 16 fields · 512 unknowns", 16)

card(26, 640, 354, 108, G_D, lw=1.0)
T(38, 730, "Coupled nonlinear transient Biot model", 16, True)
T(38, 704, "momentum + fluid-mass balance, α = 0.7", 16)
T(38, 680, "3 backward-Euler steps; one solve gives", 16)
T(38, 656, "p(x), u(x)", 16)

card(26, 540, 354, 86, G_D, lw=1.0)
T(38, 604, "42 pressure + 48 displacement", 16, True)
T(38, 580, "y = (y_H, y_M),  y in R⁹⁰", 16)
T(38, 556, "two sensor families, one solution", 16)

# ---------------- 2 Gaussian ----------------
num(428, 958, 2)
T(446, 958, "Gaussian Anchor and Sensitivity", 18, True)
T(446, 936, "(Section 3.4)", 16, False, c="#444444")
T(424, 906, "Nominal sensitivity", 16, True)
T(424, 880, "J0 = D G_Biot(m0)", 16)
T(424, 854, "A = Γ^(-1/2) J0 L,  A in R^(90×512)", 16)
T(424, 828, "r = Γ^(-1/2) [ y − G_Biot(m0) ]", 16)
T(424, 802, "fixed reference · full 90 × 512 Jacobian", 16, c="#333333")
T(424, 778, "full 512 material coordinates retained", 16, c="#333333")
T(424, 742, "Tangent Gaussian anchor", 16, True)
T(424, 714, "q_G(z|y) = N(μ_G(y), C_G)", 16)
T(424, 688, "C_G = (I + Aᵀ A)^(-1)", 16)
T(424, 662, "μ_G(y) = C_G Aᵀ r", 16)
T(424, 632, "linearize once at the fixed reference", 16)
T(424, 608, "same reference across comparisons", 16)
T(424, 584, "a physical anchor, not a reduction", 16)
T(424, 560, "to 90 or fewer coordinates", 16)

# ---------------- 3 Operator ----------------
num(34, 482, 3)
T(52, 482, "Operator Design", 18, True)
T(52, 460, "(Sections 4.1–4.4)", 16, False, c="#444444")
T(28, 430, "one shared operator A · fixed before training", 16, True)
T(28, 408, "not a learned surrogate of the PDE", 16)
T(28, 380, "Role 1 · Material information geometry", 16, True)
T(28, 354, "F = Aᵀ A = V Λ Vᵀ", 16)
T(28, 330, "F_θ0 = λ_i v_i", 16)
T(28, 306, "w = Pᵀ Vᵀ z,  w in R⁵¹²", 16)
T(28, 282, "variance along v_i: (1 + λ_i)^(-1)", 16)
T(28, 250, "Role 2 · H/M observation conditioning", 16, True)
T(28, 224, "SVD per sensor family", 16)
T(28, 200, "c_S = (I + Σ_S²)^(-1/2) U_Sᵀ r_s", 16)
T(28, 176, "c = (c_H, c_M),  c in R⁹⁰", 16)
T(28, 152, "sensor-specific scaling", 16)
T(28, 128, "Role 3 · Transformation allocation", 16, True)
T(28, 102, "top 24 of 256 (eigen-rank of F)", 16)
T(28, 78, "set fixed in all 4 layers", 16)
T(28, 54, "24 spline · 232 affine", 16)
T(28, 30, "capacity follows physics; log-det tractable", 16)

# ---------------- 4 Flow ----------------
num(428, 482, 4)
T(446, 482, "Conditional Normalizing Flow", 18, True)
ys = [424, 374, 324, 274]
for i, y in enumerate(ys):
    T(420, y+21, "L%d" % (i+1), 16, True)
    card(448, y, 92, 42, Y_D, lw=0.8)
    T(494, y+27, "passive", 16, ha="center"); T(494, y+11, "256-D", 16, ha="center")
    card(546, y, 64, 42, Y_D, lw=0.8)
    T(578, y+27, "NN", 16, ha="center"); T(578, y+11, "(a, c)", 16, ha="center")
    card(616, y, 92, 42, Y_D, lw=0.8)
    T(662, y+27, "active", 16, ha="center"); T(662, y+11, "256-D", 16, ha="center")
    T(714, y+21, "24/232", 16)

T(424, 244, "Compose four invertible layers", 16, True)
T(424, 220, "z ~ q_G → T_θ1 → ... → w", 16)
T(424, 198, "T_θ = T_θ4 · ... · T_θ1", 16)
T(424, 176, "m = m0 + LVPw", 16)
T(424, 150, "Exact 512-D conditional density", 16, True)
T(418, 126, "log q_θ(w|y) = log q_G(T_θ(w;c)|y)", 16)
T(430, 104, "+ log |det DT_θ(w;c)|", 16)
T(418, 80, "q_θ,m(m|y) = q_θ,w(w) |det L|", 16)
T(424, 54, "exactly evaluable · full-dim · log-det tractable", 16)
T(424, 28, "posterior means: mean log k · mean log E", 16)

fig.savefig("Fig3.pdf", facecolor="white", bbox_inches=None)
fig.savefig("Fig3.png", facecolor="white", dpi=300, bbox_inches=None)
print("saved")
