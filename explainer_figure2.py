"""
The second explainer figure: a shape that looks perfectly ordinary, and the
defect that only shows up when you ask the third question about it.

Left panel is the blowup profile itself. Nothing about it looks unusual.
Right panel is its curvature, the rate at which its slope is changing, near
the point where the fluid's own velocity is zero. There the curvature jumps
from one value to its exact negative, which is the sense in which the shape is
not smooth.

    python explainer_figure2.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8
NAVY, GOLD, RED = "#1A3A5C", "#8C6D2F", "#8C2F39"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

g, out = simulate(A, n=2048, w_max=1e6)
f0, _ = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, A)
print(f"profile: converged {ok}, residual {hist[-1]:.2e}, "
      f"peak {sup(g, f):.6f}")

U, Hf, _ = dg.fields(g, f)
fxx = g.bwd(g.fwd(f) * g.dmul * g.dmul)

# The stagnation point that carries the singularity is the one where the
# curvature is odd about the crossing.
zeros = np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1)))
ys = None
for i in zeros:
    c = Hf[i]
    mu = (c - 1.0) / (A * c)
    if mu > 1.5:
        ys = g.x[i]
print(f"singular stagnation point at y = {ys:.5f}")

# Centre both panels on that point.
y = (g.x - ys + np.pi) % (2 * np.pi) - np.pi
order = np.argsort(y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4))

ax1.plot(y[order], f[order], color=NAVY, lw=1.8)
ax1.axvline(0.0, ls="--", color=GOLD, lw=1.2)
ax1.annotate("the flow is at rest here", (0.0, f[order][len(order) // 2]),
             xytext=(0.55, 3.1), fontsize=8.5, color=GOLD,
             arrowprops=dict(arrowstyle="->", color=GOLD, lw=1))
ax1.set_xlim(-np.pi, np.pi)
ax1.set_xlabel("position, centred on that point")
ax1.set_ylabel("the settled shape")
ax1.set_title("The shape looks perfectly ordinary", fontsize=9.5)

win = np.abs(y) < 0.55
ax2.plot(y[win][np.argsort(y[win])], fxx[win][np.argsort(y[win])],
         color=RED, lw=1.6)
ax2.axvline(0.0, ls="--", color=GOLD, lw=1.2)
ax2.axhline(0.0, color="k", lw=0.6)
ax2.annotate("", xy=(0.0, 2.27), xytext=(0.0, -2.27),
             arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.4))
ax2.text(0.06, 0.0, "the jump", fontsize=8.5, color=GOLD, va="center")
ax2.set_xlim(-0.55, 0.55)
ax2.set_xlabel("position, zoomed in on the same point")
ax2.set_ylabel("its curvature")
ax2.set_title("Its curvature jumps across that point", fontsize=9.5)

fig.tight_layout()
fig.savefig("fig_kink.png")
print("wrote fig_kink.png")
