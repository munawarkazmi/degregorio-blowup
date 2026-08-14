"""
The one figure the explainer needs: the two kinds of blowup, in real space.

Both regimes have a peak whose height runs away to infinity. The difference is
what happens to its shape on the way. Dividing each snapshot by its own peak
height removes the growth and leaves only the shape, so the two behaviours
separate on sight: at a = 0.4 the curves narrow, at a = 0.8 they lie on top of
one another.

    python explainer_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dg

LEVELS = (10.0, 100.0, 1000.0)
CMAP = ("#3B6EA5", "#B4762B", "#8C2F39")

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), sharey=True)

for ax, a, title in ((axes[0], 0.4, "a = 0.4: the peak narrows"),
                     (axes[1], 0.8, "a = 0.8: the shape stops changing")):
    g = dg.Grid(4096)
    w = np.sin(g.x)
    for colour, lvl in zip(CMAP, LEVELS):
        out = dg.run(w, a=a, grid=g, t_max=60.0, cfl=0.02, tail_tol=1e-11,
                     w_max=lvl)
        w = out["w"]
        if out["reason"] != "amplitude cap":
            break
        peak = np.abs(w).max()
        j = int(np.argmax(np.abs(w)))
        # Centre the peak so the three snapshots are directly comparable.
        y = g.x - g.x[j]
        y = (y + np.pi) % (2 * np.pi) - np.pi
        order = np.argsort(y)
        ax.plot(y[order], (w / peak)[order], color=colour, lw=1.6,
                label=f"height {lvl:.0f}x")
    ax.set_xlim(-1.6, 1.6)
    ax.set_xlabel("position around the circle")
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

axes[0].set_ylabel("shape, with the height divided out")
fig.tight_layout()
fig.savefig("fig_two_regimes.png")
print("wrote fig_two_regimes.png")
