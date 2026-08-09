"""
Generate the figures used in README.md.

    python figures.py

Writes fig_blowup.png, fig_frozen.png, fig_profile.png, fig_spectrum.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dg
from profile_eq import guess_from_simulation, jacobian, newton, simulate, sup

plt.rcParams.update({
    "figure.dpi": 140,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE, RED, GREY = "#2b6cb0", "#c53030", "#718096"


# ---------------------------------------------------------------------------
def fig_blowup():
    """Reciprocal amplitude for a range of a, and the resulting T(a)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    avals = [0.0, 0.2, 0.4, 0.6, 0.8]
    cmap = plt.get_cmap("viridis")
    Ts = []
    for a in avals:
        g = dg.Grid(2048)
        out = dg.run(np.sin(g.x), a=a, grid=g, t_max=25.0, cfl=0.02,
                     tail_tol=1e-9, w_max=1e6)
        t, w = out["t_hist"], out["winf_hist"]
        ax1.plot(t, w[0] / w, color=cmap(a / 1.05), lw=1.6,
                 label=f"a = {a:.1f}")
        T, _, _ = dg.fit_blowup(t, w, window_decades=0.5)
        Ts.append(T)

    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$\|w_0\|_\infty\,/\,\|w(t)\|_\infty$")
    ax1.set_title("Reciprocal amplitude\na straight line reaching zero is "
                  "blowup at rate 1", fontsize=9)
    ax1.set_ylim(0, 1.03)
    ax1.legend(fontsize=7.5, frameon=False)

    sweep_a = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    sweep_T = [2.0, 2.3101, 2.7858, 3.1416, 3.6500, 4.4641, 6.0719,
               10.7569, 17.7129]
    ax2.plot(sweep_a, sweep_T, "o-", color=BLUE, ms=4, lw=1.4)
    ax2.scatter([0.0], [2.0], color=RED, zorder=5, s=36)
    ax2.annotate("exact: T(0) = 2", (0.0, 2.0), xytext=(14, -2),
                 textcoords="offset points", fontsize=8, color=RED)
    ax2.scatter([0.5], [np.pi], color=RED, zorder=5, s=36)
    ax2.annotate(r"T(1/2) = $\pi$ to 1e-11", (0.5, np.pi), xytext=(10, -14),
                 textcoords="offset points", fontsize=8, color=RED)
    ax2.axvline(1.0, ls=":", color=GREY, lw=1)
    ax2.annotate("a = 1: sin x is steady,\nso T is infinite", (1.0, 6),
                 xytext=(-96, 12), textcoords="offset points", fontsize=8,
                 color=GREY)
    ax2.set_xlabel("a")
    ax2.set_ylabel("blowup time T")
    ax2.set_yscale("log")
    ax2.set_xlim(-0.05, 1.12)
    ax2.set_title(r"T(a) for $w_0 = \sin x$" "\nboth endpoints known exactly",
                  fontsize=9)

    fig.tight_layout()
    fig.savefig("fig_blowup.png")
    plt.close(fig)
    print("wrote fig_blowup.png")


# ---------------------------------------------------------------------------
def fig_frozen():
    """Normalised spectra at successive decades: frozen versus narrowing."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

    for ax, a, title in ((axes[0], 0.4, "a = 0.4: the peak narrows"),
                         (axes[1], 0.8, "a = 0.8: the profile freezes")):
        g = dg.Grid(4096)
        w = np.sin(g.x)
        cmap = plt.get_cmap("plasma")
        for i, lvl in enumerate((1e2, 1e3, 1e4, 1e5, 1e6)):
            out = dg.run(w, a=a, grid=g, t_max=60.0, cfl=0.02,
                         tail_tol=1e-11, w_max=lvl)
            w = out["w"]
            if out["reason"] != "amplitude cap":
                break
            amp = np.abs(g.fwd(w))
            ax.semilogy(g.k, amp / amp.max(), lw=1.2, color=cmap(i / 5),
                        label=r"$\|w\|_\infty = 10^{%d}$" % (i + 2))
        ax.set_xlim(0, 900)
        ax.set_ylim(1e-17, 3)
        ax.set_xlabel("wavenumber k")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    axes[0].set_ylabel(r"$|\hat w_k| \,/\, \max_k |\hat w_k|$")
    fig.suptitle("Normalised spectra as the amplitude grows by decades. "
                 "Curves that lie on top of\none another mean the shape is "
                 "fixed and only the height is diverging.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig("fig_frozen.png")
    plt.close(fig)
    print("wrote fig_frozen.png")


# ---------------------------------------------------------------------------
def solved_profile(a=0.8, n=1024):
    g, out = simulate(a, n=n)
    f0, amp_sim = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, a)
    return g, f, amp_sim, hist


def fig_profile():
    """The solved profile itself, and how well it satisfies the equation."""
    g, f, amp_sim, res_hist = solved_profile()
    y = g.x - np.pi
    fs = np.roll(f, g.N // 2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax1.plot(y, fs, color=BLUE, lw=1.7)
    ax1.set_xlabel("y")
    ax1.set_ylabel("f(y)")
    ax1.set_xlim(-np.pi, np.pi)
    ax1.set_title("Blowup profile at a = 0.8\n"
                  r"$w(x,t) \to f(x - x_p)\,/\,(T-t)$", fontsize=9)
    ax1.annotate(rf"$\|f\|_\infty = {sup(g, f):.5f}$" "\n"
                 rf"$\max|f'| = {np.abs(g.bwd(g.fwd(f) * g.dmul)).max():.2f}$"
                 "\nbroad and gentle, exactly odd",
                 (0.03, 0.05), xycoords="axes fraction", fontsize=8)

    ax2.semilogy(range(len(res_hist)), res_hist, "o-", color=RED, ms=5, lw=1.4)
    ax2.set_xlabel("Newton iteration")
    ax2.set_ylabel(r"$\max |\,\mathrm{rhs}(f) - f\,|$")
    ax2.set_xticks(range(len(res_hist)))
    ax2.set_title("Quadratic convergence to the fixed point\n"
                  "iteration 0 is the raw simulated profile", fontsize=9)

    amp = np.abs(g.fwd(f))
    k = g.k
    sel = (k > 0) & (amp > amp.max() * 1e-13)
    ax3.loglog(k[sel], amp[sel] / amp.max(), color=BLUE, lw=1.5,
               label="profile spectrum")
    kk = k[sel][k[sel] > 8].astype(float)
    ref = (kk / 8.0) ** -3.05 * float(amp[8] / amp.max())
    ax3.loglog(kk, ref, ls="--", color=RED, lw=1.3,
               label=r"$k^{-3.05}$")
    ax3.set_xlabel("wavenumber k")
    ax3.set_ylabel(r"$|\hat f_k| / \max_k |\hat f_k|$")
    ax3.set_title("The profile is not analytic\n"
                  "algebraic decay means roughly $C^2$, no better", fontsize=9)
    ax3.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig("fig_profile.png")
    plt.close(fig)
    print("wrote fig_profile.png")


# ---------------------------------------------------------------------------
def fig_spectrum():
    """Jacobian eigenvalues in the complex plane."""
    g, f, _, ok = solved_profile(n=512)
    vals = np.linalg.eigvals(jacobian(g, f, 0.8))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    forced = (np.abs(vals - 1.0) < 1e-6) | (np.abs(vals) < 1e-6)
    ax.scatter(vals[~forced].real, vals[~forced].imag, s=9, color=GREY,
               alpha=0.6, label="rest of the spectrum")
    ax.scatter(vals[forced].real, vals[forced].imag, s=55, color=RED,
               zorder=5, label="forced by symmetry")
    ax.axvline(0, color="k", lw=0.9)
    ax.axhline(0, color="k", lw=0.4, alpha=0.4)

    ax.annotate(r"$\lambda = 1$, eigenvector $f$" "\nshift of the blowup time",
                (1.0, 0.0), xytext=(-8, 34), textcoords="offset points",
                fontsize=8, color=RED, ha="right")
    ax.annotate(r"$\lambda = 0$, eigenvector $f'$" "\ntranslation",
                (0.0, 0.0), xytext=(-118, -40), textcoords="offset points",
                fontsize=8, color=RED)
    ax.set_xlabel(r"Re $\lambda$   (growth rate in renormalised time $s$)")
    ax.set_ylabel(r"Im $\lambda$")
    ax.set_xlim(-3.2, 1.6)
    ax.set_ylim(-6, 6)
    hidden = int((np.abs(vals.imag) > 6).sum())
    ax.annotate(f"{hidden} further modes lie off the top and bottom of this\n"
                r"view, strung along Re $\lambda \approx -1$ out to "
                rf"Im $\lambda = \pm{np.abs(vals.imag).max():.0f}$."
                "\nThey are the high wavenumber modes, all stable.",
                (0.02, 0.03), xycoords="axes fraction", fontsize=7.5,
                color=GREY)
    ax.set_title("Jacobian spectrum at the a = 0.8 profile\n"
                 "nothing in the right half plane except the two forced "
                 "modes,\nso the profile is linearly stable", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("fig_spectrum.png")
    plt.close(fig)
    print("wrote fig_spectrum.png")


if __name__ == "__main__":
    fig_blowup()
    fig_frozen()
    fig_profile()
    fig_spectrum()
