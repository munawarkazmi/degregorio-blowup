"""
Why did the a = 0.6 and a = 0.8 runs reach amplitudes of 1e9 and 1e12 on a
2048 point grid without the resolution guard firing?

Two candidate explanations:

  1. Real, but with a spectrum that decays algebraically rather than
     exponentially. The guard measures the fraction of energy in the top
     quarter of the band, which for a power law spectrum k^-b is a constant
     independent of how singular the solution is. The guard would then never
     fire no matter how badly resolved things get.

  2. Numerical instability, which would show as grid scale oscillation in the
     field and as energy piling up at the very last retained mode.

The arbiter is grid refinement, not any proxy: run the same case at several N
and see where the amplitude histories part company.

    python diagnose.py
"""

import numpy as np

import dg


def spectrum_slope(g, w):
    """Least squares slope of log|w_k| against log k over the upper decade."""
    wh = np.abs(g.fwd(w))
    k = g.k
    lo = max(2, g.kcut // 10)
    sel = slice(lo, g.kcut + 1)
    amp = wh[sel]
    kk = k[sel].astype(float)
    good = amp > 0
    if good.sum() < 10:
        return np.nan
    return float(np.polyfit(np.log(kk[good]), np.log(amp[good]), 1)[0])


for a in (0.0, 0.6, 0.8):
    print()
    print("=" * 76)
    print(f"a = {a}")
    print("=" * 76)

    hist = {}
    for N in (1024, 2048, 4096, 8192):
        g = dg.Grid(N)
        out = dg.run(np.sin(g.x), a=a, grid=g, t_max=10.0, cfl=0.02,
                     tail_tol=1e-9, w_max=1e12)
        hist[N] = out
        w = out["w"]

        # Grid scale oscillation detector: energy in the top 5% of the band
        # relative to the peak mode. Instability parks itself there.
        wh = np.abs(g.fwd(w))
        top = wh[int(0.95 * g.kcut):g.kcut + 1].max() / max(wh.max(), 1e-300)

        print(f"  N = {N:5d}  t_end = {out['t']:9.5f}  "
              f"||w||_inf = {out['winf_hist'][-1]:12.4e}  "
              f"stop = {out['reason']:>15}")
        print(f"            spectral slope = {spectrum_slope(g, w):7.3f}   "
              f"top-mode / peak-mode = {top:.3e}   "
              f"tail = {out['tail_hist'][-1]:.2e}")

    # Where do the amplitude histories separate? Compare each N against the
    # finest run on a shared time axis.
    ref = hist[8192]
    print()
    print("  agreement with N = 8192, as a function of time:")
    print(f"  {'t':>8}  " + "  ".join(f"{'N=' + str(N):>12}"
                                      for N in (1024, 2048, 4096)))
    for frac in (0.25, 0.5, 0.75, 0.9, 0.99):
        t_q = ref["t_hist"][int(frac * (len(ref["t_hist"]) - 1))]
        w_ref = np.interp(t_q, ref["t_hist"], ref["winf_hist"])
        cells = []
        for N in (1024, 2048, 4096):
            h = hist[N]
            if t_q > h["t_hist"][-1]:
                cells.append(f"{'past end':>12}")
            else:
                w_n = np.interp(t_q, h["t_hist"], h["winf_hist"])
                cells.append(f"{abs(w_n - w_ref) / w_ref:12.2e}")
        print(f"  {t_q:8.4f}  " + "  ".join(cells) + f"   (||w||={w_ref:.3e})")
