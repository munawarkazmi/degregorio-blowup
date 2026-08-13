"""
Test the exchange of stability hypothesis.

Huang, Tong and Wang report that their self similar profile loses stability as
a increases through a = 0.751. The frozen profile here gains selection as a
increases through roughly the same value. The proposed reading is that these
are two halves of one exchange: the narrowing attractor, which carries a
nonzero spatial scaling exponent c_l, gives way to the frozen one, which
requires c_l = 0 exactly.

That has a direct consequence which does not need their profile at all. If the
two branches exchange stability at some a_c, then the narrowing rate observed
in the dynamics must fall to zero there. Measure it.

For a self similar blowup, omega ~ (T-t)^-1 and the peak width scales like
(T-t)^c_l, so

    width ~ amplitude^(-c_l)

and c_l is minus the slope of log width against log amplitude. The width used
here is the half width at half maximum around the peak, measured in real space,
which is robust whether the spectrum is exponential or algebraic. The earlier
strip width diagnostic is not usable for this, since it fits a single
exponential and the profiles at large a have algebraic spectra.

Calibration: at a = 0 the exact CLM solution from sin x has
||omega||_inf = 4/(4-t^2) and its peak width proportional to (2-t), so c_l = 1
exactly. Any implementation that does not return 1 there is wrong.

    python exchange.py
"""

import numpy as np

import dg
from profile_eq import simulate

# A geometric ladder rather than fixed levels. Strongly narrowing cases exhaust
# resolution early and only reach the low rungs; frozen cases climb all of
# them. Fitting over the last few rungs reached therefore uses the highest
# amplitudes available in each case, and avoids the early transient before the
# solution has settled onto whatever it is converging to.
LEVELS = tuple(4.0 * 2.0 ** k for k in range(11))
NFIT = 4


def half_width(g, w):
    """
    Distance from the peak to where |w| first falls to half its peak value,
    walking outward and interpolating linearly across the crossing.
    """
    a = np.abs(w)
    i = int(np.argmax(a))
    peak = a[i]
    target = 0.5 * peak

    out = []
    for step in (1, -1):
        j = i
        for _ in range(g.N // 2):
            k = (j + step) % g.N
            if a[k] <= target:
                # interpolate between j and k
                f = (a[j] - target) / max(a[j] - a[k], 1e-300)
                out.append((abs(j - i) + f) * g.dx)
                break
            j = k
        else:
            return np.nan
    return float(min(out))


def scaling_exponent(a, n=4096, levels=LEVELS, verbose=False):
    """c_l from the slope of log(width) against log(amplitude)."""
    g = dg.Grid(n)
    w = np.sin(g.x)
    amps, widths, t0 = [], [], 0.0
    for lvl in levels:
        out = dg.run(w, a=a, grid=g, t_max=80.0, cfl=0.02, tail_tol=1e-9,
                     w_max=lvl)
        w, t0 = out["w"], t0 + out["t"]
        if out["reason"] != "amplitude cap":
            if verbose:
                print(f"      stopped at {out['reason']} "
                      f"with |w| = {out['winf_hist'][-1]:.3e}")
            break
        hw = half_width(g, w)
        if not np.isfinite(hw):
            break
        amps.append(out["winf_hist"][-1])
        widths.append(hw)
    if len(amps) < 3:
        return np.nan, len(amps), amps, widths
    fa, fw = amps[-NFIT:], widths[-NFIT:]
    slope = np.polyfit(np.log(fa), np.log(fw), 1)[0]
    return -float(slope), len(amps), fa, fw


print()
print("=" * 74)
print("Calibration at a = 0, where c_l = 1 exactly")
print("=" * 74)
cl0, n0, amps0, w0 = scaling_exponent(0.0, n=8192)
print(f"  measured c_l = {cl0:.5f} from {n0} amplitude levels "
      f"(exact value 1)")
for A, W in zip(amps0, w0):
    print(f"    |w| = {A:9.2f}   half width = {W:.6e}")

print()
print("=" * 74)
print("The narrowing rate across a")
print("=" * 74)
print("  c_l near 1 is CLM like narrowing. c_l = 0 is a frozen profile.")
print("  If the branches exchange stability at a_c, c_l must vanish there.")
print()
print(f"  {'a':>6}  {'c_l':>10}  {'levels':>7}  {'width at first':>15}  "
      f"{'width at last':>14}")
print("  " + "-" * 60)

rows = []
for a in (0.0, 0.2, 0.4, 0.5, 0.6, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82):
    cl, nl, amps, wid = scaling_exponent(a)
    if not np.isfinite(cl):
        print(f"  {a:6.2f}  {'too few levels':>10}  {nl:7d}")
        continue
    rows.append((a, cl))
    print(f"  {a:6.2f}  {cl:10.5f}  {nl:7d}  {wid[0]:15.6e}  "
          f"{wid[-1]:14.6e}")

if len(rows) >= 4:
    av = np.array([r[0] for r in rows])
    cv = np.array([r[1] for r in rows])
    near = (cv > 0.0) & (cv < 0.35) if ((cv > 0) & (cv < 0.35)).any() else None
    print()
    print("  c_l against a, to locate where it reaches zero:")
    if near is not None and near.sum() >= 2:
        m, b = np.polyfit(av[near], cv[near], 1)
        print(f"    linear fit over the small c_l points: "
              f"c_l = {m:.4f} a + {b:.4f}")
        print(f"    reaches zero at a = {-b / m:.4f}")
        print(f"    compare a_c2 = 0.751 from Huang, Tong and Wang")
    else:
        print("    not enough points with small positive c_l to extrapolate")
