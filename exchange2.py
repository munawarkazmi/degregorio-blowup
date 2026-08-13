"""
Locate the branch merger precisely enough to compare with a = 0.751.

The narrowing branch cannot be posed on the circle: its profile equation
carries a term c_l X, which is not periodic, so c_l is forced to zero and only
the frozen branch exists there. That is not an obstruction to the comparison,
it is the reason the comparison works. A concentrating solution on the circle
stops feeling the domain once its width is small against 2 pi, so the c_l it
exhibits converges to the real line value. Measuring c_l from circle dynamics
therefore estimates the same quantity Huang, Tong and Wang compute directly.

The previous attempt put c_l = 0 somewhere in 0.78 to 0.82 and could not do
better. The limitation was precision, not principle: it fitted the slope over
the last four rungs of a ladder in steps of two, which is 0.9 decades of
amplitude, and with c_l of order 1e-3 the width changes by 1 part in 700 across
that span. Fitting over three decades instead should gain more than an order of
magnitude.

Two changes. Push each run to amplitude 1e6, which is cheap in this range since
these profiles barely narrow and stay resolved on a coarse grid. Fit over every
rung above amplitude 100, discarding the early transient before the solution
settles.

    python exchange2.py
"""

import numpy as np

import dg

LADDER = tuple(4.0 * 2.0 ** k for k in range(19))     # 4 up to about 1e6
FIT_ABOVE = 100.0


def half_width(g, w):
    a = np.abs(w)
    i = int(np.argmax(a))
    target = 0.5 * a[i]
    out = []
    for step in (1, -1):
        j = i
        for _ in range(g.N // 2):
            k = (j + step) % g.N
            if a[k] <= target:
                f = (a[j] - target) / max(a[j] - a[k], 1e-300)
                out.append((abs(j - i) + f) * g.dx)
                break
            j = k
        else:
            return np.nan
    return float(min(out))


def measure(a, n, ladder=LADDER, fit_above=FIT_ABOVE):
    g = dg.Grid(n)
    w = np.sin(g.x)
    amps, widths = [], []
    for lvl in ladder:
        out = dg.run(w, a=a, grid=g, t_max=120.0, cfl=0.02, tail_tol=1e-9,
                     w_max=lvl)
        w = out["w"]
        if out["reason"] != "amplitude cap":
            break
        hw = half_width(g, w)
        if not np.isfinite(hw):
            break
        amps.append(float(out["winf_hist"][-1]))
        widths.append(hw)
    amps, widths = np.array(amps), np.array(widths)
    sel = amps >= fit_above
    if sel.sum() < 4:
        sel = np.ones_like(amps, dtype=bool)
    if len(amps) < 4:
        return np.nan, np.nan, len(amps), amps
    x, y = np.log(amps[sel]), np.log(widths[sel])
    m, c = np.polyfit(x, y, 1)
    resid = y - (m * x + c)
    # Standard error of the slope, as a guide to whether c_l differs from zero.
    se = float(np.sqrt((resid ** 2).sum() / max(len(x) - 2, 1)
                       / ((x - x.mean()) ** 2).sum()))
    return -float(m), se, int(sel.sum()), amps


print()
print("=" * 72)
print("Calibration at a = 0, exact c_l = 1")
print("=" * 72)
cl, se, npts, amps = measure(0.0, 8192, ladder=LADDER[:5], fit_above=0.0)
print(f"  c_l = {cl:.5f} +/- {se:.5f} from {npts} points, "
      f"amplitudes up to {amps[-1]:.1f}")

print()
print("=" * 72)
print("c_l near the merger, fitted over three decades of amplitude")
print("=" * 72)
print()
print(f"  {'a':>6}  {'c_l':>12}  {'std err':>10}  {'pts':>4}  "
      f"{'max amplitude':>14}")
print("  " + "-" * 54)

rows = []
for a in (0.70, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.82, 0.84):
    cl, se, npts, amps = measure(a, 2048)
    if not np.isfinite(cl):
        print(f"  {a:6.2f}  {'failed':>12}")
        continue
    rows.append((a, cl, se))
    print(f"  {a:6.2f}  {cl:12.6f}  {se:10.2e}  {npts:4d}  "
          f"{amps[-1]:14.3e}")

if len(rows) >= 5:
    av = np.array([r[0] for r in rows])
    cv = np.array([r[1] for r in rows])
    print()
    print("  Locating the root. c_l is convex in a, so a straight line through")
    print("  it underestimates the root; that is the trap from earlier. Fit")
    print("  c_l = A (a_c - a)^p instead, scanning p and taking the best a_c.")
    # Fit only where c_l is many standard errors from zero, and scan a_c from
    # just above the largest such a. Scanning from av.max() upward is wrong:
    # a_c lies below the largest a sampled, so the search pins to its own
    # boundary and returns that boundary for every p.
    sev = np.array([r[2] for r in rows])
    use = cv > 4.0 * sev
    lo = av[use].max() + 0.005
    best = None
    for p in np.linspace(1.0, 6.0, 501):
        for ac in np.linspace(lo, lo + 0.20, 801):
            pred = (ac - av[use]) ** p
            A = float((pred * cv[use]).sum() / (pred * pred).sum())
            ss = float(((cv[use] - A * pred) ** 2).sum())
            if best is None or ss < best[0]:
                best = (ss, p, ac, A)
    ss, p, ac, A = best
    print(f"    best fit p = {p:.2f}, a_c = {ac:.4f}, "
          f"rms residual = {np.sqrt(ss / use.sum()):.2e}")
    print()
    print("    sensitivity, best a_c at each fixed p:")
    for pf in (1.5, 2.0, 2.5, 3.0, 3.5):
        b2 = None
        for ac2 in np.linspace(av.max(), av.max() + 0.15, 601):
            pred = (ac2 - av[use]) ** pf
            A2 = float((pred * cv[use]).sum() / (pred * pred).sum())
            s2 = float(((cv[use] - A2 * pred) ** 2).sum())
            if b2 is None or s2 < b2[0]:
                b2 = (s2, ac2)
        print(f"      p = {pf:.1f}:  a_c = {b2[1]:.4f}   "
              f"rms = {np.sqrt(b2[0] / use.sum()):.2e}")
    print()
    print("    compare a_c2 = 0.751 from Huang, Tong and Wang")
