"""
What sets c = H f(y*)?

beta = 1 + (c - 1) / (a c) predicts the spectral decay from one local number,
but c itself is global and so far only measured. Rearranged, the relation is

    1 / c = 1 - a mu

so asking what sets c is the same as asking what sets mu.

There is a clue in the a = 0.8 numbers. The two stagnation points gave
1/c1 = 0.19993 and 1/c2 = -0.60133, against 1 - a = 0.2 and 1 - 2a = -0.6.
That is mu1 = 1 and mu2 = 2. The second is coincidence, since mu2 is 3.21 at
a = 0.75 and 1.52 at a = 0.85 and both were confirmed against measured spectra.
The first might not be. If mu1 = 1 for every a, then one stagnation point is
always a simple linear zero of f and only the other carries the singularity,
which is a structural fact rather than a fit.

Three things here.

  1. Is mu1 = 1 exactly, for every a?
  2. Map mu2(a) finely. It appears to diverge as a decreases toward roughly
     0.69, which would mean the profile becomes analytic there, and to fall
     toward 1 as a rises, which would mean it becomes too rough to be C^1.
  3. Fix a measurement bias. The beating in |f_k| was attributed to two
     singular points pi apart, but if the even modes are simply far smaller
     than the odd ones, then binning by RMS averages signal together with
     near-zeros and drags the measured beta down. That would explain why every
     measured beta came out about 1 percent below prediction, systematically
     rather than randomly.

    python c_of_a.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup


def refine_zero(g, U, i):
    x0, x1 = g.x[i], g.x[(i + 1) % g.N]
    if x1 < x0:
        x1 += 2.0 * np.pi
    u0, u1 = U[i], U[(i + 1) % g.N]
    return x0 + (x1 - x0) * u0 / (u0 - u1)


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def stagnation(g, f, a):
    U, Hf, _ = dg.fields(g, f)
    out = []
    for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
        ys = refine_zero(g, U, i) % (2.0 * np.pi)
        c = evaluate(g, Hf, ys)
        out.append((ys, c, (c - 1.0) / (a * c)))
    return sorted(out, key=lambda r: r[2])


def solve(a, n=4096):
    g, out = simulate(a, n=n, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, a)
    return (g, f) if ok and sup(g, f) > 1e-6 else (g, None)


AVALS = (0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88)

print()
print("=" * 78)
print("Part 1: is mu1 = 1 exactly? Equivalently, is 1/c1 = 1 - a?")
print("=" * 78)
print()
print(f"  {'a':>6}  {'1/c1':>11}  {'1 - a':>9}  {'mu1':>11}  {'mu1 - 1':>10}  "
      f"{'mu2':>11}  {'1/mu2':>9}")
print("  " + "-" * 76)

rows = []
for a in AVALS:
    g, f = solve(a)
    if f is None:
        print(f"  {a:6.2f}  Newton failed or trivial")
        continue
    r = stagnation(g, f, a)
    if len(r) < 2:
        print(f"  {a:6.2f}  {len(r)} stagnation point(s) only")
        continue
    # mu1 is the smaller exponent, mu2 the larger, independent of ordering
    # by position on the circle.
    mu1, mu2 = r[0][2], r[1][2]
    c1 = r[0][1]
    rows.append((a, mu1, mu2))
    print(f"  {a:6.2f}  {1.0 / c1:11.7f}  {1.0 - a:9.5f}  {mu1:11.7f}  "
          f"{mu1 - 1.0:10.2e}  {mu2:11.6f}  {1.0 / mu2:9.6f}")

if rows:
    dev = max(abs(m1 - 1.0) for _, m1, _ in rows)
    print()
    print(f"  largest |mu1 - 1| over the range: {dev:.3e}")
    print("  If that stays at the level of the profile's own accuracy, about")
    print("  1e-3 here, then mu1 = 1 is exact and 1/c1 = 1 - a is a genuine")
    print("  identity rather than a fit.")


print()
print("=" * 78)
print("Part 2: where does mu2 diverge, and where does it reach 1?")
print("=" * 78)
print("  1/mu2 looked close to linear in a. If it is, the zero crossing gives")
print("  the value of a where the profile turns analytic, and 1/mu2 = 1 gives")
print("  where it stops being C^1.")

if len(rows) >= 4:
    av = np.array([r[0] for r in rows])
    inv = np.array([1.0 / r[2] for r in rows])
    m, c0 = np.polyfit(av, inv, 1)
    resid = inv - (m * av + c0)
    ss = ((inv - inv.mean()) ** 2).sum()
    print()
    print(f"  linear fit 1/mu2 = {m:.5f} a + {c0:.5f},  "
          f"R^2 = {1.0 - (resid ** 2).sum() / ss:.6f}")
    print(f"  1/mu2 = 0 at a = {-c0 / m:.5f}   (profile becomes analytic)")
    print(f"  1/mu2 = 1 at a = {(1.0 - c0) / m:.5f}   (profile leaves C^1)")
    print()
    print("  Residuals, to show whether linear is the right shape at all:")
    for a, r_ in zip(av, resid):
        print(f"    a = {a:.2f}   residual = {r_:+.5f}")


print()
print("=" * 78)
print("Part 3: even modes against odd modes, and the measurement bias")
print("=" * 78)

g, f = solve(0.80)
amp = np.abs(g.fwd(f))
print()
print("  |f_k| for the first modes:")
for k in range(1, 13):
    print(f"    k = {k:3d}   {amp[k]:.6e}")

hi = (g.k > 100) & (g.k < g.kcut)
ev = hi & (g.k % 2 == 0)
od = hi & (g.k % 2 == 1)
print()
print(f"  over 100 < k < k_cut:")
print(f"    rms |f_k| on even k = {np.sqrt((amp[ev] ** 2).mean()):.6e}")
print(f"    rms |f_k| on odd  k = {np.sqrt((amp[od] ** 2).mean()):.6e}")
print(f"    ratio even / odd    = "
      f"{np.sqrt((amp[ev] ** 2).mean()) / np.sqrt((amp[od] ** 2).mean()):.6e}")


def beta_from(mask_parity):
    k = g.k
    floor = amp.max() * 1e-13
    kc, vc = [], []
    lo = 8.0
    while lo * np.sqrt(2.0) <= g.kcut:
        hi_ = lo * np.sqrt(2.0)
        sel = (k >= lo) & (k < hi_) & (amp > floor)
        if mask_parity is not None:
            sel &= (k % 2 == mask_parity)
        if sel.sum() >= 3:
            kc.append(np.sqrt(lo * hi_))
            vc.append(np.sqrt((amp[sel] ** 2).mean()))
        lo = hi_
    kc, vc = np.array(kc), np.array(vc)
    use = kc > 12
    x, y = np.log(kc[use]), np.log(vc[use])
    m_, c_ = np.polyfit(x, y, 1)
    r_ = y - (m_ * x + c_)
    ss_ = ((y - y.mean()) ** 2).sum()
    return -m_, 1.0 - (r_ ** 2).sum() / ss_


r = stagnation(g, f, 0.80)
print()
print(f"  predicted beta at a = 0.8: {r[1][2] + 1:.6f}")
for label, parity in (("all modes", None), ("odd k only", 1),
                      ("even k only", 0)):
    b, r2 = beta_from(parity)
    print(f"  measured, {label:12s}: beta = {b:.6f}   R^2 = {r2:.6f}")
print()
print("  If the odd only figure lands closer to the prediction, the previous")
print("  1 percent shortfall was the even modes being mixed into the bins.")
