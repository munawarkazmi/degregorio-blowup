"""
How does T diverge as a approaches 1?

At a = 1 every multiple A sin x is a steady state, since
R_1(A sin) = A^2 (sin)(-cos) - A^2 (-sin)(cos) = 0. So the critical case has a
line of equilibria, and it is destroyed for a < 1. With eps = 1 - a,

    R_a(omega) = R_1(omega) + eps u omega_x,

and at omega = A sin x, where u = -A sin x and omega_x = A cos x, this is

    R_a(A sin x) = -(eps A^2 / 2) sin 2x.

Mode 2 is therefore generated at rate eps, takes time of order 1/eps to become
comparable to mode 1, and blowup follows in order one more. The prediction is
T ~ C / eps. The values already in hand disagree: T eps reads 1.214, 1.076 and
0.886 at a = 0.8, 0.9, 0.95, drifting down rather than settling.

Those values came from fitting the reciprocal amplitude, and at a = 0.95 that
fit reported a rate exponent of 0.48 rather than 1, so it is not trustworthy.
There is a far better estimator. For a frozen profile ||w|| = ||f|| / (T - t)
exactly, so

    T = t + ||f|| / ||w||

at any late time, with no fitting at all. Running to ||w|| = 10^6 makes the
correction term of order 10^-6, so T is essentially the stopping time.

    python critical.py
"""

import numpy as np

import dg

WMAX = 1.0e6
AMP = 4.15          # ||f||_inf, near enough for a 1e-6 correction


def blowup_time(a, n=2048, t_max=600.0):
    g = dg.Grid(n)
    out = dg.run(np.sin(g.x), a=a, grid=g, t_max=t_max, cfl=0.02,
                 tail_tol=1e-9, w_max=WMAX)
    if out["reason"] != "amplitude cap":
        return np.nan, out
    T = out["t"] + AMP / float(out["winf_hist"][-1])
    return T, out


print()
print(f"  {'a':>7}  {'eps':>8}  {'T':>12}  {'T eps':>9}  {'tail':>9}  "
      f"{'steps':>8}")
print("  " + "-" * 60)

rows = []
for a in (0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99):
    T, out = blowup_time(a)
    if not np.isfinite(T):
        print(f"  {a:7.3f}  {1 - a:8.3f}  {'no blowup':>12}  "
              f"({out['reason']}, t = {out['t']:.1f})")
        continue
    eps = 1.0 - a
    rows.append((a, eps, T))
    print(f"  {a:7.3f}  {eps:8.3f}  {T:12.6f}  {T * eps:9.5f}  "
          f"{out['tail_hist'][-1]:9.2e}  {out['steps']:8d}")

if len(rows) >= 4:
    eps = np.array([r[1] for r in rows])
    T = np.array([r[2] for r in rows])

    print()
    print("  Local exponent of T ~ eps^-p between consecutive points:")
    p = -np.diff(np.log(T)) / np.diff(np.log(eps))
    for i in range(len(p)):
        print(f"    eps {eps[i]:.3f} to {eps[i + 1]:.3f}:  p = {p[i]:.4f}")

    print()
    print("  Testing three shapes over the last points:")
    sel = eps <= 0.1
    if sel.sum() >= 3:
        e, t = eps[sel], T[sel]
        m, c = np.polyfit(np.log(e), np.log(t), 1)
        r = t - np.exp(c) * e ** m
        print(f"    T = C eps^-p         : p = {-m:.4f}, "
              f"rms {np.sqrt((r ** 2).mean()):.3e}")

        m2, c2 = np.polyfit(1.0 / e, t, 1)
        r2 = t - (m2 / e + c2)
        print(f"    T = C/eps + D        : C = {m2:.4f}, D = {c2:.4f}, "
              f"rms {np.sqrt((r2 ** 2).mean()):.3e}")

        M = np.column_stack([np.log(1.0 / e), np.ones_like(e)])
        coef, *_ = np.linalg.lstsq(M, t, rcond=None)
        r3 = t - M @ coef
        print(f"    T = C log(1/eps) + D : C = {coef[0]:.4f}, "
              f"D = {coef[1]:.4f}, rms {np.sqrt((r3 ** 2).mean()):.3e}")

        M4 = np.column_stack([1.0 / e, np.log(1.0 / e), np.ones_like(e)])
        coef4, *_ = np.linalg.lstsq(M4, t, rcond=None)
        r4 = t - M4 @ coef4
        print(f"    T = A/eps + B log(1/eps) + C : A = {coef4[0]:.4f}, "
              f"B = {coef4[1]:.4f}, rms {np.sqrt((r4 ** 2).mean()):.3e}")

    print()
    print("  The prediction from the line of equilibria is p = 1. A local")
    print("  exponent drifting rather than settling means the asymptotic")
    print("  regime has not been reached, whatever the fits report.")
