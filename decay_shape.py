"""
Is the decaying solution self-similar at all?

The prediction that it converges to -h/t with h a profile is refuted:
h = -t w carries an 8 percent residual in R(h) = h at t = 400, and Newton
started from it walks to member 1 rather than to anything new.

Two readings remain. The decay is self-similar but has not converged by
t = 400, in which case the shape should still be settling and ||w||_inf t
should still be drifting toward some limit. Or it is not self-similar, in which
case the shape keeps changing and the 1/t reading is a coincidence of the
window sampled.

The test is the one used for the blowup regime, and it needs no ansatz:
compare normalised spectra at successive times. Curves lying on top of one
another mean a fixed shape. Run to t = 2000 rather than 400 so that any slow
convergence has room to show.

    python decay_shape.py
"""

import numpy as np

import dg
from profile_eq import residual, sup

A = 0.8
C = 0.50
TIMES = (100.0, 200.0, 400.0, 800.0, 1400.0, 2000.0)


def normalised(g, w):
    amp = np.abs(g.fwd(w))
    return amp / amp.max()


g = dg.Grid(2048)
w = np.sin(g.x) + C * np.sin(2.0 * g.x)
w = w / np.abs(w).max()

print()
print(f"  {'t':>8}  {'||w||_inf':>13}  {'||w|| t':>10}  {'decay exp':>10}  "
      f"{'resid of -tw':>13}  {'stop':>10}")
print("  " + "-" * 70)

snaps, t0, prev = [], 0.0, None
for target in TIMES:
    out = dg.run(w, a=A, grid=g, t_max=target - t0, cfl=0.02, tail_tol=1e-9,
                 w_max=1e4)
    w, t0 = out["w"], t0 + out["t"]
    amp = float(np.abs(w).max())
    ex = "-"
    if prev is not None:
        ex = f"{-(np.log(amp) - np.log(prev[1])) / (np.log(t0) - np.log(prev[0])):10.4f}"
    r = float(np.abs(residual(g, -w * t0, A)).max())
    print(f"  {t0:8.1f}  {amp:13.6e}  {amp * t0:10.5f}  {ex:>10}  "
          f"{r:13.3e}  {out['reason']:>10}")
    snaps.append((t0, normalised(g, w)))
    prev = (t0, amp)
    if out["reason"] not in ("t_max",):
        print(f"    stopped early: {out['reason']}")
        break

print()
print("  normalised spectra against the first snapshot:")
print(f"  {'t':>8}  {'max deviation':>15}")
print("  " + "-" * 26)
ref = snaps[0][1]
for t, s in snaps[1:]:
    n = min(len(ref), len(s))
    print(f"  {t:8.1f}  {float(np.abs(s[:n] - ref[:n]).max()):15.3e}")

print()
print("  Deviations converging to a constant mean the shape freezes and the")
print("  decay is self-similar. Deviations that keep growing mean it is not,")
print("  and the 1/t reading was a property of the window sampled.")
