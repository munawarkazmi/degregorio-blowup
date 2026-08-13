"""
Redo the decay bracket properly.

basin_decay.py ran at N = 1024 with t_max = 300 and every decaying case
stopped near t = 17.5 rather than at the time limit, so those runs ended for
some reason it never printed and the label rested on the amplitude at that
moment rather than on an asymptotic outcome. The one case that did run to
completion, c = 0.5 at N = 2048 to t = 400, decayed to 7.4e-3 with |w| t
settling near 2.97.

Repeat the bracket at that standard, printing the stop reason, and report
|w| t at the end so that a genuine 1/t decay is distinguishable from a
transient dip.

    python basin_decay2.py
"""

import numpy as np

import dg

A = 0.8


def run(c, n=2048, t_max=400.0, w_max=1e4):
    g = dg.Grid(n)
    w0 = np.sin(g.x) + c * np.sin(2.0 * g.x)
    w0 = w0 / np.abs(w0).max()
    return g, dg.run(w0, a=A, grid=g, t_max=t_max, cfl=0.02, tail_tol=1e-9,
                     w_max=w_max)


print()
print(f"  {'c':>6}  {'stop':>16}  {'t':>8}  {'final |w|':>12}  "
      f"{'|w| t':>9}  {'verdict':>9}")
print("  " + "-" * 66)

for c in (0.46, 0.47, 0.475, 0.48, 0.50, 0.52, 0.56, 0.57, 0.575, 0.58):
    g, out = run(c)
    amp = float(out["winf_hist"][-1])
    t = out["t"]
    if out["reason"] == "amplitude cap":
        verdict = "blowup"
        prod = "-"
    elif out["reason"] == "t_max" and amp < 0.05:
        verdict = "decay"
        prod = f"{amp * t:9.3f}"
    else:
        verdict = "unclear"
        prod = f"{amp * t:9.3f}"
    print(f"  {c:6.3f}  {out['reason']:>16}  {t:8.2f}  {amp:12.4e}  "
          f"{prod:>9}  {verdict:>9}")

print()
print("  A block of decaying c between blowing up ones on both sides means")
print("  a third outcome with an open basin, not a separatrix.")
