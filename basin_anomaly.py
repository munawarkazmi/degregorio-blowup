"""
Why did sin x + 0.5 sin 2x not blow up?

basin.py reported "no blowup" for that datum, which conflicts with two things
it also reported. Part A found every datum with a sign change blowing up, and
generic.py found sin x + 0.3 sin 2x reaching the member 1 profile with
||f||_inf = 4.1484819 like everything else. So somewhere between an admixture
of 0.3 and 0.5 the behaviour changes.

The classification is also too coarse to tell what happened. It labels anything
that fails to reach the amplitude cap as "no blowup", which lumps together a
solution that stays bounded, a run that exhausted its time budget while still
growing, and a run that went under-resolved because it was narrowing. Those
have completely different meanings: the last would mean the datum left the
frozen regime rather than left the basin.

Print the stop reason, and sweep the admixture.

    python basin_anomaly.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, residual, sup

A = 0.8


def run_case(c, n=2048, t_max=400.0, w_max=1e4):
    g = dg.Grid(n)
    w0 = np.sin(g.x) + c * np.sin(2.0 * g.x)
    w0 = w0 / np.abs(w0).max()
    out = dg.run(w0, a=A, grid=g, t_max=t_max, cfl=0.02, tail_tol=1e-9,
                 w_max=w_max)
    return g, out


print()
print("=" * 74)
print("Sweeping the admixture in omega_0 = sin x + c sin 2x")
print("=" * 74)
print("  t_max raised to 400 so that a slow blowup is not mistaken for none.")
print()
print(f"  {'c':>6}  {'max |w|':>12}  {'t':>9}  {'stop reason':>16}  "
      f"{'||f||_inf':>11}")
print("  " + "-" * 62)

for c in (0.0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0):
    g, out = run_case(c)
    amp = float(out["winf_hist"][-1])
    tag = "-"
    if out["reason"] == "amplitude cap":
        f0, _ = guess_from_simulation(g, out)
        if f0 is not None:
            f, hist, ok = newton(g, f0, A)
            if ok and sup(g, f) > 1e-6:
                tag = f"{sup(g, f):.6f}"
    print(f"  {c:6.2f}  {amp:12.4e}  {out['t']:9.2f}  {out['reason']:>16}  "
          f"{tag:>11}")

print()
print("  A stop at t_max with a small amplitude means genuinely bounded.")
print("  A stop at t_max with a large or growing amplitude means the budget")
print("  ran out. Under-resolved would mean narrowing, hence a different")
print("  regime rather than a different basin.")

print()
print("=" * 74)
print("The borderline case in detail")
print("=" * 74)
for c in (0.4, 0.5):
    g, out = run_case(c, t_max=400.0)
    t, w = out["t_hist"], out["winf_hist"]
    print()
    print(f"  c = {c}: stop {out['reason']} at t = {out['t']:.2f}")
    print(f"    {'t':>8}  {'|w|':>12}")
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        i = int(frac * (len(t) - 1))
        print(f"    {t[i]:8.2f}  {w[i]:12.5e}")
    growing = w[-1] > 1.5 * w[int(0.75 * (len(w) - 1))]
    print(f"    still growing at the end: {growing}")
