"""
Follow-up to profile_solve.py, fixing the wrong question.

Part 1 there worked: at a = 0.8 the simulated profile satisfies rhs(f) = f to
1.3e-5 before Newton touches it, Newton converges quadratically, and the peak
height agrees with the simulation's reciprocal slope to 4e-6.

Parts 2 and 3 did not, and the giveaway is the a = 0 row returning ||f|| = 0.
That is the trivial solution rhs(0) = 0, which satisfies the equation and means
nothing. Continuing a branch from a = 0.8 follows whatever solution is
connected to that one, which is not necessarily the solution the dynamics picks
out, and at small a it evidently is not: the predicted rate was off by a factor
of eight at a = 0.1.

Existence of a profile is not the interesting claim. Selection is. So start
Newton from each simulation's own profile, and ask two things per value of a:

  1. Does the simulated profile already nearly solve rhs(f) = f? A small
     residual before Newton runs means the dynamics has converged to a frozen
     profile. A large one means it has not, whatever solutions may exist.
  2. Does the resulting ||f||_inf predict that run's measured reciprocal slope?

Part 1 also gets a resolution check, since the 4e-6 agreement there is right at
the N = 2048 truncation level of the profile itself and could be luck.

    python profile_branch.py
"""

import numpy as np

import dg
from profile_eq import (guess_from_simulation, newton, residual, simulate)

print()
print("=" * 78)
print("Part 1: a = 0.8 under grid refinement")
print("=" * 78)
print("  If the agreement is real it should improve with N, since the profile")
print("  itself is only resolved to about exp(-delta k_cut).")
print()
print(f"  {'N':>6}  {'raw residual':>13}  {'||f|| equation':>15}  "
      f"{'||f|| slope':>13}  {'rel diff':>10}")
print("  " + "-" * 66)

for n in (1024, 2048, 4096):
    g, out = simulate(0.8, n=n)
    f0, amp_sim = guess_from_simulation(g, out)
    raw = float(np.abs(residual(g, f0, 0.8)).max())
    f, hist, ok = newton(g, f0, 0.8)
    if not ok:
        print(f"  {n:6d}  Newton failed, last residual {hist[-1]:.2e}")
        continue
    amp = float(np.abs(f).max())
    print(f"  {n:6d}  {raw:13.3e}  {amp:15.9f}  {amp_sim:13.9f}  "
          f"{abs(amp - amp_sim) / amp:10.2e}")


# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("Part 2: which a does the dynamics actually select a frozen profile for?")
print("=" * 78)
print("  Newton is started from each simulation's own profile, not continued")
print("  from a neighbour, so each row finds the branch that run converged to.")
print("  'raw residual' is the test that matters: it is measured before Newton")
print("  runs, so it says whether the dynamics had already found a profile.")
print()
print(f"  {'a':>6}  {'raw residual':>13}  {'||f||_inf':>11}  "
      f"{'predicted':>12}  {'measured':>12}  {'rel diff':>9}")
print("  " + "-" * 72)

for a in (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
    g, out = simulate(a)
    f0, amp_sim = guess_from_simulation(g, out)
    if f0 is None:
        print(f"  {a:6.2f}  no growth in this run")
        continue

    raw = float(np.abs(residual(g, f0, a)).max())
    f, hist, ok = newton(g, f0, a)
    amp = float(np.abs(f).max()) if ok else np.nan

    # Guard against the trivial solution, which solves the equation exactly
    # and carries no information.
    if ok and amp < 1e-6:
        print(f"  {a:6.2f}  {raw:13.3e}  {'trivial f=0':>11}  "
              f"{'-':>12}  {-1.0 / amp_sim:12.6f}  {'-':>9}")
        continue
    if not ok:
        print(f"  {a:6.2f}  {raw:13.3e}  {'no solution':>11}  "
              f"{'-':>12}  {-1.0 / amp_sim:12.6f}  {'-':>9}")
        continue

    pred, meas = -1.0 / amp, -1.0 / amp_sim
    print(f"  {a:6.2f}  {raw:13.3e}  {amp:11.6f}  {pred:12.6f}  "
          f"{meas:12.6f}  {abs(pred - meas) / abs(pred):9.2e}")

print()
print("  A small raw residual and a small rel diff together mean the blowup is")
print("  the frozen profile the equation describes. A large raw residual means")
print("  the run is still narrowing and the ansatz does not apply there, no")
print("  matter what solutions the equation happens to admit.")
