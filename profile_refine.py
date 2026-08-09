"""
Two loose ends from profile_branch.py.

First, its Part 1 compared a grid maximum against a peak height derived from
interpolated sup norms, so the agreement did not improve with N and ||f||_inf
wandered by 3e-3 across resolutions. With both sides measured the same way it
should now converge.

Second, a = 0.9 showed a raw residual of 2.9, far worse than its neighbours at
0.85 and 0.8. The natural reading is not that the profile stops existing at
0.9 but that the run stopped too early: T(0.9) is about 10.8 against 6.1 at
a = 0.8, so at a fixed amplitude cap of 1e6 that solution has had less time to
settle onto its profile. If that is right, running further must drive the
residual down.

    python profile_refine.py
"""

import numpy as np

import dg
from profile_eq import (guess_from_simulation, newton, residual, simulate,
                        sup)

print()
print("=" * 72)
print("Part 1: a = 0.8 under grid refinement, both sides measured alike")
print("=" * 72)
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
    amp = sup(g, f)
    print(f"  {n:6d}  {raw:13.3e}  {amp:15.9f}  {amp_sim:13.9f}  "
          f"{abs(amp - amp_sim) / amp:10.2e}")


print()
print("=" * 72)
print("Part 2: a = 0.9, run further")
print("=" * 72)
print("  If a = 0.9 is the same phenomenon caught earlier in its approach,")
print("  the residual falls as the amplitude cap rises. If the profile really")
print("  does not exist there, it will not.")
print()
print(f"  {'w_max':>9}  {'t reached':>11}  {'raw residual':>13}  "
      f"{'predicted':>11}  {'measured':>11}  {'rel diff':>9}")
print("  " + "-" * 68)

for w_max in (1e4, 1e6, 1e8):
    g, out = simulate(0.9, w_max=w_max)
    f0, amp_sim = guess_from_simulation(g, out)
    if f0 is None:
        print(f"  {w_max:9.0e}  no usable growth")
        continue
    raw = float(np.abs(residual(g, f0, 0.9)).max())
    f, hist, ok = newton(g, f0, 0.9)
    if not ok or sup(g, f) < 1e-6:
        print(f"  {w_max:9.0e}  {out['t']:11.5f}  {raw:13.3e}  "
              f"{'no solution':>11}  {-1.0 / amp_sim:11.6f}  {'-':>9}")
        continue
    amp = sup(g, f)
    pred, meas = -1.0 / amp, -1.0 / amp_sim
    print(f"  {w_max:9.0e}  {out['t']:11.5f}  {raw:13.3e}  {pred:11.6f}  "
          f"{meas:11.6f}  {abs(pred - meas) / abs(pred):9.2e}")
