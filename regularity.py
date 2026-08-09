"""
How smooth is the blowup profile?

shape_check.py found the a = 0.8 profile is broad and gentle: max |f'| = 7.04
against ||f||_inf = 4.16, so the steepest feature is about 0.59 wide, and the
profile is exactly odd, with max +4.16122 at y = 0 and min -4.16122 at
y = 1.95. Nothing about that shape suggests a nearby complex singularity.

Yet its spectrum is still at 1e-10 by k = 900. An analytic profile decays like
exp(-delta k), which for a feature that broad should have exhausted double
precision long before k = 900. Algebraic decay k^-beta instead means a
singularity sitting on the real axis and a profile of finite regularity: beta
maps to roughly C^(beta-1).

This is not a detail. It decides what kind of object a proof would have to
enclose, and it would explain why ||f||_inf converges so slowly in N, moving
1.5e-3 between N = 2048 and N = 4096 when spectral truncation should have given
far better.

Fit both laws over the same window and compare residuals.

    python regularity.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup


def fit(x, y):
    m, c = np.polyfit(x, y, 1)
    r = y - (m * x + c)
    ss = ((y - y.mean()) ** 2).sum()
    return m, (1.0 - (r ** 2).sum() / ss if ss > 0 else np.nan)


print()
print(f"{'N':>7}  {'||f||_inf':>11}  {'exponential':>22}  {'algebraic':>24}")
print(f"{'':>7}  {'':>11}  {'delta':>10} {'R^2':>11}  "
      f"{'beta':>10} {'R^2':>13}")
print("-" * 70)

for n in (1024, 2048, 4096):
    g, out = simulate(0.8, n=n)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, 0.8)
    if not ok:
        print(f"{n:7d}  Newton failed")
        continue

    amp = np.abs(g.fwd(f))
    k = g.k
    # Window: above the roundoff floor, below the energy containing modes.
    sel = (amp > amp.max() * 1e-13) & (amp < amp.max() * 1e-2) & (k > 0)
    kk = k[sel].astype(float)
    ly = np.log(amp[sel])

    d_slope, d_r2 = fit(kk, ly)                 # log amp vs k
    b_slope, b_r2 = fit(np.log(kk), ly)         # log amp vs log k

    print(f"{n:7d}  {sup(g, f):11.6f}  {-d_slope:10.6f} {d_r2:11.7f}  "
          f"{-b_slope:10.6f} {b_r2:13.7f}")

print()
print("The larger R^2 wins. Algebraic decay k^-beta means the profile is")
print("roughly C^(beta-1) and no better, so its Fourier series converges at an")
print("algebraic rate and every quantity read off it inherits that rate. A")
print("beta that holds steady as N grows, rather than drifting, is the sign")
print("that this is the profile's own regularity and not a truncation artefact.")
