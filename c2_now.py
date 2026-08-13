"""
The control variate correction, applied to the table c2_solve.py already
printed. No solves, just the analysis step.

c1 has a known exact value, 1/(1-a), proved in finding 8. Its measured wander
across resolutions is therefore a direct gauge of the discretisation error in
reading anything off the profile. c2 is read off the same profile and carries
the same error, so regressing c2 against c1 and evaluating at the exact c1
removes it, without needing to know the error's size or its rate in N.
"""

import numpy as np

A = 0.8
N = np.array([512, 768, 1024, 1536, 2048, 3072, 4096])
C1 = np.array([4.98704376, 4.99127266, 5.00692610, 4.99557146,
               4.99666292, 4.99776928, 5.00170061])
C2 = np.array([-1.64861379, -1.65274178, -1.66810312, -1.65694968,
               -1.65801989, -1.65910542, -1.66296777])

exact = 1.0 / (1.0 - A)
s, b = np.polyfit(C1, C2, 1)
resid = C2 - (s * C1 + b)
r2 = 1.0 - (resid ** 2).sum() / ((C2 - C2.mean()) ** 2).sum()
per_point = C2 - s * (C1 - exact)

print()
print(f"  exact c1 = {exact:.8f}")
print(f"  regression c2 = {s:.6f} c1 + {b:.6f},   R^2 = {r2:.9f}")
print()
print(f"  {'N':>6}  {'c1 - 5':>11}  {'c2 raw':>13}  {'c2 corrected':>14}")
print("  " + "-" * 50)
for n, a1, a2, cc in zip(N, C1, C2, per_point):
    print(f"  {n:6d}  {a1 - exact:+11.2e}  {a2:13.8f}  {cc:14.8f}")

c2c = float(per_point.mean())
sd = float(per_point.std(ddof=1))
print()
print(f"  raw spread       {C2.max() - C2.min():.3e}")
print(f"  corrected spread {per_point.max() - per_point.min():.3e}   "
      f"({(C2.max() - C2.min()) / (per_point.max() - per_point.min()):.0f}x "
      f"better)")
print()
print(f"  c2   = {c2c:.8f} +/- {sd:.1e}")

mu2 = (c2c - 1.0) / (A * c2c)
dmu = sd / (A * c2c ** 2)
print(f"  mu2  = {mu2:.8f} +/- {dmu:.1e}")
print(f"  beta = {mu2 + 1.0:.8f}")

target = 1.0 / (1.0 - 2.0 * A)
print()
print(f"  mu2 = 2 exactly would need c2 = 1/(1-2a) = {target:.8f}")
print(f"  measured c2 is off by {c2c - target:+.3e}, "
      f"which is {abs(c2c - target) / sd:.0f} sigma")
print(f"  mu2 - 2 = {mu2 - 2.0:+.3e}, which is {abs(mu2 - 2.0) / dmu:.0f} sigma")
print()
if abs(mu2 - 2.0) > 5 * dmu:
    print("  So mu2 is not 2 and beta is not 3, now decisively rather than")
    print("  by a margin smaller than the scatter.")
