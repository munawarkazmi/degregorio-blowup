"""
Locate the branch merger from the c_l table measured by exchange2.py.

Refits the measured values without re-running the simulations, and fixes the
scan range bug: a_c lies below the largest a sampled, so a search starting at
that largest a pins to its own boundary and reports it for every exponent.

The question this settles is narrow and specific. Is c_l zero at a = 0.751?
If it is significantly positive there, the merger is not at that value and the
numerical coincidence with Huang, Tong and Wang is refuted rather than merely
unsupported.
"""

import numpy as np

# a, c_l, standard error, from exchange2.py at N = 2048, three decades of
# amplitude, fitted above amplitude 100.
DATA = [
    (0.70, 0.032321, 1.45e-03),
    (0.72, 0.016646, 1.23e-03),
    (0.74, 0.007273, 7.46e-04),
    (0.75, 0.004536, 5.43e-04),
    (0.76, 0.002701, 3.99e-04),
    (0.77, 0.001294, 2.99e-04),
    (0.78, 0.000396, 1.56e-04),
    (0.79, 0.000716, 1.47e-04),
    (0.80, 0.000398, 1.21e-04),
    (0.82, -0.000029, 1.41e-05),
    (0.84, 0.000011, 7.82e-06),
]

av = np.array([d[0] for d in DATA])
cv = np.array([d[1] for d in DATA])
sev = np.array([d[2] for d in DATA])

print()
print("  significance of c_l against zero:")
print(f"  {'a':>6}  {'c_l':>12}  {'std err':>10}  {'sigma':>7}")
print("  " + "-" * 40)
for a, c, s in DATA:
    print(f"  {a:6.2f}  {c:12.6f}  {s:10.2e}  {c / s:7.1f}")

print()
print("  Non monotonicity at a = 0.78, 0.79, 0.80 (3.96e-4, 7.16e-4, 3.98e-4)")
print("  exceeds the quoted standard errors, so the real noise floor is about")
print("  5e-4 rather than the 1.5e-4 the regression reports.")

print()
print("=" * 66)
print("Is c_l zero at a = 0.751?")
print("=" * 66)
i = list(av).index(0.75)
print(f"  c_l(0.75) = {cv[i]:.6f} +/- {sev[i]:.6f}, "
      f"which is {cv[i] / sev[i]:.1f} standard errors from zero")
print(f"  even against a conservative 5e-4 noise floor it is "
      f"{cv[i] / 5e-4:.1f} floors from zero")

print()
print("=" * 66)
print("Where is the merger?")
print("=" * 66)
use = cv > 4.0 * sev
print(f"  fitting c_l = A (a_c - a)^p over the {use.sum()} points with "
      f"c_l > 4 sigma, a <= {av[use].max():.2f}")
lo = av[use].max() + 0.005

best = None
for p in np.linspace(1.0, 6.0, 1001):
    for ac in np.linspace(lo, lo + 0.20, 801):
        pred = (ac - av[use]) ** p
        A = float((pred * cv[use]).sum() / (pred * pred).sum())
        ss = float((((cv[use] - A * pred) / sev[use]) ** 2).sum())
        if best is None or ss < best[0]:
            best = (ss, p, ac, A)
ss, p, ac, A = best
print(f"  best fit: p = {p:.2f}, a_c = {ac:.4f}, "
      f"chi squared = {ss:.2f} on {use.sum() - 3} degrees of freedom")

print()
print("  degeneracy, best a_c at each fixed p:")
print(f"  {'p':>6}  {'a_c':>8}  {'chi sq':>9}")
print("  " + "-" * 28)
for pf in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
    b = None
    for ac2 in np.linspace(lo, lo + 0.20, 801):
        pred = (ac2 - av[use]) ** pf
        A2 = float((pred * cv[use]).sum() / (pred * pred).sum())
        s2 = float((((cv[use] - A2 * pred) / sev[use]) ** 2).sum())
        if b is None or s2 < b[0]:
            b = (s2, ac2)
    print(f"  {pf:6.1f}  {b[1]:8.4f}  {b[0]:9.2f}")

print()
print("  The exponent and the root trade off against each other, so a_c is")
print("  bracketed rather than determined. Compare a_c2 = 0.751 from Huang,")
print("  Tong and Wang.")
