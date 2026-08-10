"""
Parts 2 and 3 of c_of_a.py, which never ran.

Part 1 finished and gave the mu1 = 1 identity out to a = 0.84. The sweep then
stalled: at a = 0.86 and 0.88 the profile is too rough for N = 4096, since mu2
is falling toward 1 there, so Newton ran its full 25 iterations without
converging and each of those is a dense 4096 by 4096 solve. Those two values
were never going to produce a number, so take the Part 1 output as given and do
the two analyses that were waiting behind it.

    python c_of_a2.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

# From c_of_a.py Part 1, at N = 4096.
AV = np.array([0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84])
MU2 = np.array([15.362001, 5.795385, 3.736745, 2.835871, 2.328739,
                2.001668, 1.770433, 1.594342])

print()
print("=" * 74)
print("Part 2: the shape of mu2(a)")
print("=" * 74)
print("  1/mu2 rises smoothly with a. If it were linear, its zero would mark")
print("  where the profile turns analytic and 1/mu2 = 1 where it leaves C^1.")
print()
print(f"  {'a':>6}  {'mu2':>10}  {'1/mu2':>9}  {'d(1/mu2)/da':>12}")
print("  " + "-" * 44)
inv = 1.0 / MU2
slopes = np.diff(inv) / np.diff(AV)
for i, a in enumerate(AV):
    s = f"{slopes[i]:12.4f}" if i < len(slopes) else f"{'':>12}"
    print(f"  {a:6.2f}  {MU2[i]:10.6f}  {inv[i]:9.6f}  {s}")

m, c0 = np.polyfit(AV, inv, 1)
res = inv - (m * AV + c0)
ss = ((inv - inv.mean()) ** 2).sum()
print()
print(f"  linear fit: 1/mu2 = {m:.5f} a + {c0:.5f},  "
      f"R^2 = {1.0 - (res ** 2).sum() / ss:.6f}")
print(f"  residuals: " + "  ".join(f"{r:+.4f}" for r in res))
print()
print("  The slope column falls monotonically from 5.37 to 3.12, so 1/mu2 is")
print("  convex and not linear. A straight line through it fits well by R^2")
print("  and still extrapolates badly, which is exactly the trap that produced")
print("  the biased beta earlier. Quote the curvature, not the intercept.")

for lo in (2, 4):
    m2, c2 = np.polyfit(AV[lo:], inv[lo:], 1)
    print(f"    using only a >= {AV[lo]:.2f}: 1/mu2 = 0 at a = {-c2 / m2:.4f}, "
          f"1/mu2 = 1 at a = {(1.0 - c2) / m2:.4f}")
print("  Those two windows disagree, which settles it: no reliable")
print("  extrapolation to where the profile turns analytic.")


print()
print("=" * 74)
print("Part 3: are the even modes small, and did that bias beta?")
print("=" * 74)

g, out = simulate(0.80, n=4096, w_max=1e6)
f0, _ = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, 0.80)
print(f"  profile: converged {ok}, residual {hist[-1]:.2e}, "
      f"||f||_inf = {sup(g, f):.6f}")

amp = np.abs(g.fwd(f))
print()
print("  |f_k| for the first modes:")
for k in range(1, 11):
    print(f"    k = {k:3d}   {amp[k]:.6e}")

hi = (g.k > 100) & (g.k < g.kcut)
ev = hi & (g.k % 2 == 0)
od = hi & (g.k % 2 == 1)
re_, ro = np.sqrt((amp[ev] ** 2).mean()), np.sqrt((amp[od] ** 2).mean())
print()
print(f"  over 100 < k < k_cut:   rms even = {re_:.4e}   "
      f"rms odd = {ro:.4e}   ratio = {re_ / ro:.4e}")


def beta_from(parity):
    k, floor = g.k, amp.max() * 1e-13
    kc, vc, lo = [], [], 8.0
    while lo * np.sqrt(2.0) <= g.kcut:
        hi_ = lo * np.sqrt(2.0)
        sel = (k >= lo) & (k < hi_) & (amp > floor)
        if parity is not None:
            sel = sel & (k % 2 == parity)
        if sel.sum() >= 3:
            kc.append(np.sqrt(lo * hi_))
            vc.append(np.sqrt((amp[sel] ** 2).mean()))
        lo = hi_
    kc, vc = np.array(kc), np.array(vc)
    use = kc > 12
    x, y = np.log(kc[use]), np.log(vc[use])
    mm, cc = np.polyfit(x, y, 1)
    r = y - (mm * x + cc)
    s = ((y - y.mean()) ** 2).sum()
    return -mm, 1.0 - (r ** 2).sum() / s


print()
print(f"  predicted beta at a = 0.8 (from mu2 + 1): {MU2[5] + 1:.6f}")
for label, parity in (("all modes", None), ("odd k only", 1),
                      ("even k only", 0)):
    b, r2 = beta_from(parity)
    print(f"    measured, {label:12s}: beta = {b:9.6f}   R^2 = {r2:.6f}")
