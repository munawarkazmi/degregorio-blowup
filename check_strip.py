"""
Control test for dg.strip_width against the one case where the analyticity
strip is known exactly.

For the a = 0 solution from sin(x), the denominator 4 + 4 t cos z + t^2
vanishes at cos z = -(4 + t^2) / (4t), so the strip half width is

    delta(t) = arccosh((4 + t^2) / (4 t)),

and since ||w||_inf = 4 / (4 - t^2), the product ||w||_inf * delta tends to
1/2 as t -> 2. If the measurement cannot reproduce that here, it means nothing
at any other value of a.
"""

import numpy as np

import dg

g = dg.Grid(8192)
out = dg.run(np.sin(g.x), a=0.0, grid=g, t_max=2.0, cfl=0.01, tail_tol=1e-9)
t, w, d = out["t_hist"], out["winf_hist"], out["delta_hist"]

print()
print("delta should equal arccosh((4 + t^2) / (4t)), and w * delta -> 1/2")
print()
print(f"{'t':>9} {'||w||':>10} {'delta meas':>11} {'delta exact':>12} "
      f"{'rel err':>9} {'w*delta':>9}")
print("-" * 66)
for frac in (0.5, 0.7, 0.85, 0.95, 0.99, 1.0):
    i = int(frac * (len(t) - 1))
    de = float(np.arccosh((4.0 + t[i] ** 2) / (4.0 * t[i])))
    print(f"{t[i]:9.5f} {w[i]:10.3f} {d[i]:11.6f} {de:12.6f} "
          f"{abs(d[i] - de) / de:9.2e} {w[i] * d[i]:9.5f}")
