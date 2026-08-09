"""
Grid refinement in diagnose.py established that the large amplitude runs are
real: across a 4x range in N the amplitude histories agree to 1e-5 or better,
and at a = 0.8 every resolution reaches 1e12 at t = 6.072 with a spectral tail
of 1e-14. Nothing is under-resolved. So characterise the growth.

Two questions, in order.

  1. Is it a power law, w ~ C (T - t)^-p, or an exponential? Power law means
     1 / w is a straight line in t hitting zero at T; exponential means log w
     is the straight line and there is no finite T.

  2. If it is a power law, is it self-similar? The test is the product
     ||w||_inf * delta, where delta is the analyticity strip half width read
     off the spectral decay rate. For the a = 0 exact solution this product
     tends to 1/2, because delta ~ (2 - t) / 2 while ||w||_inf ~ 1 / (2 - t).
     A product that stays put means the peak narrows in step with its height.
     That case is the control: whatever the code reports at a = 0 must come
     out at 1/2, or the measurement is not trustworthy anywhere else.

    python mechanism.py
"""

import numpy as np

import dg


def linearity(x, y):
    """R^2 of a straight line fit, as a shape test."""
    m, c = np.polyfit(x, y, 1)
    r = y - (m * x + c)
    ss = ((y - y.mean()) ** 2).sum()
    return (1.0 - (r ** 2).sum() / ss) if ss > 0 else np.nan


print()
print("Part 1: which growth law, and where does the singularity sit?")
print()
print(f"{'a':>5}  {'law':>12}  {'R2 power':>9}  {'R2 exp':>9}  {'T':>9}  "
      f"{'p':>7}  {'growth':>9}  {'stop':>15}")
print("-" * 88)

runs = {}
for a in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
    # N = 2048 is justified by diagnose.py, which found the amplitude histories
    # for these cases agree to 1e-5 or better between N = 1024 and N = 4096.
    g = dg.Grid(2048)
    out = dg.run(np.sin(g.x), a=a, grid=g, t_max=25.0, cfl=0.02,
                 tail_tol=1e-9, w_max=1e8)
    runs[a] = (g, out)

    t, w = out["t_hist"], out["winf_hist"]
    growth = w[-1] / w[0]
    if growth < 100.0:
        print(f"{a:5.2f}  {'no growth':>12}  {'-':>9}  {'-':>9}  {'inf':>9}  "
              f"{'-':>7}  {growth:9.2f}  {out['reason']:>15}")
        continue

    sel = w >= w[-1] * 1e-2          # last two decades
    i0 = int(np.flatnonzero(~sel)[-1]) + 1 if (~sel).any() else 0
    tt, ww = t[i0:], w[i0:]
    r2_pow = linearity(tt, 1.0 / ww)
    r2_exp = linearity(tt, np.log(ww))
    T, p, _ = dg.fit_blowup(t, w, window_decades=0.5)

    law = "power law" if r2_pow > r2_exp else "exponential"
    tstr = f"{T:9.4f}" if np.isfinite(T) else f"{'inf':>9}"
    pstr = f"{p:7.4f}" if np.isfinite(p) else f"{'-':>7}"
    print(f"{a:5.2f}  {law:>12}  {r2_pow:9.6f}  {r2_exp:9.6f}  {tstr}  "
          f"{pstr}  {growth:9.2e}  {out['reason']:>15}")


print()
print()
print("Part 2: is the peak narrowing in step with its height?")
print("||w||_inf * delta along each run. The a = 0 row is the control and")
print("must approach 0.5, which is the exact value for the CLM solution.")
print()
hdr = f"{'a':>5}  " + "  ".join(f"{'w=1e' + str(e):>12}" for e in range(1, 8))
print(hdr)
print("-" * len(hdr))

for a in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
    g, out = runs[a]
    w, d = out["winf_hist"], out["delta_hist"]
    cells = []
    for e in range(1, 8):
        target = 10.0 ** e
        if w[-1] < target or w[0] > target:
            cells.append(f"{'-':>12}")
            continue
        i = int(np.argmin(np.abs(w - target)))
        val = w[i] * d[i]
        cells.append(f"{val:12.4f}" if np.isfinite(val) else f"{'nan':>12}")
    print(f"{a:5.2f}  " + "  ".join(cells))

print()
print("A column that holds steady across a row is self-similar narrowing.")
print("A column that grows means the amplitude outruns the peak's sharpening.")
