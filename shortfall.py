"""
Where does the 1 percent shortfall in measured beta come from?

At a = 0.8 the stagnation point formula predicts beta = 3.001668 and the
binned spectral fit returns 2.975974, low by 0.9 percent, and the same sign of
error appears at every a. Parity was ruled out: even and odd modes differ by
only 12 percent in RMS and fitting them separately moves the answer the wrong
way.

The prediction side is not the problem. mu2 shifts by only about 4e-3 between
N = 1024 and N = 4096, an order below the 0.026 gap, so the error lives in the
measurement. Four candidates, each with a different fingerprint.

  Low k contamination. The spectrum only settles onto its power law somewhere
  past k = 10, and analytic corrections to f ~ C|y-y*|^mu contribute k^-(mu+2)
  and faster. Those are steeper than the leading term, so they would bias a
  fitted slope upward at the low end, which is the wrong direction. Testable by
  raising the lower edge of the window: if this is it, beta should rise toward
  3.0017 and stay there.

  A k^-2 component. If f' jumps at the mu = 1 stagnation point, that point
  contributes k^-2 rather than nothing, and a shallower term drags the fitted
  slope down exactly as observed. The fingerprint is a local slope that falls
  monotonically toward 2 as k grows. The direct check is whether f' is actually
  discontinuous there. If it is continuous, this dies.

  Dealiasing damage near k_cut. The profile solves the masked equation, so its
  highest retained modes may be distorted. Testable by dropping the top octave.

  Nothing systematic, just the fit. Testable by whether the answer depends on
  the window at all.

    python shortfall.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8
PRED = 3.001668


def refine_zero(g, U, i):
    x0, x1 = g.x[i], g.x[(i + 1) % g.N]
    if x1 < x0:
        x1 += 2.0 * np.pi
    u0, u1 = U[i], U[(i + 1) % g.N]
    return x0 + (x1 - x0) * u0 / (u0 - u1)


def fit_band(k, amp, k_lo, k_hi, ratio=2.0):
    """Geometric bins, RMS in each, least squares slope of log against log."""
    floor = amp.max() * 1e-13
    kc, vc, lo = [], [], float(k_lo)
    while lo * ratio <= k_hi:
        hi = lo * ratio
        sel = (k >= lo) & (k < hi) & (amp > floor)
        if sel.sum() >= 3:
            kc.append(np.sqrt(lo * hi))
            vc.append(np.sqrt((amp[sel] ** 2).mean()))
        lo = hi
    if len(kc) < 3:
        return np.nan, np.nan, 0
    x, y = np.log(np.array(kc)), np.log(np.array(vc))
    m, c = np.polyfit(x, y, 1)
    r = y - (m * x + c)
    ss = ((y - y.mean()) ** 2).sum()
    return -m, (1.0 - (r ** 2).sum() / ss if ss > 0 else np.nan), len(kc)


g, out = simulate(A, n=4096, w_max=1e6)
f0, _ = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, A)
print()
print(f"profile: converged {ok}, residual {hist[-1]:.2e}, "
      f"||f||_inf = {sup(g, f):.6f},  predicted beta = {PRED:.6f}")

amp = np.abs(g.fwd(f))
k = g.k

print()
print("=" * 74)
print("Test 1: raise the lower edge of the fit window")
print("=" * 74)
print(f"  {'k_lo':>7}  {'k_hi':>7}  {'beta':>10}  {'error':>10}  "
      f"{'R^2':>10}  {'bins':>5}")
print("  " + "-" * 58)
for k_lo in (8, 16, 32, 64, 128, 256):
    b, r2, n = fit_band(k, amp, k_lo, g.kcut)
    print(f"  {k_lo:7d}  {g.kcut:7d}  {b:10.6f}  {b - PRED:+10.6f}  "
          f"{r2:10.6f}  {n:5d}")

print()
print("=" * 74)
print("Test 2: drop the top octave, in case the mask distorts it")
print("=" * 74)
print(f"  {'k_lo':>7}  {'k_hi':>7}  {'beta':>10}  {'error':>10}  {'R^2':>10}")
print("  " + "-" * 52)
for k_hi in (g.kcut, g.kcut // 2, g.kcut // 4):
    b, r2, n = fit_band(k, amp, 16, k_hi)
    print(f"  {16:7d}  {k_hi:7d}  {b:10.6f}  {b - PRED:+10.6f}  {r2:10.6f}")

print()
print("=" * 74)
print("Test 3: local slope over single octaves, to see the trend bare")
print("=" * 74)
print("  A slope drifting down toward 2 is the k^-2 fingerprint. A slope that")
print("  rises to 3.0017 and flattens means the bias was at the low end.")
print()
print(f"  {'band':>16}  {'beta':>10}  {'error':>10}")
print("  " + "-" * 40)
lo = 8.0
while lo * 4 <= g.kcut:
    b, _, _ = fit_band(k, amp, lo, lo * 4, ratio=2.0 ** 0.5)
    print(f"  {int(lo):6d} to {int(lo * 4):6d}  {b:10.6f}  {b - PRED:+10.6f}")
    lo *= 2

print()
print("=" * 74)
print("Test 4: is f' discontinuous at the mu = 1 stagnation point?")
print("=" * 74)
print("  A jump there would put a k^-2 term in the spectrum. Continuity to")
print("  machine precision kills that explanation outright.")

U, Hf, fx = dg.fields(g, f)
fxx = g.bwd(g.fwd(f) * g.dmul * g.dmul)
zeros = [refine_zero(g, U, i) % (2 * np.pi)
         for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1)))]
for ys in sorted(zeros):
    c = float(np.interp(ys, g.x, Hf))
    mu = (c - 1.0) / (A * c)
    j = int(np.argmin(np.abs(g.x - ys)))
    print()
    print(f"  y* = {ys:.6f},  mu = {mu:.5f}")
    print(f"    {'offset':>9}  {'f_x right':>12}  {'f_x left':>12}  "
          f"{'difference':>12}")
    for d in (12, 24, 48, 96):
        ip, im = (j + d) % g.N, (j - d) % g.N
        print(f"    {d * g.dx:9.5f}  {fx[ip]:12.6f}  {fx[im]:12.6f}  "
              f"{fx[ip] - fx[im]:12.3e}")
    lim_r = float(np.mean([fx[(j + d) % g.N] for d in (12, 18, 24)]))
    lim_l = float(np.mean([fx[(j - d) % g.N] for d in (12, 18, 24)]))
    print(f"    f' jump across y* is about {lim_r - lim_l:.3e}, "
          f"against |f'| ~ {abs(lim_r):.3f}")
