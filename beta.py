"""
Is the spectral decay exponent exactly 3?

regularity.py fitted |f_k| ~ k^-beta over one wide window and got 3.044, 3.048,
3.045 at N = 1024, 2048, 4096. Stable across N, and 1.5 percent off 3. Taken at
face value that says beta is not 3. But a single wide window fit has two known
ways to be biased here, and both push the same way.

  A log correction. If |f_k| = C k^-3 (log k)^gamma, the local log-log slope is
  -3 + gamma / log k, which over k from 10 to 1300 varies only between -3.09
  and -3.03 for gamma = -0.2. A wide window fit would report a stable 3.04 and
  hide a true exponent of 3.

  Oscillation. A singularity at a single point y0 gives f_k ~ C k^-beta
  e^(i k y0), whose modulus is clean. Two singular points give a sum of two
  such terms and |f_k| then beats, which shows up as the band thickening
  visible at high k in fig_profile. Fitting through a beating signal biases the
  slope by however the beats happen to fall in the window.

So do three things instead of one wide fit.

  1. Local slope by octave. If beta_local(k) drifts toward 3 as k grows, the
     exponent is 3 with a correction. If it sits flat at 3.045, it is not 3.
  2. Fit the pure power law against a power law with a log correction on the
     same data and compare residuals.
  3. Locate the singularity. The phase of f_k advances by y0 per unit k, so the
     phase increment identifies where the singular point is, and whether there
     is more than one. In real space, beta = 3 means precisely a jump in f'',
     since a jump in the m-th derivative gives k^-(m+1).

    python beta.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup


def profile_at(n, a=0.8, solve=True):
    """Newton-solved profile where affordable, simulated profile otherwise."""
    g, out = simulate(a, n=n, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    if not solve:
        return g, f0, "simulated"
    f, hist, ok = newton(g, f0, a)
    return (g, f, "newton") if ok else (g, f0, "simulated")


def binned_spectrum(g, f, k_lo=8, ratio=2.0 ** 0.5):
    """
    RMS amplitude in geometric bins. Binning averages over the beats, which a
    pointwise fit cannot do, and the RMS is the right average because the
    beating is in amplitude rather than in the envelope.
    """
    amp = np.abs(g.fwd(f))
    k = g.k
    floor = amp.max() * 1e-13
    centres, vals = [], []
    lo = float(k_lo)
    while lo * ratio <= g.kcut:
        hi = lo * ratio
        sel = (k >= lo) & (k < hi) & (amp > floor)
        if sel.sum() >= 3:
            centres.append(np.sqrt(lo * hi))
            vals.append(np.sqrt((amp[sel] ** 2).mean()))
        lo = hi
    return np.array(centres), np.array(vals)


print()
print("=" * 78)
print("Part 1: local slope by octave. Does beta drift toward 3?")
print("=" * 78)
print("  Slope between adjacent geometric bins, so each row is a local")
print("  measurement rather than an average over the whole range.")

records = {}
for n, solve in ((4096, True), (8192, False), (16384, False)):
    g, f, kind = profile_at(n, solve=solve)
    kc, vc = binned_spectrum(g, f)
    slopes = -np.diff(np.log(vc)) / np.diff(np.log(kc))
    mids = np.sqrt(kc[:-1] * kc[1:])
    records[n] = (g, f, kind, kc, vc, mids, slopes)
    print()
    print(f"  N = {n}  ({kind} profile, ||f||_inf = {sup(g, f):.6f})")
    print(f"    {'k':>9}  {'local beta':>11}")
    for m, s in zip(mids, slopes):
        if m > 12:
            print(f"    {m:9.1f}  {s:11.5f}")

print()
print("=" * 78)
print("Part 2: pure power law against a power law with a log correction")
print("=" * 78)
print("  Model A:  log|f_k| = c - beta log k")
print("  Model B:  log|f_k| = c - beta log k + gamma log(log k)")
print("  If B fits better and returns beta near 3, the exponent is 3 and the")
print("  1.045 excess in the wide fit was the log term being absorbed.")
print()
print(f"  {'N':>7}  {'A: beta':>9}  {'A: resid':>10}  {'B: beta':>9}  "
      f"{'B: gamma':>10}  {'B: resid':>10}")
print("  " + "-" * 66)

for n, (g, f, kind, kc, vc, mids, slopes) in records.items():
    use = kc > 12
    x, y = np.log(kc[use]), np.log(vc[use])
    ma, ca = np.polyfit(x, y, 1)
    ra = float(np.sqrt(((y - (ma * x + ca)) ** 2).mean()))

    M = np.column_stack([np.ones_like(x), -x, np.log(x)])
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    rb = float(np.sqrt(((y - M @ coef) ** 2).mean()))
    print(f"  {n:7d}  {-ma:9.5f}  {ra:10.3e}  {coef[1]:9.5f}  "
          f"{coef[2]:10.5f}  {rb:10.3e}")

print()
print("=" * 78)
print("Part 3: where is the singularity, and what kind is it?")
print("=" * 78)

g, f, kind, kc, vc, mids, slopes = records[16384]
fh = g.fwd(f)
k = g.k

band = (k > g.kcut // 4) & (k < g.kcut) & (np.abs(fh) > np.abs(fh).max() * 1e-13)
idx = np.flatnonzero(band)
phase = np.unwrap(np.angle(fh[idx]))
dphase = np.diff(phase) / np.diff(idx.astype(float))
y0 = float(np.median(dphase)) % (2.0 * np.pi)
print(f"  median phase advance per mode = {y0:.6f}")
print(f"  so the singular point sits at y = {y0:.6f} "
      f"(or equivalently {y0 - 2 * np.pi:.6f})")
print(f"  spread in the phase advance: {float(np.std(dphase)):.3e}   "
      f"(small means a single dominant singular point)")

fx = g.bwd(fh * g.dmul)
print()
print(f"  for reference, f = 0 and |f'| is largest at "
      f"y = {g.x[int(np.argmax(np.abs(fx)))]:.6f}")
print(f"  f peaks at y = {g.x[int(np.argmax(f))]:.6f} and troughs at "
      f"y = {g.x[int(np.argmin(f))]:.6f}")

print()
print("  beta = 3 means a jump in f''. Sampling f'' across the located point:")
fxx = g.bwd(fh * g.dmul * g.dmul)
j = int(np.argmin(np.abs(g.x - y0)))
for d in (-40, -20, -8, -3, -1, 0, 1, 3, 8, 20, 40):
    i = (j + d) % g.N
    print(f"    y = {g.x[i]:9.6f}   f'' = {fxx[i]:12.5f}")
