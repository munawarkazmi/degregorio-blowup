"""
Can the measurement be made good enough for the residual to mean anything?

residual2.py showed the binned estimator carries about 2.4e-2 of pure window
sensitivity: shifting the band from 128 to 512 across to 128.31 to 511.88, the
same window to three digits, moves beta from 3.00315 to 2.98626. With only
three or four bins in a fit, which modes land in which bin dominates the slope.
The +1.3e-3 residual sits twenty times below that, so it was never a measurable
quantity.

That also reframes the earlier agreement. N = 4096 and N = 8192 gave 3.00303
and 3.00291 in the same relative band, which read as accuracy but is only
reproducibility: the same deterministic window bias applied to two similar
spectra returns the same biased answer. Correlated, not independent.

The binning existed to average over the beating between the two singular
points. But least squares over every mode in a band does that too, and with
hundreds of points instead of three or four it should be far better
conditioned. If the window spread drops to the 1e-3 level then the residual
becomes resolvable and worth attributing. If it does not, the honest ceiling
here is 1e-2 and no residual at that level is a finding.

    python residual3.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8
PRED = 3.0016682


def fit_binned(k, amp, lo, hi, ratio=2.0 ** 0.5):
    floor = amp.max() * 1e-13
    kc, vc, x = [], [], float(lo)
    while x * ratio <= hi:
        y = x * ratio
        sel = (k >= x) & (k < y) & (amp > floor)
        if sel.sum() >= 3:
            kc.append(np.sqrt(x * y))
            vc.append(np.sqrt((amp[sel] ** 2).mean()))
        x = y
    if len(kc) < 3:
        return np.nan, 0
    return (-float(np.polyfit(np.log(kc), np.log(vc), 1)[0]), len(kc))


def fit_raw(k, amp, lo, hi):
    """Least squares over every mode in the band, no binning."""
    sel = (k >= lo) & (k <= hi) & (amp > amp.max() * 1e-13)
    if sel.sum() < 20:
        return np.nan, int(sel.sum())
    x, y = np.log(k[sel].astype(float)), np.log(amp[sel])
    return -float(np.polyfit(x, y, 1)[0]), int(sel.sum())


g, out = simulate(A, n=4096, w_max=1e6)
f0, _ = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, A)
amp = np.abs(g.fwd(f))
print()
print(f"  Newton profile at N = 4096, residual {hist[-1]:.2e}, "
      f"||f|| = {sup(g, f):.7f}")
print(f"  prediction from this profile: {PRED:.7f}")

WINDOWS = [(96, 512), (128, 512), (128, 724), (96, 724), (128, 384),
           (181, 724), (128, 1024), (64, 512), (150, 600), (110, 450)]

print()
print(f"  {'window':>16}  {'binned':>10}  {'bins':>5}  {'raw':>10}  "
      f"{'modes':>6}")
print("  " + "-" * 54)
binned, raw = [], []
for lo, hi in WINDOWS:
    b, nb = fit_binned(g.k, amp, lo, hi)
    r, nm = fit_raw(g.k, amp, lo, hi)
    binned.append(b)
    raw.append(r)
    bs = f"{b:10.6f}" if np.isfinite(b) else f"{'-':>10}"
    print(f"  {lo:6d} to {hi:6d}  {bs}  {nb:5d}  {r:10.6f}  {nm:6d}")

binned = [b for b in binned if np.isfinite(b)]
print()
print(f"  binned estimator spread: {max(binned) - min(binned):.3e}")
print(f"  raw estimator spread:    {max(raw) - min(raw):.3e}")
print(f"  raw estimator mean:      {np.mean(raw):.6f}   "
      f"(prediction {PRED:.6f}, difference {np.mean(raw) - PRED:+.3e})")

print()
print("=" * 74)
print("The raw estimator across resolutions, in the clean relative band")
print("=" * 74)
print(f"  {'N':>7}  {'profile':>10}  {'beta raw':>11}  {'modes':>6}  "
      f"{'predicted':>11}  {'gap':>11}")
print("  " + "-" * 62)


def evaluate(gr, field, x):
    fh = gr.fwd(field)
    c = 2.0 * fh / gr.N
    c[0] = fh[0] / gr.N
    c[-1] = fh[-1] / gr.N
    return float(np.real(np.sum(c * np.exp(1j * gr.k * x))))


def predict(gr, ff, a):
    U, Hf, _ = dg.fields(gr, ff)
    out_ = []
    for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
        x0, x1 = gr.x[i], gr.x[(i + 1) % gr.N]
        if x1 < x0:
            x1 += 2.0 * np.pi
        u0, u1 = U[i], U[(i + 1) % gr.N]
        xz = x0 + (x1 - x0) * u0 / (u0 - u1)
        c = evaluate(gr, Hf, xz % (2 * np.pi))
        out_.append((c - 1.0) / (a * c))
    return sorted(out_)[1] + 1.0


for n in (2048, 4096, 8192):
    gr, o = simulate(A, n=n, w_max=1e6)
    fs, _ = guess_from_simulation(gr, o)
    if n <= 4096:
        ff, h, good = newton(gr, fs, A)
        tag = "Newton" if good else "simulated"
        if not good:
            ff = fs
    else:
        ff, tag = fs, "simulated"
    ar = np.abs(gr.fwd(ff))
    b, nm = fit_raw(gr.k, ar, 0.094 * gr.kcut, 0.375 * gr.kcut)
    p = predict(gr, ff, A)
    print(f"  {n:7d}  {tag:>10}  {b:11.6f}  {nm:6d}  {p:11.6f}  "
          f"{b - p:+11.3e}")
