"""
The decisive test of beta = 1 + (c - 1) / (a c).

beta_jump.py showed mu is not universal. It runs 15.36, 3.21, 2.00, 1.52 as a
goes 0.70, 0.75, 0.80, 0.85, so beta is a continuously varying function of a
that happens to pass through 3 near a = 0.8. That reading rests entirely on the
stagnation point formula, which so far has only been checked against a measured
spectrum at a = 0.8, where it gave 3.0017 against a measured 3.045.

One agreement at one value of a is not enough, especially when the number in
question is close to a round one. So predict beta at other values of a from the
local quantity c = H f(y*), and measure the spectral decay there independently.
The predictions are far apart, 4.21 against 2.52, so this cannot pass by luck.

The measurement bins geometrically and takes the RMS within each bin, which
averages over the beating caused by the two stagnation points sitting pi apart.
Fitting pointwise through that beat is what produced the biased 3.045 in the
first place.

    python beta_predict.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup


def refine_zero(g, U, i):
    x0, x1 = g.x[i], g.x[(i + 1) % g.N]
    if x1 < x0:
        x1 += 2.0 * np.pi
    u0, u1 = U[i], U[(i + 1) % g.N]
    return x0 + (x1 - x0) * u0 / (u0 - u1)


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def predicted_beta(g, f, a):
    U, Hf, _ = dg.fields(g, f)
    best = None
    for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
        ys = refine_zero(g, U, i) % (2.0 * np.pi)
        c = evaluate(g, Hf, ys)
        mu = (c - 1.0) / (a * c)
        # The governing singularity is the least smooth one that is genuinely
        # singular. mu = 1 with matching one sided constants is smooth, so skip
        # anything at an integer below 2.
        if mu > 1.5 and (best is None or mu < best[2]):
            best = (ys, c, mu)
    return best


def measured_beta(g, f, k_lo=8, ratio=2.0 ** 0.5):
    amp = np.abs(g.fwd(f))
    k = g.k
    floor = amp.max() * 1e-13
    kc, vc = [], []
    lo = float(k_lo)
    while lo * ratio <= g.kcut:
        hi = lo * ratio
        sel = (k >= lo) & (k < hi) & (amp > floor)
        if sel.sum() >= 3:
            kc.append(np.sqrt(lo * hi))
            vc.append(np.sqrt((amp[sel] ** 2).mean()))
        lo = hi
    kc, vc = np.array(kc), np.array(vc)
    use = kc > 12
    if use.sum() < 4:
        return np.nan, np.nan
    x, y = np.log(kc[use]), np.log(vc[use])
    m, c0 = np.polyfit(x, y, 1)
    r = y - (m * x + c0)
    ss = ((y - y.mean()) ** 2).sum()
    return -m, (1.0 - (r ** 2).sum() / ss if ss > 0 else np.nan)


print()
print(f"  {'a':>6}  {'y*':>10}  {'c':>10}  {'beta predicted':>15}  "
      f"{'beta measured':>14}  {'R^2':>8}  {'rel diff':>9}")
print("  " + "-" * 82)

for a in (0.72, 0.75, 0.78, 0.80, 0.82, 0.85):
    g, out = simulate(a, n=4096, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, a)
    if not ok or sup(g, f) < 1e-6:
        print(f"  {a:6.2f}  Newton failed or trivial")
        continue
    best = predicted_beta(g, f, a)
    if best is None:
        print(f"  {a:6.2f}  no singular stagnation point found")
        continue
    ys, c, mu = best
    bp = mu + 1.0
    bm, r2 = measured_beta(g, f)
    print(f"  {a:6.2f}  {ys:10.6f}  {c:10.6f}  {bp:15.6f}  {bm:14.6f}  "
          f"{r2:8.5f}  {abs(bp - bm) / bp:9.3f}")

print()
print("  If the predicted column tracks the measured one as both move, the")
print("  formula holds and beta is a function of a rather than a constant.")
print("  Agreement only at a = 0.8 would mean the match there was luck.")
print()
print("  Caveat on the measured column: once beta is large the spectrum falls")
print("  below the roundoff floor within a few octaves, so there is little")
print("  range left to fit and the measurement degrades. Low R^2 there is the")
print("  measurement failing, not the prediction.")
