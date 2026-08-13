"""
Solve for c2 using c1 as a control variate.

c2_solve.py refined the profile at seven resolutions and found c2 bouncing
between -1.6486 and -1.6681 with no monotone trend, so a direct extrapolation
in N returned nonsense: exponent 3.3, amplitude 1.1e7, residual 4.2e-3. The
same run also showed c1 wandering between 4.9870 and 5.0069 when its exact
value is 1/(1-a) = 5.

That second fact is the lever. Both c1 and c2 are read off the same profile and
carry the same discretisation error, and one of them has a known answer. If the
error enters both linearly then c2 is a linear function of c1 across
resolutions, and evaluating that line at the exact c1 removes the error without
needing to know its size or its rate.

The consecutive slopes dc2/dc1 from that run were -0.976, -0.981, -0.981,
-0.982, -0.982, -0.982, which is about as linear as measured data gets. This
script does it properly: regress, evaluate at c1 = 1/(1-a), and check that the
corrected values from every resolution collapse onto one number.

The point of the exercise is that mu2 = 2 exactly, hence beta = 3, has never
been excluded. The raw scatter of 5e-3 was larger than the 2e-3 it would take
to decide.

    python c2_control.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

NS = (512, 768, 1024, 1536, 2048, 3072, 4096)


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def zeros_of_U(g, f, a):
    U, Hf, _ = dg.fields(g, f)
    out = []
    for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
        x0, x1 = g.x[i], g.x[(i + 1) % g.N]
        if x1 < x0:
            x1 += 2.0 * np.pi
        u0, u1 = U[i], U[(i + 1) % g.N]
        x = x0 + (x1 - x0) * u0 / (u0 - u1)
        for _ in range(60):
            du = evaluate(g, Hf, x)
            if du == 0.0:
                break
            step = evaluate(g, U, x) / du
            x -= step
            if abs(step) < 1e-15:
                break
        c = evaluate(g, Hf, x)
        out.append((x % (2.0 * np.pi), c, (c - 1.0) / (a * c)))
    return sorted(out, key=lambda r: r[2])


def collect(a, ns=NS):
    c1s, c2s = [], []
    for n in ns:
        g, out = simulate(a, n=n, w_max=1e6)
        f0, _ = guess_from_simulation(g, out)
        f, hist, ok = newton(g, f0, a)
        if not ok:
            continue
        z = zeros_of_U(g, f, a)
        if len(z) == 2:
            c1s.append(z[0][1])
            c2s.append(z[1][1])
    return np.array(c1s), np.array(c2s)


def corrected(a, ns=NS, verbose=True):
    c1, c2 = collect(a, ns)
    if len(c1) < 3:
        return np.nan, np.nan, np.nan
    exact = 1.0 / (1.0 - a)
    s, b = np.polyfit(c1, c2, 1)
    r = c2 - (s * c1 + b)
    ss = ((c2 - c2.mean()) ** 2).sum()
    r2 = 1.0 - (r ** 2).sum() / ss if ss > 0 else np.nan
    c2_corr = s * exact + b
    per_point = c2 - s * (c1 - exact)

    if verbose:
        print()
        print(f"  a = {a},  exact c1 = {exact:.8f}")
        print(f"  regression c2 = {s:.6f} c1 + {b:.6f},  R^2 = {r2:.8f}")
        print()
        print(f"  {'N':>6}  {'c1':>12}  {'c1 - exact':>12}  {'c2 raw':>13}  "
              f"{'c2 corrected':>14}")
        print("  " + "-" * 66)
        for n, a1, a2, cc in zip(ns, c1, c2, per_point):
            print(f"  {n:6d}  {a1:12.8f}  {a1 - exact:+12.2e}  {a2:13.8f}  "
                  f"{cc:14.8f}")
        print()
        print(f"  raw c2 spread:       {c2.max() - c2.min():.3e}")
        print(f"  corrected spread:    {per_point.max() - per_point.min():.3e}")
        print(f"  improvement factor:  "
              f"{(c2.max() - c2.min()) / (per_point.max() - per_point.min()):.0f}x")
    return c2_corr, float(np.std(per_point)), r2


print()
print("=" * 78)
print("Part 1: the control variate correction at a = 0.8")
print("=" * 78)

c2c, c2sd, r2 = corrected(0.8)
A = 0.8
mu2 = (c2c - 1.0) / (A * c2c)
# Propagate the scatter through mu2 = (c-1)/(ac), dmu/dc = 1/(a c^2)
dmu = c2sd / (A * c2c ** 2)

print()
print("=" * 78)
print("Part 2: is mu2 = 2, and therefore beta = 3?")
print("=" * 78)
print()
print(f"  c2   = {c2c:.8f}  +/- {c2sd:.1e}")
print(f"  mu2  = {mu2:.8f}  +/- {dmu:.1e}")
print(f"  beta = {mu2 + 1.0:.8f}")
print()
print(f"  mu2 = 2 requires c2 = 1/(1-2a) = {1.0 / (1.0 - 2.0 * A):.8f}")
print(f"  measured c2 differs from it by "
      f"{c2c - 1.0 / (1.0 - 2.0 * A):+.3e}, which is "
      f"{abs(c2c - 1.0 / (1.0 - 2.0 * A)) / c2sd:.0f} times the scatter")
print()
print(f"  mu2 - 2 = {mu2 - 2.0:+.3e}, which is {abs(mu2 - 2.0) / dmu:.0f} "
      f"times the scatter")

print()
print("=" * 78)
print("Part 3: does the correction work at other a?")
print("=" * 78)
print("  The technique needs mu1 = 1 to be exact, which it is for any a, so it")
print("  should transfer. Fewer resolutions here to keep the runtime sane.")
print()
print(f"  {'a':>6}  {'c2 corrected':>14}  {'scatter':>10}  {'R^2':>10}  "
      f"{'mu2':>11}  {'beta':>11}")
print("  " + "-" * 70)
for a in (0.75, 0.78, 0.82):
    cc, sd, rr = corrected(a, ns=(1024, 1536, 2048, 3072, 4096), verbose=False)
    if not np.isfinite(cc):
        print(f"  {a:6.2f}  failed")
        continue
    m = (cc - 1.0) / (a * cc)
    print(f"  {a:6.2f}  {cc:14.8f}  {sd:10.2e}  {rr:10.7f}  {m:11.7f}  "
          f"{m + 1.0:11.7f}")
