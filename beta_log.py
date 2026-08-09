"""
The resonant case: is beta exactly 3, with a logarithm?

Three results now point the same way.

  Binned local slopes scatter around 3 with no drift (2.91 to 3.07 across k
  from 16 to 724), so the wide window fit's 3.045 was the beating bias.

  The stagnation point analysis gives mu = (c - 1) / (a c) = 2.0017 at
  y* = 4.1157, and mu = 1.0001 at the other zero of U.

  Both mu land on integers, which naively means f vanishes smoothly there and
  is not singular at all, which would leave the algebraic decay unexplained.

The resolution is that integer mu is precisely the resonant case. The profile
equation has a regular singular point at y*, and when the Frobenius exponents
differ by an integer the second solution carries a logarithm:

    f ~ C (y - y*)^mu + D (y - y*)^mu log|y - y*|

and |y|^m log|y| has Fourier coefficients decaying like k^-(m+1) exactly. So
mu = 2 with D nonzero gives beta = 3 exactly, no fitting required.

That is falsifiable in real space. Differentiating y^2 log y twice gives
2 log y + 3, so:

    f'' diverges logarithmically at y*, and f'' against log|y - y*| is a
    straight line whose slope is 2D.

At the mu = 1 point the same argument would put a log divergence in f' itself,
so max|f'| would be unbounded. It was measured at 7.04, finite and modest, so
D = 0 there and that point is smooth. Consistent with beta = 3 rather than 2.

Two checks here: does mu converge to an integer as N grows, and is f'' actually
logarithmic at y*.

    python beta_log.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8


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


def stagnation(g, f, a):
    U, Hf, _ = dg.fields(g, f)
    out = []
    for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
        ys = refine_zero(g, U, i) % (2.0 * np.pi)
        c = evaluate(g, Hf, ys)
        out.append((ys, c, (c - 1.0) / (a * c)))
    return sorted(out, key=lambda r: r[2])


print()
print("=" * 74)
print("Part 1: does mu converge to an integer as N grows?")
print("=" * 74)
print("  A deviation that shrinks with N means mu is exactly 1 and 2, and the")
print("  algebraic decay has to come from the resonant logarithm. A deviation")
print("  that holds steady means mu is genuinely non-integer.")
print()
print(f"  {'N':>7}  {'y* (first)':>11}  {'mu':>10}  {'mu - 1':>10}  "
      f"{'y* (second)':>12}  {'mu':>10}  {'mu - 2':>10}")
print("  " + "-" * 76)

profiles = {}
for n in (1024, 2048, 4096):
    g, out = simulate(A, n=n, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, A)
    if not ok:
        print(f"  {n:7d}  Newton failed")
        continue
    profiles[n] = (g, f)
    r = stagnation(g, f, A)
    print(f"  {n:7d}  {r[0][0]:11.6f}  {r[0][2]:10.6f}  {r[0][2] - 1:10.2e}  "
          f"{r[1][0]:12.6f}  {r[1][2]:10.6f}  {r[1][2] - 2:10.2e}")


print()
print("=" * 74)
print("Part 2: is f'' logarithmic at the mu = 2 point?")
print("=" * 74)
print("  If f ~ D (y-y*)^2 log|y-y*| then f'' = 2D log|y-y*| + const, so f''")
print("  plotted against log|y-y*| is a straight line. A finite, flat f'' would")
print("  mean no logarithm and no singularity there.")

g, f = profiles[4096]
r = stagnation(g, f, A)
ys = r[1][0]
fh = g.fwd(f)
fxx = g.bwd(fh * g.dmul * g.dmul)

print()
print(f"  y* = {ys:.6f},  mu = {r[1][2]:.6f}")
print()
print(f"  {'y - y*':>12}  {'log|y-y*|':>11}  {'f''''(y)':>12}   "
      f"{'f''''(y*-d)':>12}")
print("  " + "-" * 54)

j = int(np.argmin(np.abs(g.x - ys)))
xs, ys_r, ys_l = [], [], []
for d in (4, 6, 9, 13, 19, 28, 41, 60, 88, 129):
    ip, im = (j + d) % g.N, (j - d) % g.N
    dy = d * g.dx
    xs.append(np.log(dy))
    ys_r.append(fxx[ip])
    ys_l.append(fxx[im])
    print(f"  {dy:12.6f}  {np.log(dy):11.5f}  {fxx[ip]:12.5f}   "
          f"{fxx[im]:12.5f}")

for name, vals in (("right", ys_r), ("left", ys_l)):
    m, c = np.polyfit(xs, vals, 1)
    resid = np.array(vals) - (m * np.array(xs) + c)
    ss = ((np.array(vals) - np.mean(vals)) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss if ss > 0 else np.nan
    print()
    print(f"  {name} side:  f'' = {m:.5f} log|y-y*| + {c:.5f},  "
          f"R^2 = {r2:.6f}")
    print(f"             so D = {m / 2:.5f}")

print()
print("  A high R^2 with a nonzero slope is the logarithm, and therefore")
print("  beta = 3 exactly. Both sides should give the same slope, since the")
print("  log term is even in (y - y*).")
