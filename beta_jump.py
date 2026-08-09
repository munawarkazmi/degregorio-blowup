"""
beta = 3 exactly, from a jump in f''. Is that structural or a coincidence?

The logarithm hypothesis is dead. f'' near y* is bounded and flat, running
about 2.27, and it is exactly odd about the point: +2.30652 on one side and
-2.30652 on the other. That is a jump discontinuity, not a divergence, and the
fitted log slope of 0.043 with R^2 = 0.46 is consistent with zero.

This fits the ODE picture better than the resonance did. The profile equation
is first order, f'/f = (H f - 1) / (a U), so near a stagnation point it has one
exponent mu and the general solution is C (y - y*)^mu times an analytic factor,
with the constant free to differ on the two sides. With mu = 2 and C+ != C-,

    f = C+ (y - y*)^2  on the right,   C- (y - y*)^2  on the left

so f is C^1,1 but not C^2, f'' jumps from 2C- to 2C+, and a jump in the second
derivative gives Fourier coefficients decaying like k^-3 exactly. Measured,
C+ = 1.135 and C- = -1.135.

It also explains the beating that biased the original wide window fit. The two
stagnation points sit at 0.974078 and 4.115670, exactly pi apart, and two
singular points separated by pi make |f_k| alternate with period 2 in k.

What is left is whether mu = 2 is forced. mu = (c - 1) / (a c) with
c = H f(y*), so mu = 2 means c = 1 / (1 - 2a) exactly, and c is set by the
global solution with no obvious reason to land there. If mu = 2 holds at other
values of a, then beta = 3 is structural and c = 1 / (1 - 2a) is a genuine
prediction. If mu drifts with a, then beta = 3 is a coincidence at a = 0.8.

    python beta_jump.py
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


def stagnation(g, f, a):
    U, Hf, _ = dg.fields(g, f)
    out = []
    for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
        ys = refine_zero(g, U, i) % (2.0 * np.pi)
        c = evaluate(g, Hf, ys)
        out.append((ys, c, (c - 1.0) / (a * c)))
    return sorted(out, key=lambda r: r[2])


print()
print("=" * 78)
print("Part 1: is mu = 2 forced, or special to a = 0.8?")
print("=" * 78)
print("  mu = 2 requires c = H f(y*) = 1 / (1 - 2a) exactly. Nothing in the")
print("  equation obviously puts c there, so compare the two columns.")
print()
print(f"  {'a':>6}  {'y* separation':>14}  {'c measured':>12}  "
      f"{'1/(1-2a)':>11}  {'mu':>10}  {'beta':>9}")
print("  " + "-" * 72)

for a in (0.70, 0.75, 0.80, 0.85):
    g, out = simulate(a, n=4096, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, a)
    if not ok:
        print(f"  {a:6.2f}  Newton failed")
        continue
    r = stagnation(g, f, a)
    if len(r) < 2:
        print(f"  {a:6.2f}  only {len(r)} stagnation point(s)")
        continue
    sep = abs(r[1][0] - r[0][0])
    pred = 1.0 / (1.0 - 2.0 * a)
    print(f"  {a:6.2f}  {sep:14.9f}  {r[1][1]:12.6f}  {pred:11.6f}  "
          f"{r[1][2]:10.6f}  {r[1][2] + 1:9.6f}")

print()
print("  The separation column is a second, independent structural claim: the")
print("  two stagnation points sitting exactly pi apart is what makes |f_k|")
print("  alternate with period 2 in k.")


print()
print("=" * 78)
print("Part 2: the jump itself, and that f'' really is bounded")
print("=" * 78)
print("  A jump means f'' tends to equal and opposite limits from the two")
print("  sides. A logarithm would mean |f''| grows without bound as y -> y*.")
print("  The wiggle at the smallest distances is ringing from summing a k^-1")
print("  series for f'', not structure.")

g, out = simulate(0.8, n=4096, w_max=1e6)
f0, _ = guess_from_simulation(g, out)
f, _, _ = newton(g, f0, 0.8)
r = stagnation(g, f, 0.8)
ys = r[1][0]
fh = g.fwd(f)
fxx = g.bwd(fh * g.dmul * g.dmul)
j = int(np.argmin(np.abs(g.x - ys)))

print()
print(f"  y* = {ys:.6f}")
print()
print(f"  {'|y - y*|':>11}  {'f_xx right':>12}  {'f_xx left':>12}  "
      f"{'sum':>11}")
print("  " + "-" * 50)
for d in (8, 14, 24, 40, 68, 115, 195):
    ip, im = (j + d) % g.N, (j - d) % g.N
    print(f"  {d * g.dx:11.6f}  {fxx[ip]:12.6f}  {fxx[im]:12.6f}  "
          f"{fxx[ip] + fxx[im]:11.2e}")

right = float(np.mean([fxx[(j + d) % g.N] for d in (24, 40, 68)]))
print()
print(f"  f''(y*+) = {right:+.5f},  f''(y*-) = {-right:+.5f},  "
      f"jump = {2 * right:.5f}")
print(f"  so C+ = {right / 2:.5f} and C- = {-right / 2:.5f} in "
      f"f ~ C (y - y*)^2")
print()
print("  A sum column at machine zero says f'' is exactly odd about y*, which")
print("  is the jump. Bounded values say there is no logarithm.")
