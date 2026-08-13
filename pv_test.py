"""
Retry the PV identity, off grid.

c2_form.py subtracted the two simple poles with cot and summed on the grid, and
got ratios of 1e-2 to 1, apparently a failure. But the reported scale of the
regularised integrand was 1e6 to 1e8, when it should be order 1. That is not a
failed identity, it is catastrophic cancellation: if a grid point sits very
close to a zero of U then 1/U and the cot term are each enormous and their
difference loses every digit.

The likely cause is that the stagnation points sit essentially on grid points.
The profile is rolled so that its extremum lands on index 0, and f is odd about
y1, so y1 has every reason to fall on a grid point or a half grid point rather
than somewhere generic.

Fix: evaluate the integrand on a grid shifted by half a cell. Shifting is exact
for a band limited field, multiply the coefficients by exp(i k delta), so
nothing is interpolated away. The trapezoid rule is still spectrally accurate
on the shifted points, and no point is near a pole.

The identity being tested, from integrating f'/f = (U' - 1)/(aU) once around
the circle with f and U both returning to themselves:

    PV integral of dy/U over the circle = 0

It needs no jump in ln|f| at either zero, which holds: at y1 the profile is
smooth with mu = 1, and at y2 the one sided constants satisfy C+ = -C-, so
|C+| = |C-| and the log does not jump.

    python pv_test.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def shift(g, field, delta):
    """Exact translation of a band limited field by delta."""
    return g.bwd(g.fwd(field) * np.exp(1j * g.k * delta))


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
        out.append((x % (2.0 * np.pi), evaluate(g, Hf, x)))
    return sorted(out, key=lambda r: (r[1] - 1.0) / (a * r[1]))


print()
print("=" * 74)
print("Part 1: where do the stagnation points sit relative to the grid?")
print("=" * 74)
print()
print(f"  {'a':>6}  {'N':>6}  {'dist(y1, grid)/dx':>19}  "
      f"{'dist(y2, grid)/dx':>19}")
print("  " + "-" * 56)

cases = {}
for a in (0.75, 0.78, 0.80, 0.82):
    g, out = simulate(a, n=1024, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, a)
    if not ok:
        continue
    z = zeros_of_U(g, f, a)
    cases[a] = (g, f, z)
    d = []
    for ys, _ in z:
        j = int(np.round(ys / g.dx))
        d.append(abs(ys - j * g.dx) / g.dx)
    print(f"  {a:6.2f}  {g.N:6d}  {d[0]:19.3e}  {d[1]:19.3e}")

print()
print("  A number near zero means the zero lands on a grid point and the")
print("  on grid subtraction was doomed.")


print()
print("=" * 74)
print("Part 2: the PV integral, evaluated on a half shifted grid")
print("=" * 74)
print()
print(f"  {'a':>6}  {'shift':>8}  {'PV':>15}  {'scale':>11}  {'ratio':>11}")
print("  " + "-" * 56)

for a, (g, f, z) in cases.items():
    U, _, _ = dg.fields(g, f)
    (y1, c1), (y2, c2) = z
    for frac in (0.5, 0.25, 0.5001):
        delta = frac * g.dx
        xs = g.x + delta
        Us = shift(g, U, delta)
        reg = (1.0 / Us
               - 0.5 / c1 / np.tan((xs - y1) / 2.0)
               - 0.5 / c2 / np.tan((xs - y2) / 2.0))
        pv = float(reg.sum() * g.dx)
        scale = float(np.abs(reg).sum() * g.dx)
        print(f"  {a:6.2f}  {frac:8.4f}  {pv:15.3e}  {scale:11.4f}  "
              f"{abs(pv) / scale:11.2e}")
    print()

print("  A scale of order 1 says the subtraction worked this time. A ratio")
print("  near machine zero then says PV of the integral of dy/U vanishes,")
print("  which is an exact constraint linking U to its own zeros.")
