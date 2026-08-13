"""
Two loose ends from generic.py.

The high mode datum, sin 3x + 0.4 sin 4x, returned c1 = c2 = 5.0069, that is
mu = 1 at both listed stagnation points, and beta = 2. That is not a different
profile. It is the same profile under a symmetry that was not noticed before.

If f solves rhs(f) = f then so does g(x) = f(nx) for any positive integer n.
The Hilbert transform commutes with integer dilation on the circle, so
Hg(x) = (Hf)(nx); the velocity picks up a compensating factor, U_g(x) =
U_f(nx)/n, since U_g' = Hg; and g'(x) = n f'(nx). The two factors of n cancel
in the advection term:

    rhs(g) = f(nx) (Hf)(nx) - a (U_f(nx)/n)(n f'(nx)) = rhs(f)(nx) = g(x)

So profiles come in families, and a datum built from mode 3 lands on the third
member rather than on something new. The n-th member has 2n stagnation points,
which is why listing only the first two by exponent gave two simple zeros and
missed the singular ones.

This also explains why generic.py reported the amplitude spread as 3e-3: the
n = 3 copy needs three times the resolution to represent equally well, so its
constants carry three times the discretisation error, not a different value.

Second loose end: the one signed datum 1 - cos x did not blow up at all,
reaching only |w| = 2.4 by t = 120. Lei, Liu and Ren proved global regularity
for one signed data at a = 1. This suggests that result, or its mechanism,
reaches down to a = 0.8.

    python dilation.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, residual, simulate, sup

A = 0.8


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def all_stagnation(g, f, a):
    """Every zero of U, with its exponent. Not just the first two."""
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
            s = evaluate(g, U, x) / du
            x -= s
            if abs(s) < 1e-15:
                break
        c = evaluate(g, Hf, x)
        out.append((x % (2 * np.pi), c, (c - 1.0) / (a * c)))
    return sorted(out)


print()
print("=" * 70)
print("Part 1: is f(nx) a solution whenever f is?")
print("=" * 70)

g = dg.Grid(2048)
_, out = simulate(A, n=2048, w_max=1e6)
f0, _ = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, A)
print(f"  base profile at N = 2048: residual {hist[-1]:.2e}, "
      f"||f||_inf = {sup(g, f):.7f}")
print()
print(f"  {'n':>3}  {'grid':>7}  {'residual of f(nx)':>19}  {'||f(nx)||_inf':>14}")
print("  " + "-" * 50)
for n in (1, 2, 3, 4):
    gn = dg.Grid(2048 * n)
    fn = np.interp(np.mod(n * gn.x, 2 * np.pi), g.x, f, period=2 * np.pi)
    # Rebuild spectrally rather than by interpolation error: sample the
    # base profile's series at nx exactly.
    fh = g.fwd(f)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    fn = np.real(np.exp(1j * np.outer(n * gn.x, g.k)) @ c)
    print(f"  {n:3d}  {gn.N:7d}  {np.abs(residual(gn, fn, A)).max():19.3e}  "
          f"{sup(gn, fn):14.7f}")

print()
print("  A residual at the level of the base profile's own means f(nx) solves")
print("  the equation whenever f does, and the amplitude is unchanged.")

print()
print("=" * 70)
print("Part 2: the high mode datum lands on the n = 3 member")
print("=" * 70)

gh = dg.Grid(2048)
w0 = np.sin(3.0 * gh.x) + 0.4 * np.sin(4.0 * gh.x)
outh = dg.run(w0, a=A, grid=gh, t_max=120.0, cfl=0.02, tail_tol=1e-9,
              w_max=1e6)
fh0, _ = guess_from_simulation(gh, outh)
fh1, hh, okh = newton(gh, fh0, A)
st = all_stagnation(gh, fh1, A)
print(f"  converged {okh}, ||f||_inf = {sup(gh, fh1):.7f}, "
      f"{len(st)} stagnation points")
print()
print(f"  {'y*':>10}  {'c':>12}  {'mu':>10}")
print("  " + "-" * 36)
for y, c, mu in st:
    print(f"  {y:10.6f}  {c:12.6f}  {mu:10.6f}")
print()
mus = sorted(m for _, _, m in st)
print(f"  exponents present: "
      f"{', '.join(f'{m:.4f}' for m in mus)}")
print(f"  the base profile has 2 stagnation points with mu = 1 and mu = 2;")
print(f"  the n = 3 member should have 6, three of each.")

print()
print("=" * 70)
print("Part 3: does one signed data blow up at a = 0.8?")
print("=" * 70)
print(f"  {'a':>6}  {'max |w| reached':>16}  {'t':>9}  {'stop':>16}")
print("  " + "-" * 54)
for a in (0.5, 0.6, 0.7, 0.8, 0.9):
    gg = dg.Grid(1024)
    o = dg.run(1.0 - np.cos(gg.x), a=a, grid=gg, t_max=200.0, cfl=0.02,
               tail_tol=1e-9, w_max=1e6)
    print(f"  {a:6.2f}  {o['winf_hist'][-1]:16.4e}  {o['t']:9.2f}  "
          f"{o['reason']:>16}")
print()
print("  Lei, Liu and Ren proved global regularity for one signed data at")
print("  a = 1. Bounded amplitudes here would indicate that this reaches")
print("  well below a = 1, which the frozen profile picture does not predict.")
