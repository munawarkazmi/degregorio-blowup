"""
Is the frozen profile odd about a shifted point? If it is, finding 11 is empty.

Finding 11 shows the two stagnation points sit exactly `pi` apart and proves it
is forced, by splitting `U` into even and odd Fourier parts and observing that
both must vanish at `y1` separately. Chen, ARMA 241 (2021), constructs the
frozen profile on the circle for `a` near 1 and his profile is *odd*. For an
odd profile the statement is automatic: if `f` is odd about `x0` then `Hf` is
even about `x0`, so `U` is odd about `x0`, so `U(x0) = 0`, and periodicity
forces `U(x0 + pi) = 0` too. Two stagnation points, `pi` apart, for free.

Finding 11 already checked that `f` does not have only odd modes: at `a = 0.8`
the mode 2 amplitude is 3315 against 6520 for mode 1. But that is the wrong
test. "Only odd modes" is oddness about `x = 0` *and* a half period symmetry;
oddness about some shifted `x0` is much weaker and is what would make the
result trivial.

The exact test. Write `f = sum_k c_k e^{ikx}`. Then `f` is odd about `x0`
exactly when `g(s) = f(x0 + s)` is odd, i.e. when every `c_k e^{i k x0}` is
purely imaginary:

    R(x0) = || Re[c_k e^{i k x0}] || / || c_k ||

vanishes at some `x0`. `R` is scale free and needs no fitting. If `R` bottoms
out at machine precision, `f` is odd about that point and finding 11 restates
oddness. If it bottoms out at order one, the profile is genuinely not odd
about any point and the parity argument has content that Chen's setting cannot
supply.

    python parity_check.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate


def coeffs(g, field):
    """Complex Fourier coefficients c_k with field = Re sum_k c_k e^{ikx}."""
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return c


def oddness_residual(c, x0):
    """R(x0), the part of the spectrum that oddness about x0 forbids."""
    k = np.arange(len(c))
    z = c * np.exp(1j * k * x0)
    num = np.linalg.norm(np.real(z[1:]))
    den = np.linalg.norm(np.abs(c[1:]))
    return num / den


def best_x0(c, n_scan=200000):
    xs = np.linspace(0.0, 2.0 * np.pi, n_scan, endpoint=False)
    r = np.array([oddness_residual(c, x) for x in xs])
    i = int(np.argmin(r))
    lo, hi = xs[i - 1], xs[(i + 1) % n_scan]
    if hi < lo:
        hi += 2.0 * np.pi
    for _ in range(200):                      # golden-free bisection on slope
        m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
        if oddness_residual(c, m1) < oddness_residual(c, m2):
            hi = m2
        else:
            lo = m1
    x = 0.5 * (lo + hi)
    return x % (2.0 * np.pi), oddness_residual(c, x), r.min()


print(__doc__.split("    python")[0])
print("=" * 72)

g = dg.Grid(1024)

print("\n  Controls, to show the statistic does what it claims:\n")
print(f"  {'field':>34}  {'best x0':>10}  {'R(x0)':>12}")
print("  " + "-" * 60)
for name, field in (
    ("sin(x - 0.7), odd about 0.7", np.sin(g.x - 0.7)),
    ("sin s + 0.5 sin 2s + 0.25 sin 3s, s = x - 0.7",
     np.sin(g.x - 0.7) + 0.5 * np.sin(2 * (g.x - 0.7))
     + 0.25 * np.sin(3 * (g.x - 0.7))),
    ("sin x + 0.5 cos 2x, odd about nothing",
     np.sin(g.x) + 0.5 * np.cos(2.0 * g.x)),
):
    x0, r, rscan = best_x0(coeffs(g, field))
    print(f"  {name[:34]:>34}  {x0:10.6f}  {r:12.3e}")

print("\n  The frozen profile:\n")
print(f"  {'a':>6}  {'N':>6}  {'best x0':>11}  {'R(x0)':>12}  "
      f"{'|f_2|/|f_1|':>12}  {'mean':>10}")
print("  " + "-" * 68)
for a in (0.75, 0.78, 0.80, 0.82):
    gg, out = simulate(a, n=1024, w_max=1e6)
    f0, _ = guess_from_simulation(gg, out)
    f, _, ok = newton(gg, f0, a)
    if not ok:
        print(f"  {a:6.2f}  newton failed")
        continue
    c = coeffs(gg, f)
    x0, r, _ = best_x0(c)
    print(f"  {a:6.2f}  {1024:6d}  {x0:11.6f}  {r:12.3e}  "
          f"{abs(c[2]) / abs(c[1]):12.4f}  {f.mean():10.2e}")

    if a == 0.80:                     # locate the stagnation points for context
        U, _, _ = dg.fields(gg, f)
        zs = []
        for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
            x1, x2 = gg.x[i], gg.x[(i + 1) % gg.N]
            if x2 < x1:
                x2 += 2.0 * np.pi
            zs.append(x1 + (x2 - x1) * U[i] / (U[i] - U[(i + 1) % gg.N]))
        zs = sorted(z % (2.0 * np.pi) for z in zs)
        print()
        print(f"    stagnation points   {zs[0]:.6f}  {zs[1]:.6f}   "
              f"separation {zs[1] - zs[0]:.9f}")
        print(f"    best x0             {x0:.6f}")
        print(f"    x0 - y1             {(x0 - zs[0]) % np.pi:.6f}   "
              f"(0 or pi means x0 sits on a stagnation point)")

print("""
  Reading it: R at machine precision means f is odd about that point, and
  finding 11 is then a restatement of oddness, which Chen's construction
  assumes outright. R of order one means no such point exists and the parity
  argument says something his setting cannot.
""")
