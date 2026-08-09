"""
Predict beta instead of fitting it.

The profile equation a U f' = f (H f - 1) rearranges to

    f' / f = (H f - 1) / (a U)

so f can only be singular where U vanishes, at a stagnation point of the
profile's own velocity. Two things follow immediately.

At such a point y*, setting U(y*) = 0 in the equation gives
0 = f(y*) (H f(y*) - 1), so generically f(y*) = 0 as well.

Near y*, expand U(y) = U'(y*)(y - y*) + ... and use U' = H f, so with

    c = H f(y*)

the equation becomes f'/f = mu / (y - y*) with

    mu = (c - 1) / (a c)

hence f ~ C |y - y*|^mu. A |y|^mu singularity has Fourier coefficients decaying
like k^-(mu+1), so

    beta = 1 + (c - 1) / (a c)

and beta = 3 requires exactly c = 1 / (1 - 2a), which at a = 0.8 means
H f(y*) = -5/3.

That is a sharp test. Measuring one number, H f at a stagnation point, predicts
the entire spectral decay rate, with no fitting window and no bins. If the
prediction lands on the fitted beta, the mechanism is confirmed and the value
of beta is settled by whatever c happens to be, rather than by numerology
about 3.

U has several zeros, since it is periodic with zero mean. The one that governs
the spectrum is the one with the smallest mu, because that is the least smooth
singularity and it dominates at large k.

    python beta_theory.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8
N = 4096


def refine_zero(g, U, i):
    """Linear interpolation of a sign change, then Newton on the interpolant."""
    x0, x1 = g.x[i], g.x[(i + 1) % g.N]
    if x1 < x0:
        x1 += 2.0 * np.pi
    u0, u1 = U[i], U[(i + 1) % g.N]
    return x0 + (x1 - x0) * u0 / (u0 - u1)


def evaluate(g, field, x):
    """Value of the spectral interpolant of `field` at an arbitrary point."""
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


g, out = simulate(A, n=N, w_max=1e6)
f0, _ = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, A)
print()
print(f"profile at a = {A}, N = {N}: converged {ok}, residual {hist[-1]:.2e}, "
      f"||f||_inf = {sup(g, f):.6f}")

U, Hf, fx = dg.fields(g, f)

sign_change = np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1)))
print()
print(f"U vanishes at {len(sign_change)} points. At each one:")
print()
print(f"  {'y*':>10}  {'c = Hf(y*)':>11}  {'f(y*)':>11}  {'mu':>10}  "
      f"{'beta = mu+1':>12}")
print("  " + "-" * 60)

rows = []
for i in sign_change:
    ys = refine_zero(g, U, i) % (2.0 * np.pi)
    c = evaluate(g, Hf, ys)
    fv = evaluate(g, f, ys)
    mu = (c - 1.0) / (A * c)
    rows.append((ys, c, fv, mu))
    print(f"  {ys:10.6f}  {c:11.6f}  {fv:11.3e}  {mu:10.6f}  {mu + 1.0:12.6f}")

rows.sort(key=lambda r: r[3])
ys, c, fv, mu = rows[0]
print()
print(f"Smallest mu is {mu:.6f} at y* = {ys:.6f}, so that singularity governs")
print(f"the spectrum and the prediction is")
print()
print(f"    beta = {mu + 1.0:.6f}")
print()
print(f"For comparison, beta = 3 would require c = 1 / (1 - 2a) = "
      f"{1.0 / (1.0 - 2.0 * A):.6f},")
print(f"and the measured c is {c:.6f}.")
print()
print(f"Equivalently, the value of a that would give beta = 3 with this same c")
print(f"is a = (c - 1) / (2c) = {(c - 1.0) / (2.0 * c):.6f}, against the "
      f"a = {A} used here.")

print()
print("Sanity: f should vanish at every stagnation point, since U(y*) = 0")
print("forces f(y*) (c - 1) = 0 in the equation itself.")
print(f"  largest |f(y*)| over the stagnation points: "
      f"{max(abs(r[2]) for r in rows):.3e}")
