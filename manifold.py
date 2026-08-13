"""
Why does mode truncation converge so slowly, and what sets the escape time?

Near the line of equilibria, write omega = A sin x + v with v orthogonal to
sin x. Since R is quadratic, DR_1(A sin) = A M with M = DR_1(sin x) fixed, so

    v_t = A M v - (eps/2) A^2 sin 2x,     A' = -(3/4) A b_2

with b_2 the sin 2x component of v. If M were strictly stable the response
would saturate at v ~ eps A M^{-1} sin 2x, giving

    A' = kappa eps A^2,   kappa = -(3/8)(M^{-1})_{22}

a Riccati equation with A blowing up at t = 1/(kappa eps A_0), hence p = 1.
The measured exponent is below 1, so that step fails, and it fails precisely
when M has spectrum accumulating at zero: then the response never saturates,
slow modes keep absorbing the forcing, and the effective kappa drifts.

That also explains the truncation behaviour. Cutting at m modes discards the
slowest part of the spectrum, which is exactly the part that matters, and the
gap to the full answer closes like a power of m rather than exponentially:
measured gaps at eps = 0.1 are 2.207, 1.898, 1.632, 1.447, 1.204, 0.848 for
m = 2, 4, 6, 8, 12, 24, which is close to m^(-1/2).

So compute the spectrum of M and look at how it approaches zero.

    python manifold.py
"""

import numpy as np

import dg


def dR(g, f, a):
    """Jacobian of R at f, without the -I of the profile equation."""
    u_f, ux_f, fx_f = dg.fields(g, f)
    J = np.empty((g.N, g.N))
    e = np.zeros(g.N)
    for j in range(g.N):
        e[j] = 1.0
        u_d, ux_d, dx_d = dg.fields(g, e)
        J[:, j] = g.bwd(g.fwd(e * ux_f + f * ux_d
                              - a * (u_d * fx_f + u_f * dx_d)))
        e[j] = 0.0
    return J


print()
print("=" * 72)
print("Spectrum of M = DR_1(sin x)")
print("=" * 72)

for n in (128, 256, 512):
    g = dg.Grid(n)
    M = dR(g, np.sin(g.x), 1.0)
    vals = np.linalg.eigvals(M)
    vals = vals[np.argsort(-vals.real)]
    near = vals[np.abs(vals) < 5.0]
    near = near[np.argsort(-near.real)]
    print()
    print(f"  N = {n}: {len(vals)} eigenvalues, "
          f"{int((vals.real > 1e-8).sum())} with positive real part")
    print(f"  leading 12 by real part:")
    for v in vals[:12]:
        print(f"    {v.real:12.7f}  {v.imag:+12.7f} i")

g = dg.Grid(512)
M = dR(g, np.sin(g.x), 1.0)
vals = np.linalg.eigvals(M)
real = np.sort(vals.real)

print()
print("=" * 72)
print("How does the spectrum approach zero?")
print("=" * 72)
neg = np.sort(-vals.real[vals.real < -1e-10])
print(f"  {len(neg)} eigenvalues with negative real part")
print()
print(f"  {'rank j':>8}  {'-Re lambda':>13}  {'ratio to 1/j':>14}  "
      f"{'ratio to 1/j^2':>15}")
print("  " + "-" * 56)
for j in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89):
    if j <= len(neg):
        v = neg[j - 1]
        print(f"  {j:8d}  {v:13.8f}  {v * j:14.6f}  {v * j * j:15.6f}")

print()
print("  A column that holds steady identifies the law. Constant in the")
print("  1/j column means the spectrum accumulates at zero like 1/j, which")
print("  is slow enough that no finite truncation captures the tail and the")
print("  quasi steady reduction is invalid.")

print()
print("=" * 72)
print("Is there an exactly neutral subspace?")
print("=" * 72)
tiny = vals[np.abs(vals) < 1e-8]
print(f"  {len(tiny)} eigenvalues within 1e-8 of zero")
sinx = np.sin(g.x)
cosx = np.cos(g.x)
print(f"  M applied to sin x : max |M sin x|  = {np.abs(M @ sinx).max():.3e}")
print(f"  M applied to cos x : max |M cos x|  = {np.abs(M @ cosx).max():.3e}")
print()
print("  sin x spans the amplitude direction along the line of equilibria and")
print("  cos x the translation direction, so both should be annihilated.")
