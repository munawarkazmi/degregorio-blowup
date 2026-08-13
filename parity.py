"""
The parity calculation, and what it actually predicts.

Hypothesis to be tested: the feedback of mode 2 into the growth of mode 1
vanishes at linear order by parity, leaving a quadratic term and hence
T ~ eps^(-2/3).

Carrying it out refutes that. With omega = A sin x + B sin 2x, and using
H(sin kx) = -cos kx and u = -(1/k) sin kx,

    omega H omega = -(A^2/2) sin 2x - AB sin 3x - (B^2/2) sin 4x
    u omega_x     = -(A^2/2) sin 2x - (5AB/4) sin 3x + (3AB/4) sin x
                    - (B^2/2) sin 4x

The sin x terms cancel in the first but not the second, so
R_a = omega H omega - a u omega_x has

    sin x  : -(3a/4) A B
    sin 2x : -(eps/2) A^2
    sin 3x : (5a/4 - 1) A B
    sin 4x : -(eps/2) B^2

The linear feedback is there, with coefficient -3a/4. So the two mode system is

    A' = -(3a/4) A B,      B' = -(eps/2) A^2

which integrates exactly. Writing L = log A gives L' = -(3/4)B at a = 1 and
L'' = (3 eps/8) e^{2L}, a Liouville equation. Multiplying by L' and using
L(0) = 0, L'(0) = 0 gives L'^2 = (3 eps/8)(e^{2L} - 1), so

    integral dL / sqrt(e^{2L} - 1) = arccos(e^{-L}) -> pi/2

and therefore

    T = (pi/2) sqrt(8 / (3 eps)) = 2.56510 eps^(-1/2)

So the reduction predicts p = 1/2, not the 2/3 that was guessed and not the 1
from the naive drift argument. The measured exponent is about 0.65, between
the two, which suggests the cascade beyond mode 2 is what carries it.

Three checks: verify the coefficients numerically, verify the Liouville
prediction against a two mode Galerkin run, and see whether the exponent moves
toward the measured value as more modes are retained.

    python parity.py
"""

import numpy as np

import dg

WMAX = 1.0e6


def galerkin_grid(n, m):
    """Grid whose mask keeps only wavenumbers up to m, an m mode truncation."""
    g = dg.Grid(n)
    g.mask = (g.k <= m).astype(float)
    g.mask[-1] = 0.0
    g.kcut = min(m, g.kcut)
    return g


print()
print("=" * 74)
print("Part 1: are the mode coefficients right?")
print("=" * 74)
print("  Project R_a(A sin x + B sin 2x) onto sin kx and compare.")

g = dg.Grid(256)
A, B, a = 0.7, -0.3, 0.85
w = A * np.sin(g.x) + B * np.sin(2.0 * g.x)
r = dg.rhs(g, w, a)
coef = -2.0 * np.imag(np.fft.rfft(r)) / g.N     # coefficient of sin kx

pred = {1: -(3.0 * a / 4.0) * A * B,
        2: -((1.0 - a) / 2.0) * A ** 2,
        3: (5.0 * a / 4.0 - 1.0) * A * B,
        4: -((1.0 - a) / 2.0) * B ** 2}
print()
print(f"  a = {a}, A = {A}, B = {B}")
print(f"  {'mode':>5}  {'measured':>13}  {'predicted':>13}  {'diff':>11}")
print("  " + "-" * 48)
for k in (1, 2, 3, 4, 5):
    p = pred.get(k, 0.0)
    print(f"  {k:5d}  {coef[k]:13.9f}  {p:13.9f}  {coef[k] - p:11.2e}")

print()
print("=" * 74)
print("Part 2: does the two mode system obey T = 2.56510 eps^(-1/2)?")
print("=" * 74)
print()
print(f"  {'eps':>7}  {'T (2 modes)':>13}  {'2.5651/sqrt(eps)':>17}  "
      f"{'ratio':>8}")
print("  " + "-" * 50)
for eps in (0.2, 0.1, 0.05, 0.02, 0.01):
    gm = galerkin_grid(512, 2)
    out = dg.run(np.sin(gm.x), a=1.0 - eps, grid=gm, t_max=2000.0, cfl=0.02,
                 tail_tol=np.inf, w_max=WMAX)
    T = out["t"]
    pr = 2.565100 / np.sqrt(eps)
    print(f"  {eps:7.3f}  {T:13.6f}  {pr:17.6f}  {T / pr:8.5f}")

print()
print("=" * 74)
print("Part 3: does the exponent move toward 0.65 as modes are added?")
print("=" * 74)
print()
hdr = f"  {'modes':>6}  " + "  ".join(f"{'eps=' + f'{e:g}':>11}"
                                      for e in (0.1, 0.05, 0.02))
print(hdr + f"  {'local p':>9}")
print("  " + "-" * (len(hdr) + 11))
for m in (2, 3, 4, 6, 8, 12, 24):
    n = max(256, 8 * m)
    Ts = []
    for eps in (0.1, 0.05, 0.02):
        gm = galerkin_grid(n, m)
        out = dg.run(np.sin(gm.x), a=1.0 - eps, grid=gm, t_max=3000.0,
                     cfl=0.02, tail_tol=np.inf, w_max=WMAX)
        Ts.append(out["t"] if out["reason"] == "amplitude cap" else np.nan)
    if np.isfinite(Ts).all():
        p = -(np.log(Ts[2]) - np.log(Ts[0])) / (np.log(0.02) - np.log(0.1))
        cells = "  ".join(f"{t:11.4f}" for t in Ts)
        print(f"  {m:6d}  {cells}  {p:9.4f}")
    else:
        cells = "  ".join(f"{t:11.4f}" if np.isfinite(t) else f"{'-':>11}"
                          for t in Ts)
        print(f"  {m:6d}  {cells}  {'-':>9}")

print()
print("  The full equation gives p about 0.65 over this range of eps. If the")
print("  truncated exponent climbs from 0.5 toward that as modes are added,")
print("  the cascade past mode 2 is what sets it and no two mode argument")
print("  can produce the right answer.")
