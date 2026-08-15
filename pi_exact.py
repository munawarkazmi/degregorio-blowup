"""
T(1/2) = pi is exact. Here is the proof, and here is the check.

Finding 3 recorded T(1/2) = pi to eleven digits with no derivation, and left
open whether it was exact or a coincidence. It is exact, and it follows from
the exact pole dynamics solutions of Silantyev, Lushnikov, Siegel and Ambrose,
"Exact periodic solutions of the generalized Constantin-Lax-Majda equation
with dissipation", arXiv:2411.01891, Studies in Applied Mathematics 2025.
That paper is on the circle, which is our setting, and it solves exactly the
two cases a = 0 and a = 1/2. Equation numbers below are theirs.

Their ansatz uses X = tan(x/2), which maps (-pi, pi) to the real line. At
a = 1/2 the advection term makes logarithms out of simple poles, so the
invariant class is a double pole plus a simple pole, equation (41), tied by

    w_1 = 2 i v_c w_2 / (1 - v_c^2)                                       (42)

which is what cancels the logarithms. Restricting to real v_c and purely
imaginary w_2 = i w_2i, equation (56), gives the real ODE system (57), a
conserved quantity (58), and a separable equation (59) for v_c alone.

Our initial datum is in this class, at the one point where the parametrisation
degenerates. Partial fractions give

    sin x = 2X / (1 + X^2) = 1/(X - i) + 1/(X + i)

so sin x is a single conjugate pole pair at v_c = 1, w_1 = 1, w_2 = 0. That is
consistent with their remark that v_c -> 1 sends the complex singularity to
infinite height, which is what "entire" means. But (42) at v_c = 1 reads
1 = (2i/0) * 0, so sin x is reached only as a limit.

The limit is harmless. Solving (42) for our datum gives

    w_2i(0) = (v_c^2(0) - 1) / (2 v_c(0))

and substituting that into the coefficient of (59) cancels the vanishing factor
against the vanishing amplitude:

    K = w_2i(0) v_c(0) / [(v_c^2(0) - 1)(v_c^2(0) + 1)^2] = 1 / [2 (v_c^2(0) + 1)^2]

which at v_c(0) = 1 is K = 1/8, finite. So the flow through our datum is
regular, and (59) becomes

    dv_c/dt = (v_c^2 + 1)^3 / (32 v_c^2),    v_c(0) = 1.

Their antiderivative is G(x) = x(x^2 - 1)/(x^2 + 1)^2 + arctan x, which
satisfies G'(x) = 8 x^2 / (x^2 + 1)^3, so the implicit solution (60) is

    G(v_c(t)) = G(1) + t/4.

K > 0, so v_c increases. Blowup is v_c -> infinity, their type B, the complex
singularity reaching the real axis at x = +-pi rather than at 0. Since
G(1) = pi/4 and G(infinity) = pi/2,

    T = 4 [G(infinity) - G(1)] = 4 [pi/2 - pi/4] = pi.

The same construction at a = 0 gives v_c(t) = (2 + t)/(2 - t) and T = 2, which
is CLM and is how the machinery is checked below. Note the (2 + t) that finding
3 blamed for biasing the a = 0 fit: it is this pole trajectory.

Solving (42) and (58) for the amplitudes along our trajectory gives the full
closed form, with v = v_c(t) and X = tan(x/2):

    w_1(t)  = (v^2 + 1)^2 / 4
    w_2i(t) = (v^2 - 1)(v^2 + 1)^2 / (8 v)
    w(x, t) = 2 w_1 X / (X^2 + v^2)  -  4 w_2i v X / (X^2 + v^2)^2

which at t = 0 collapses to sin x.

    python pi_exact.py
"""

import numpy as np

import dg

TWO_PI = 2.0 * np.pi


def G(v):
    """Antiderivative with G'(v) = 8 v^2 / (v^2 + 1)^3, from (60)."""
    return v * (v * v - 1.0) / (v * v + 1.0) ** 2 + np.arctan(v)


def v_of_t(t, a_half=True):
    """
    Invert G(v) = G(1) + t/4 for v_c(t), by bisection on u = v/(1+v).

    G is strictly increasing, so the inversion is unconditional. Working in u
    keeps the bracket finite as v runs off to infinity at t = T.
    """
    target = G(1.0) + 0.25 * t
    lo, hi = 0.5, 1.0 - 1e-16          # u = 1/2 is v = 1, u -> 1 is v -> inf
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        v = mid / (1.0 - mid)
        if G(v) < target:
            lo = mid
        else:
            hi = mid
    u = 0.5 * (lo + hi)
    return u / (1.0 - u)


def exact_w(x, t):
    """The a = 1/2 solution from w_0 = sin x, in closed form."""
    v = v_of_t(t)
    w1 = (v * v + 1.0) ** 2 / 4.0
    w2i = (v * v - 1.0) * (v * v + 1.0) ** 2 / (8.0 * v)
    X = np.tan(0.5 * x)
    d = X * X + v * v
    return 2.0 * w1 * X / d - 4.0 * w2i * v * X / (d * d)


def exact_w_a0(x, t):
    """
    The a = 0 solution from w_0 = sin x, for the control.

    Same ansatz with the double pole switched off. The nu -> 0 limit of (35)
    gives v_c = (2 + t)/(2 - t) and w_1 = 4/(2 - t)^2, which is (v + 1)^2/4,
    the a = 0 counterpart of the (v^2 + 1)^2/4 above. Reducing the result by
    hand gives 4 sin x / (4 + 4 t cos x + t^2), the Constantin-Lax-Majda
    solution, which is asserted below rather than taken on trust.
    """
    v = (2.0 + t) / (2.0 - t)
    X = np.tan(0.5 * x)
    return 2.0 * ((v + 1.0) ** 2 / 4.0) * X / (X * X + v * v)


print(__doc__.split("    python")[0])
print("=" * 70)

# ---------------------------------------------------------------- conventions
g = dg.Grid(256)
err = np.abs(g.hilbert(np.sin(g.x)) + np.cos(g.x)).max()
print(f"\n  H(sin x) = -cos x, as in (4)            error {err:.2e}")
assert err < 1e-12

err = np.abs(exact_w(g.x, 0.0) - np.sin(g.x)).max()
print(f"  closed form at t = 0 equals sin x       error {err:.2e}")
assert err < 1e-12

for tt in (0.0, 0.7, 1.4):
    clm = 4.0 * np.sin(g.x) / (4.0 + 4.0 * tt * np.cos(g.x) + tt * tt)
    err = np.abs(exact_w_a0(g.x, tt) - clm).max()
    print(f"  a = 0 pole form is CLM at t = {tt:4.1f}      error {err:.2e}")
    assert err < 1e-12

# ------------------------------------------------------------- the blowup time
print(f"\n  G(1)                {G(1.0):.15f}   pi/4 = {np.pi / 4:.15f}")
print(f"  G(inf)              {G(1e12):.15f}   pi/2 = {np.pi / 2:.15f}")
T_exact = 4.0 * (0.5 * np.pi - G(1.0))
print(f"  T = 4[G(inf) - G(1)]  {T_exact:.15f}   pi   = {np.pi:.15f}")
print(f"  difference from pi                      {abs(T_exact - np.pi):.2e}")

# The same statement, reached by marching the ODE instead of integrating it.
def dv(v):
    return (v * v + 1.0) ** 3 / (32.0 * v * v)


v, t = 1.0, 0.0
while v < 20.0:
    h = min(1e-4, 1.6 / v ** 3)        # dv/dt ~ v^4/32, so h ~ 32/v^3 is O(1)
    k1, k2 = dv(v), dv(v + 0.5 * h * dv(v))
    k3 = dv(v + 0.5 * h * k2)
    k4 = dv(v + h * k3)
    v += (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    t += h
# v -> inf takes the remaining time 4[G(inf) - G(v)], known in closed form
t += 4.0 * (0.5 * np.pi - G(v))
print(f"  RK4 on dv/dt, tail closed              T = {t:.12f}")

# -------------------------------------------- the closed form against the PDE
print("\n  Closed form against the solver, w_0 = sin x, a = 1/2:")
print(f"\n  {'t':>6}  {'N':>6}  {'max |w_pde - w_exact|':>22}  {'rel':>10}")
print("  " + "-" * 50)
for t_stop in (0.5, 1.0, 2.0, 2.5):
    for N in (2048, 8192):
        gg = dg.Grid(N)
        out = dg.run(np.sin(gg.x), a=0.5, grid=gg, t_max=t_stop, cfl=0.01,
                     tail_tol=1e-9, w_max=1e12)
        we = exact_w(gg.x, out["t"])
        e = np.abs(out["w"] - we).max()
        print(f"  {out['t']:6.2f}  {N:6d}  {e:22.3e}  "
              f"{e / np.abs(we).max():10.2e}")

print("\n  Control, the same test at a = 0 where CLM is known:")
print(f"\n  {'t':>6}  {'N':>6}  {'max |w_pde - w_exact|':>22}  {'rel':>10}")
print("  " + "-" * 50)
for t_stop in (0.5, 1.0, 1.5):
    gg = dg.Grid(8192)
    out = dg.run(np.sin(gg.x), a=0.0, grid=gg, t_max=t_stop, cfl=0.01,
                 tail_tol=1e-9, w_max=1e12)
    we = exact_w_a0(gg.x, out["t"])
    e = np.abs(out["w"] - we).max()
    print(f"  {out['t']:6.2f}  {8192:6d}  {e:22.3e}  "
          f"{e / np.abs(we).max():10.2e}")

print()
print("  If the a = 1/2 column tracks the a = 0 control, the closed form is")
print("  the solution, and T(1/2) = pi is a theorem rather than a fit.")
print()
