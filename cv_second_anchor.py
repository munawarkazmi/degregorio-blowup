"""
Does the control variate survive a second, independent anchor?

Finding 12 corrects `c2` by regressing it against `c1`, whose exact value
1/(1-a) is known, and evaluating at that value. One anchor proves nothing about
the method: if the correction were an artefact of `c1` specifically, the answer
would still look self consistent across resolutions. A second anchor with an
independently known exact value is the test.

The profile carries one: finding 13's constraint

    P := PV oint dy/U = 0

is exact, computed from the same discrete profile, and has nothing to do with
`c1`. If regressing `c2` against `P` and evaluating at `P = 0` reproduces the
`c1` anchored value of `c2`, the method is doing what it claims. If the two
disagree, it is not.

**But a control variate needs its anchor to carry the same discretisation error
as the target.** So the first question is not whether the second anchor agrees,
it is whether `P` varies with `N` at all. If `P` sits at machine precision for
every resolution then it carries no error signal, cannot serve as an anchor,
and its agreement with zero is telling us nothing about the accuracy of the
profile. Part 1 settles that before Part 2 is attempted.

Part 3 asks why, by testing whether `1/U` is odd about `y1`. Finding 19
established that `f` is odd about `y1` to 1e-13, hence `Hf` is even about it,
hence `U` is odd about it. An odd integrand integrates to zero on the circle by
antisymmetry alone, with no reference to the profile equation.

    python cv_second_anchor.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate

NS = (512, 768, 1024, 1536, 2048, 3072)


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def shift(g, field, delta):
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


def pv_and_scale(g, f, z):
    """The regularised PV integral on a half shifted grid, and its scale."""
    U, _, _ = dg.fields(g, f)
    (y1, c1), (y2, c2) = z
    delta = 0.5 * g.dx
    xs = g.x + delta
    Us = shift(g, U, delta)
    reg = (1.0 / Us
           - 0.5 / c1 / np.tan((xs - y1) / 2.0)
           - 0.5 / c2 / np.tan((xs - y2) / 2.0))
    return float(reg.sum() * g.dx), float(np.abs(reg).sum() * g.dx)


def solve(a, n):
    g, out = simulate(a, n=n, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, _, ok = newton(g, f0, a)
    if not ok:
        return None
    z = zeros_of_U(g, f, a)
    if len(z) != 2:
        return None
    pv, scale = pv_and_scale(g, f, z)
    return g, f, z, pv, scale


print(__doc__.split("    python")[0])
print("=" * 76)

print("\n  Part 1: does P carry a discretisation error signal?\n")
print(f"  {'a':>6}  {'N':>6}  {'c1 - exact':>12}  {'c2':>13}  "
      f"{'P':>12}  {'P/scale':>10}")
print("  " + "-" * 68)

data = {}
for a in (0.75, 0.80):
    exact = 1.0 / (1.0 - a)
    rows = []
    for n in NS:
        r = solve(a, n)
        if r is None:
            continue
        g, f, z, pv, scale = r
        rows.append((n, z[0][1], z[1][1], pv, scale))
        print(f"  {a:6.2f}  {n:6d}  {z[0][1] - exact:+12.2e}  {z[1][1]:13.8f}  "
              f"{pv:12.2e}  {abs(pv) / scale:10.1e}")
    data[a] = rows
    print()

print("  If the P column wanders in step with the c1 column, it is an anchor.")
print("  If it sits at machine precision throughout, it is not.\n")

print("=" * 76)
print("\n  Part 2: the second anchored correction, where P allows it\n")
for a, rows in data.items():
    if len(rows) < 3:
        continue
    exact = 1.0 / (1.0 - a)
    c1 = np.array([r[1] for r in rows])
    c2 = np.array([r[2] for r in rows])
    P = np.array([r[3] for r in rows])

    s1, b1 = np.polyfit(c1, c2, 1)
    via_c1 = s1 * exact + b1

    spread = P.max() - P.min()
    print(f"  a = {a}")
    print(f"    via c1 anchor          c2 = {via_c1:.8f}   "
          f"beta = {1.0 + (via_c1 - 1.0) / (a * via_c1):.7f}")
    if spread < 1e-10:
        print(f"    via P anchor           refused: P varies by only "
              f"{spread:.1e} across N,")
        print(f"                           which is machine precision, not an "
              f"error signal.")
    else:
        s2, b2 = np.polyfit(P, c2, 1)
        via_P = b2
        print(f"    via P anchor           c2 = {via_P:.8f}   "
              f"beta = {1.0 + (via_P - 1.0) / (a * via_P):.7f}")
        print(f"    difference             {abs(via_c1 - via_P):.2e}")
    print()

print("=" * 76)
print("\n  Part 3: is P zero because of the equation, or because of symmetry?\n")
a = 0.80
r = solve(a, 2048)
g, f, z, pv, scale = r
y1 = z[0][0]
U, _, _ = dg.fields(g, f)
# 1/U odd about y1 means U(y1 + s) = -U(y1 - s). Test U itself, on a shifted
# grid so no sample lands on a zero.
s = g.x + 0.5 * g.dx
Up = np.array([evaluate(g, U, y1 + t) for t in s])
Um = np.array([evaluate(g, U, y1 - t) for t in s])
asym = np.abs(Up + Um).max() / np.abs(U).max()
print(f"    max |U(y1+s) + U(y1-s)| / ||U||_inf  =  {asym:.3e}")
print()
print("    If that is machine zero then U is odd about y1, 1/U is odd about")
print("    y1, and PV oint dy/U vanishes by antisymmetry alone. The identity")
print("    would then be true of the profile equation in general but vacuous")
print("    as a numerical check on this particular solution, in exactly the")
print("    way finding 11's pi separation turned out to be.")
print()
