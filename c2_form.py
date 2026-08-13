"""
Hunt a closed form for c2, and test an exact identity that falls out on the way.

Two threads.

**The identity.** The equation integrates exactly once. From
f'/f = (U' - 1)/(a U),

    ln|f| = (1/a) [ ln|U| - integral dy/U ]

so f = C |U|^(1/a) exp(-(1/a) integral dy/U) on each interval between zeros of
U. Near a simple zero with U ~ c (y - y*) this reproduces |y - y*|^mu with
mu = (c-1)/(ac), which is a consistency check on the whole picture. Requiring
f to close up going once around the circle then demands

    PV integral over the circle of dy/U = 0

which is a genuine scalar constraint on U. Testable: subtract the two simple
poles using cot, since near y* the function (1/(2c)) cot((y - y*)/2) has
exactly the same residue as 1/U, and the principal value of cot integrates to
zero on the circle.

**The form.** c2 is now known to about 1e-5 by the control variate of finding
12, which is precise enough to reject candidate closed forms rather than merely
fail to confirm them. A Mobius form in a, mu2 = (p + qa)/(1 + ra), fitted on
three points and checked on a fourth already misses by 1.2e-3, a hundred times
the data's precision. To test anything richer than that needs more than four
values of a.

The control variate makes that cheap. At a = 0.8 the corrected value from
N = 512 sits within 1.5e-5 of the one from N = 4096, so the expensive grids are
not needed: N up to 1536 suffices, and a dozen values of a become affordable.

    python c2_form.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate

NS = (512, 768, 1024, 1536)
AVALS = (0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83)


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


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


def c2_corrected(a, ns=NS):
    """Control variate: regress c2 on c1, evaluate at the exact c1."""
    c1s, c2s, keep = [], [], None
    for n in ns:
        g, out = simulate(a, n=n, w_max=1e6)
        f0, _ = guess_from_simulation(g, out)
        f, hist, ok = newton(g, f0, a)
        if not ok:
            continue
        z = zeros_of_U(g, f, a)
        if len(z) != 2:
            continue
        c1s.append(z[0][1])
        c2s.append(z[1][1])
        keep = (g, f, z)
    if len(c1s) < 3:
        return np.nan, np.nan, keep
    c1s, c2s = np.array(c1s), np.array(c2s)
    s, _ = np.polyfit(c1s, c2s, 1)
    per = c2s - s * (c1s - 1.0 / (1.0 - a))
    return float(per.mean()), float(per.std(ddof=1)), keep


print()
print("=" * 76)
print("Part 1: c2 across a, by control variate on cheap grids")
print("=" * 76)
print()
print(f"  {'a':>6}  {'c2':>14}  {'scatter':>10}  {'mu2':>12}  {'beta':>12}")
print("  " + "-" * 60)

data = []
last = {}
for a in AVALS:
    c2, sd, keep = c2_corrected(a)
    if not np.isfinite(c2):
        print(f"  {a:6.2f}  failed")
        continue
    mu = (c2 - 1.0) / (a * c2)
    data.append((a, c2, mu))
    last[a] = keep
    print(f"  {a:6.2f}  {c2:14.8f}  {sd:10.2e}  {mu:12.7f}  {mu + 1.0:12.7f}")

av = np.array([d[0] for d in data])
c2v = np.array([d[1] for d in data])
muv = np.array([d[2] for d in data])


print()
print("=" * 76)
print("Part 2: does PV integral of dy/U vanish?")
print("=" * 76)
print("  Subtract both simple poles with cot, which carries the same residue,")
print("  and whose own principal value on the circle is zero.")
print()
print(f"  {'a':>6}  {'PV integral':>14}  {'scale':>12}  {'ratio':>11}")
print("  " + "-" * 48)
for a in (0.75, 0.78, 0.80, 0.82):
    if a not in last or last[a] is None:
        continue
    g, f, z = last[a]
    U, _, _ = dg.fields(g, f)
    (y1, c1), (y2, c2) = z
    reg = (1.0 / U
           - 0.5 / c1 / np.tan((g.x - y1) / 2.0)
           - 0.5 / c2 / np.tan((g.x - y2) / 2.0))
    pv = float(reg.sum() * g.dx)
    scale = float(np.abs(reg).sum() * g.dx)
    print(f"  {a:6.2f}  {pv:14.6e}  {scale:12.4f}  {abs(pv) / scale:11.2e}")
print()
print("  A ratio near machine zero says the identity holds. The subtraction")
print("  cancels two large terms near each pole, so a few digits are lost and")
print("  1e-10 rather than 1e-16 is the realistic target.")


print()
print("=" * 76)
print("Part 3: candidate closed forms for mu2(a)")
print("=" * 76)
print("  Each form is fitted by least squares on all points. With scatter at")
print("  1e-5, a residual above 1e-4 rejects the form outright.")
print()


def report(name, resid):
    rms = float(np.sqrt((resid ** 2).mean()))
    verdict = "plausible" if rms < 1e-4 else "REJECTED"
    print(f"  {name:<34}  rms {rms:9.2e}   max {np.abs(resid).max():9.2e}  "
          f"{verdict}")


# Mobius: mu (1 + r a) = p + q a
M = np.column_stack([np.ones_like(av), av, -muv * av])
coef, *_ = np.linalg.lstsq(M, muv, rcond=None)
report("mu2 = (p + qa) / (1 + ra)", muv - M @ coef)

# Quadratic over linear
M = np.column_stack([np.ones_like(av), av, av ** 2, -muv * av])
coef, *_ = np.linalg.lstsq(M, muv, rcond=None)
report("mu2 = (p + qa + sa^2) / (1 + ra)", muv - M @ coef)

# Linear over quadratic
M = np.column_stack([np.ones_like(av), av, -muv * av, -muv * av ** 2])
coef, *_ = np.linalg.lstsq(M, muv, rcond=None)
report("mu2 = (p + qa) / (1 + ra + sa^2)", muv - M @ coef)

# Quadratic over quadratic
M = np.column_stack([np.ones_like(av), av, av ** 2, -muv * av, -muv * av ** 2])
coef, *_ = np.linalg.lstsq(M, muv, rcond=None)
report("mu2 = quadratic / quadratic", muv - M @ coef)

# Same shapes for c2 rather than mu2
M = np.column_stack([np.ones_like(av), av, -c2v * av])
coef, *_ = np.linalg.lstsq(M, c2v, rcond=None)
report("c2 = (p + qa) / (1 + ra)", c2v - M @ coef)

M = np.column_stack([np.ones_like(av), av, av ** 2, -c2v * av, -c2v * av ** 2])
coef, *_ = np.linalg.lstsq(M, c2v, rcond=None)
report("c2 = quadratic / quadratic", c2v - M @ coef)

# In the variable 1/(1-a), which is what c1 is
b = 1.0 / (1.0 - av)
M = np.column_stack([np.ones_like(b), b, -muv * b])
coef, *_ = np.linalg.lstsq(M, muv, rcond=None)
report("mu2 = (p + qb) / (1 + rb),  b = 1/(1-a)", muv - M @ coef)

M = np.column_stack([np.ones_like(b), b, b ** 2, -muv * b, -muv * b ** 2])
coef, *_ = np.linalg.lstsq(M, muv, rcond=None)
report("mu2 = quadratic / quadratic in b", muv - M @ coef)

print()
print("  A rejection here is a real result: with c2 known to 1e-5, these")
print("  shapes are excluded rather than merely unconfirmed.")
