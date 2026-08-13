"""
Solve for c2 properly, by refinement and extrapolation.

c2 = H f(y2) is the one quantity still only measured. A closed form looks
unlikely: writing f = i(psi' - conj(psi)') and U = psi + conj(psi) with psi the
positive frequency part, the equation becomes

    a (psi + psi~)(psi'' - psi~'') = (psi' - psi~')(psi' + psi~' - 1)

and the mixed products psi psi~'' and psi~ psi'' split across both halves of
the spectrum. That non separation is the whole difficulty of the model, and it
is why a = 0 has a Riccati solution and a != 0 does not.

So get the number instead. Two things to fix relative to how c2 has been read
off so far.

  Reading it from one grid. The profile is only C^1,1, so everything converges
  algebraically, and mu2 has bounced between 1.999354, 2.003911 and 2.001668 at
  N = 1024, 2048 and 4096 with no monotone trend. Three points cannot separate
  a convergence rate from noise. Take seven and fit.

  Locating y2 by interpolation. Already shown harmless at 4e-16, but it costs
  nothing to polish by Newton on the interpolant, since U' = H f exactly.

There is also an unexplained structural fact worth pinning here. The two
stagnation points came out exactly pi apart at every a tested, to ten digits.
Writing U(y1) = 0 and U(y1 + pi) = 0 and adding and subtracting shows that the
even mode part and the odd mode part of U must each vanish at y1 separately,
which is two conditions where one would be generic. If that holds to machine
precision it is a real constraint on the profile, not a coincidence.

    python c2_solve.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8
NS = (512, 768, 1024, 1536, 2048, 3072, 4096)


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def zeros_of_U(g, f, a):
    """Both stagnation points, polished by Newton, sorted by mu."""
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
        c = evaluate(g, Hf, x)
        out.append((x % (2.0 * np.pi), c, (c - 1.0) / (a * c)))
    return sorted(out, key=lambda r: r[2])


print()
print("=" * 78)
print("Part 1: c1, c2 and the separation, refined")
print("=" * 78)
print()
print(f"  {'N':>6}  {'c1':>12}  {'1/(1-a)':>10}  {'c2':>13}  {'mu2':>12}  "
      f"{'separation - pi':>16}")
print("  " + "-" * 76)

rows = []
for n in NS:
    g, out = simulate(A, n=n, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, A)
    if not ok:
        print(f"  {n:6d}  Newton failed")
        continue
    z = zeros_of_U(g, f, A)
    if len(z) != 2:
        print(f"  {n:6d}  {len(z)} stagnation points")
        continue
    sep = abs(z[1][0] - z[0][0])
    sep = min(sep, 2.0 * np.pi - sep)
    rows.append((n, z[0][1], z[1][1], z[1][2], sep, g, f))
    print(f"  {n:6d}  {z[0][1]:12.8f}  {1.0 / (1.0 - A):10.6f}  "
          f"{z[1][1]:13.8f}  {z[1][2]:12.8f}  {sep - np.pi:+16.3e}")

print()
print("=" * 78)
print("Part 2: extrapolate c2 to the continuum")
print("=" * 78)
print("  Fit c2(N) = c2_inf + A N^-p by least squares on the three parameters.")
print("  A clean p near 2 would match truncating a k^-3 spectrum at k_cut.")

if len(rows) >= 4:
    Nv = np.array([r[0] for r in rows], dtype=float)
    c2v = np.array([r[2] for r in rows])
    mu2v = np.array([r[3] for r in rows])

    def extrapolate(vals, label):
        best = None
        for p in np.linspace(0.5, 4.0, 351):
            X = np.column_stack([np.ones_like(Nv), Nv ** (-p)])
            coef, *_ = np.linalg.lstsq(X, vals, rcond=None)
            r = vals - X @ coef
            ss = float((r ** 2).sum())
            if best is None or ss < best[0]:
                best = (ss, p, coef[0], coef[1])
        ss, p, v_inf, amp = best
        rms = np.sqrt(ss / len(vals))
        print()
        print(f"  {label}: exponent p = {p:.3f}, "
              f"extrapolated value = {v_inf:.8f}")
        print(f"    amplitude = {amp:+.4e}, rms residual = {rms:.3e}")
        print(f"    finest grid value = {vals[-1]:.8f}, "
              f"shift from it = {v_inf - vals[-1]:+.3e}")
        return v_inf, rms

    c2_inf, c2_rms = extrapolate(c2v, "c2 ")
    mu2_inf, mu2_rms = extrapolate(mu2v, "mu2")

    print()
    print("  Consistency: mu2 computed from the extrapolated c2 should match")
    print("  the separately extrapolated mu2.")
    print(f"    from extrapolated c2: {(c2_inf - 1.0) / (A * c2_inf):.8f}")
    print(f"    extrapolated directly: {mu2_inf:.8f}")

    print()
    print("  Candidates for c2, against the extrapolated value:")
    cands = [("1/(1-2a)  [mu2 = 2]", 1.0 / (1.0 - 2.0 * A)),
             ("-1/(1-a)", -1.0 / (1.0 - A)),
             ("-(1+a)/(1-a)", -(1.0 + A) / (1.0 - A)),
             ("a/(a-1)", A / (A - 1.0)),
             ("-1/a", -1.0 / A),
             ("-2a", -2.0 * A)]
    print(f"    {'candidate':>22}  {'value':>12}  {'difference':>12}")
    for name, val in cands:
        print(f"    {name:>22}  {val:12.8f}  {val - c2_inf:+12.3e}")

print()
print("=" * 78)
print("Part 3: why exactly pi? Do both parity halves of U vanish at y1?")
print("=" * 78)
print("  U(y1) = 0 and U(y1 + pi) = 0 together force the even mode part and")
print("  the odd mode part of U each to vanish at y1. Two conditions where")
print("  one would be generic, so check both directly.")

if rows:
    n, _, _, _, _, g, f = rows[-1]
    U, _, _ = dg.fields(g, f)
    Uh = g.fwd(U)
    even, odd = Uh.copy(), Uh.copy()
    even[g.k % 2 == 1] = 0.0
    odd[g.k % 2 == 0] = 0.0
    Ue, Uo = g.bwd(even), g.bwd(odd)
    z = zeros_of_U(g, f, A)
    print()
    print(f"  at N = {n}, scale of U is {np.abs(U).max():.4f}")
    for idx, (ys, c, mu) in enumerate(z):
        print(f"    y{idx + 1} = {ys:.9f}:  U_even = "
              f"{evaluate(g, Ue, ys):+.3e}   U_odd = "
              f"{evaluate(g, Uo, ys):+.3e}")
    print()
    print("  Both near machine zero means the pi separation is exact structure.")
    print("  One of them merely small means the separation is approximate and")
    print("  the ten digit agreement was luck.")
