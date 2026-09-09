"""
Is the control variate finding the truth, or a consistent wrong number?

Finding 12 corrects `c2` by regressing it against `c1` across resolutions and
evaluating the line at the exact `c1 = 1/(1-a)`. The spread collapses by a
factor of 801 and the paper quotes five digits. The assumption underneath has
never been tested: that the discretisation error enters `c1` and `c2` linearly
with one common coefficient, so the line through the measured pairs passes
through the truth rather than merely near it. The evidence offered is
`R^2 = 0.99999744` on seven points, which is the shape of evidence that
produced findings 9 and 11.

`cv_second_anchor.py` tried a second exactly known quantity on the same profile
and failed structurally: `PV oint dy/U = 0` holds by antisymmetry, carries no
discretisation error, and anchors nothing.

A manufactured solution was the next idea and it fails for the mirror reason,
which is worth recording since it looks like it ought to work. Build an `f*`
with the right structure, compute the source `s = rhs(f*) - f*` that makes it
an exact solution of a modified equation, solve that on the same grids and
score the correction against the planted answer. Part 4 does it. The error it
produces is the wrong shape: pinning the solution with a source leaves `c1`
accurate to 6e-10 while `c2` is off by 4e-5, so the anchor does not move and
there is nothing to regress against. The real `c1` wanders by 2e-2. A
manufactured problem can be given the right profile but not the right error.

So the first three parts test the assumption on the real profile, where the
error is the one that matters, using the fact that the assumption has testable
consequences even though `c2` itself is unknown.

Part 1 asks what the error actually looks like. If it is one fixed shape whose
amplitude alone depends on `N`, then every linear functional of it is
proportional to every other across the ladder, and the control variate is
justified for any pair of constants, not just this one. That is a statement
about the rank of the matrix of differences, and the singular values answer it.
Profiles are aligned on their stagnation point first, since a relative
translation would show up as a large spurious mode that no reading of `Hf` at a
stagnation point is sensitive to.

Part 2 is a held out test. Split the ladder, correct on each half separately,
and compare. The corrected value is supposed to be resolution independent, so
the two halves must agree to the quoted precision. They share no resolutions,
so agreement is not automatic.

Part 3 changes the error rather than the resolution. Hold `N` fixed and vary
the dealiasing fraction, which alters the retained band without touching the
grid. If the pairs from that sweep fall on the line the `N` ladder traced, the
relation is a property of the profile. If they trace a different line, the
correction is calibrated to the ladder it was measured on and the five digits
are an artefact of it.

    python cv_manufactured.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, jacobian, newton, residual, simulate

A = 0.8
C1_EXACT = 1.0 / (1.0 - A)
NS = (512, 768, 1024, 1536, 2048, 3072)
FRACS = (0.20, 0.23, 0.26, 0.29, 1.0 / 3.0)
FRAC_N = 2048
ALIGN = 4096


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def stagnation_points(g, f, a=A):
    """
    Zeros of U with H f evaluated there, sorted by the local exponent so that
    the first is the smooth point and the second is the singular one. Same
    reading as c2_control.zeros_of_U, repeated here because that module runs its
    experiment at import.
    """
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


def cut_grid(n, frac=1.0 / 3.0):
    g = dg.Grid(n)
    g.kcut = int(frac * n)
    g.mask = (g.k <= g.kcut).astype(float)
    g.mask[-1] = 0.0
    return g


def interp(f, n_to):
    """Spectral interpolation onto n_to points."""
    fh = np.fft.rfft(f)
    out = np.zeros(n_to // 2 + 1, dtype=complex)
    m = min(len(fh), len(out))
    out[:m] = fh[:m]
    out[-1] = 0.0
    return np.fft.irfft(out, n_to) * (n_to / len(f))


def shift(f, delta):
    """Spectral translation, f(y - delta)."""
    fh = np.fft.rfft(f)
    k = np.arange(len(fh))
    return np.fft.irfft(fh * np.exp(-1j * k * delta), len(f))


def band(g, v):
    return g.bwd(g.fwd(v))


def correct(c1, c2, c1_exact=C1_EXACT):
    """The finding 12 correction: regress, then evaluate at the exact anchor."""
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    s, b = np.polyfit(c1, c2, 1)
    r = c2 - (s * c1 + b)
    ss = ((c2 - c2.mean()) ** 2).sum()
    r2 = 1.0 - (r ** 2).sum() / ss if ss > 0 else np.nan
    return float(s * c1_exact + b), float(s), float(r2)


def solve_profile(g, seed):
    f0 = band(g, interp(seed, g.N))
    f, _, ok = newton(g, f0, A)
    if not ok:
        return None
    pts = stagnation_points(g, f)
    if len(pts) != 2:
        return None
    return f, pts[0], pts[1]


# ---------------------------------------------------------------------------
# One simulation seeds everything
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print(f"Solving the profile at a = {A}")
print("=" * 78)
print()
print("  Newton converges to the fixed point of the discrete equation, which")
print("  is a property of the grid and not of the seed, so one simulation")
print("  interpolated onto each grid serves the whole script.")
print()

g_seed, out_seed = simulate(A, n=2048, w_max=1e6)
seed, _ = guess_from_simulation(g_seed, out_seed)

print(f"  {'N':>6}  {'kcut':>6}  {'y1':>12}  {'c1':>13}  {'c1 - exact':>11}  "
      f"{'c2':>13}")
print("  " + "-" * 70)

ladder = []
for n in NS:
    g = cut_grid(n)
    got = solve_profile(g, seed)
    if got is None:
        print(f"  {n:6d}  {g.kcut:6d}  did not converge")
        continue
    f, p1, p2 = got
    ladder.append({"n": n, "g": g, "f": f, "y1": p1[0],
                   "c1": p1[1], "c2": p2[1]})
    print(f"  {n:6d}  {g.kcut:6d}  {p1[0]:12.8f}  {p1[1]:13.8f}  "
          f"{p1[1] - C1_EXACT:+11.2e}  {p2[1]:13.8f}")

if len(ladder) < 4:
    raise SystemExit("\n  too few resolutions converged to analyse\n")

c1_lad = np.array([r["c1"] for r in ladder])
c2_lad = np.array([r["c2"] for r in ladder])
full_corr, full_slope, full_r2 = correct(c1_lad, c2_lad)

print()
print(f"  full ladder: c2 = {full_slope:+.6f} c1 + const,  R^2 = {full_r2:.8f}")
print(f"  corrected c2 = {full_corr:.8f}")

# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("Part 1: is the discretisation error a single shape?")
print("=" * 78)
print()
print("  Every profile is interpolated to a common grid and shifted so its")
print("  smooth stagnation point sits at the same place, since a relative")
print("  translation is a large mode that c1 and c2 cannot see. Differences")
print("  are taken against the finest, and the singular values of the stack")
print("  say how many independent shapes the error has.")

fine = ladder[-1]
ref = shift(interp(fine["f"], ALIGN), -fine["y1"])
rows = []
for r in ladder[:-1]:
    al = shift(interp(r["f"], ALIGN), -r["y1"])
    rows.append(al - ref)
D = np.array(rows)

sv = np.linalg.svd(D, compute_uv=False)
print()
print(f"  {'i':>3}  {'singular value':>16}  {'fraction of sv[0]':>18}")
print("  " + "-" * 41)
for i, s in enumerate(sv):
    print(f"  {i:3d}  {s:16.6e}  {s / sv[0]:18.3e}")

ratio = sv[0] / sv[1] if len(sv) > 1 and sv[1] > 0 else np.inf
energy = float(sv[0] ** 2 / (sv ** 2).sum())
print()
print(f"  sv[0] / sv[1]                  = {ratio:.1f}")
print(f"  fraction of variance in mode 0 = {energy:.8f}")
print()
if energy > 0.9999:
    print("  The error is one shape to better than a part in 1e4. Any two")
    print("  linear functionals of it are then proportional across the ladder,")
    print("  which is exactly what the correction assumes, and it holds for")
    print("  reasons that have nothing to do with which two constants are read.")
elif energy > 0.99:
    print("  The error is dominated by one shape but not purely one shape. The")
    print("  correction is then right to leading order, with a second order")
    print("  term the regression cannot see and does not correct.")
else:
    print("  The error has several independent shapes of comparable size, so")
    print("  proportionality of any two functionals is not structural and the")
    print("  linear fit is a coincidence of this particular pair.")

# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("Part 2: split the ladder, correct on each half")
print("=" * 78)
print()
print("  The corrected value is supposed to be resolution independent. The two")
print("  halves share no resolutions, so agreement is a real check and not an")
print("  identity.")

half = len(ladder) // 2
lo = slice(0, half + len(ladder) % 2)
hi = slice(half + len(ladder) % 2, None)

print()
print(f"  {'half':>10}  {'N':>22}  {'slope':>11}  {'corrected c2':>14}")
print("  " + "-" * 62)
corr_halves = []
for name, sl in (("coarse", lo), ("fine", hi)):
    if len(c1_lad[sl]) < 2:
        print(f"  {name:>10}  too few points")
        continue
    if len(c1_lad[sl]) == 2:
        cc, ss, _ = correct(c1_lad[sl], c2_lad[sl])
        rr = float("nan")
    else:
        cc, ss, rr = correct(c1_lad[sl], c2_lad[sl])
    corr_halves.append(cc)
    ns = ",".join(str(r["n"]) for r in ladder[sl])
    print(f"  {name:>10}  {ns:>22}  {ss:+11.6f}  {cc:14.8f}")

if len(corr_halves) == 2:
    gap = abs(corr_halves[0] - corr_halves[1])
    print()
    print(f"  the two halves differ by {gap:.3e}")
    print(f"  finding 12 quotes c2 to +/- 1.0e-05")
    print()
    if gap < 1e-5:
        print("  Inside the quoted error bar. Splitting the ladder does not")
        print("  move the answer.")
    else:
        print("  Outside the quoted error bar. The correction is not yet")
        print("  resolution independent at the precision claimed for it.")

# ---------------------------------------------------------------------------
# Part 3
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("Part 3: a second knob, the dealiasing fraction at fixed N")
print("=" * 78)
print()
print(f"  N is held at {FRAC_N} and the retained band is varied instead, which")
print("  induces the error a different way on the same grid.")
print()
print(f"  {'frac':>7}  {'kcut':>6}  {'c1':>13}  {'c1 - exact':>11}  {'c2':>13}")
print("  " + "-" * 56)

knob_b = []
for frac in FRACS:
    g = cut_grid(FRAC_N, frac)
    got = solve_profile(g, seed)
    if got is None:
        print(f"  {frac:7.4f}  {g.kcut:6d}  did not converge")
        continue
    _, p1, p2 = got
    knob_b.append((p1[1], p2[1]))
    print(f"  {frac:7.4f}  {g.kcut:6d}  {p1[1]:13.8f}  "
          f"{p1[1] - C1_EXACT:+11.2e}  {p2[1]:13.8f}")

if len(knob_b) >= 3:
    c1b = np.array([p[0] for p in knob_b])
    c2b = np.array([p[1] for p in knob_b])
    corr_b, slope_b, r2_b = correct(c1b, c2b)
    print()
    print(f"  knob A, resolution:  slope {full_slope:+.6f}  "
          f"R^2 {full_r2:.8f}  c2 {full_corr:.8f}")
    print(f"  knob B, dealiasing:  slope {slope_b:+.6f}  "
          f"R^2 {r2_b:.8f}  c2 {corr_b:.8f}")
    print()
    print(f"  slopes differ by            {abs(full_slope - slope_b):.3e}")
    print(f"  corrected values differ by  {abs(full_corr - corr_b):.3e}")
    print()
    if abs(full_corr - corr_b) < 1e-5:
        print("  The two knobs agree to the quoted precision. The linear")
        print("  relation is a property of the profile rather than of the")
        print("  ladder, and the correction does not depend on which")
        print("  discretisation error induced it.")
    else:
        print("  The two knobs disagree beyond the quoted precision. The")
        print("  correction is calibrated to the error it was measured on, and")
        print("  the error bar in finding 12 should be widened to at least")
        print(f"  {abs(full_corr - corr_b):.1e}.")

# ---------------------------------------------------------------------------
# Part 4
# ---------------------------------------------------------------------------

print()
print("=" * 78)
print("Part 4: the manufactured solution, and why it cannot settle this")
print("=" * 78)
print()
print("  f* is built with the structure of the real profile: odd about y1 so")
print("  both stagnation points are forced, a simple zero there, and f*'(y2)=0")
print("  with a jump in f'' at y2 = y1 + pi, giving mu = 2 and a k^-3 spectrum.")
print("  b_1 is fixed by requiring that double zero, not chosen. The source")
print("  s = rhs(f*) - f* then makes f* an exact solution, and both constants")
print("  are known in closed form as sums of the coefficients.")

Y1_STAR = 0.975612
KMAX = 1 << 13
N_REF = 1 << 16
NS_STAR = (256, 512, 1024, 2048)


def star_coefficients(kmax=KMAX):
    b = np.zeros(kmax + 1)
    k = np.arange(2, kmax + 1)
    b[k] = (-1.0) ** k / k ** 3.0
    b[1] = np.pi ** 2 / 6.0 - 1.0
    return b


b_star = star_coefficients()
k_all = np.arange(len(b_star))
c1_star = -float(b_star.sum())
c2_star = -float((b_star * (-1.0) ** k_all).sum())

k = np.arange(N_REF // 2 + 1)
fh = np.zeros(N_REF // 2 + 1, dtype=complex)
kk = np.arange(1, len(b_star))
fh[kk] = -1j * (N_REF / 2.0) * b_star[kk]
fh *= np.exp(-1j * k * Y1_STAR)
hmul = -1j * np.ones(k.shape, dtype=complex)
hmul[0] = 0.0
umul = np.zeros(k.shape, dtype=complex)
umul[1:] = -1.0 / k[1:]

f_ref = np.fft.irfft(fh, N_REF)
hf_ref = np.fft.irfft(fh * hmul, N_REF)
u_ref = np.fft.irfft(fh * umul, N_REF)
fx_ref = np.fft.irfft(fh * (1j * k), N_REF)
s_ref = f_ref * hf_ref - A * u_ref * fx_ref - f_ref

xs = np.arange(0, N_REF, N_REF // 8)
direct = np.array([np.sum(b_star[kk] * np.sin(kk * (2.0 * np.pi * j / N_REF
                                                    - Y1_STAR))) for j in xs])
print()
print(f"  spectrum against direct series: {np.abs(f_ref[xs] - direct).max():.2e}")
print(f"  planted c1 = {c1_star:.12f}")
print(f"  planted c2 = {c2_star:.12f}")
print()
print(f"  {'N':>6}  {'c1 - true':>12}  {'c2 - true':>12}  {'newton':>9}")
print("  " + "-" * 46)


def newton_source(g, f, a, s, tol=1e-12, max_iter=40):
    """
    Newton on rhs(f) - f = s. No regularisation along the translation mode and
    none is wanted: the source breaks translation invariance, since
    J f' = (rhs(f) - f)' = s' rather than zero, so the Jacobian that is
    singular for the homogeneous problem is nonsingular here.
    """
    for _ in range(max_iter):
        r = residual(g, f, a) - s
        rn = float(np.abs(r).max())
        if rn < tol:
            return f, rn, True
        if not np.isfinite(rn) or rn > 1e8:
            return f, rn, False
        try:
            f = f - np.linalg.solve(jacobian(g, f, a), r)
        except np.linalg.LinAlgError:
            return f, rn, False
    r = float(np.abs(residual(g, f, a) - s).max())
    return f, r, r < tol


star_rows = []
for n in NS_STAR:
    assert N_REF % n == 0, "manufactured grids must divide the reference grid"
    g = dg.Grid(n)
    step = N_REF // n
    f, rn, ok = newton_source(g, band(g, f_ref[::step]), A,
                              band(g, s_ref[::step]))
    if not ok:
        print(f"  {n:6d}  newton failed, residual {rn:.2e}")
        continue
    pts = sorted(stagnation_points(g, f))
    if len(pts) != 2:
        print(f"  {n:6d}  {len(pts)} stagnation points, expected 2")
        continue
    d = [abs((p - Y1_STAR + np.pi) % (2.0 * np.pi) - np.pi) for p, _, _ in pts]
    i1 = int(np.argmin(d))
    c1n, c2n = pts[i1][1], pts[1 - i1][1]
    star_rows.append((c1n, c2n))
    print(f"  {n:6d}  {c1n - c1_star:+12.2e}  {c2n - c2_star:+12.2e}  {rn:9.1e}")

if len(star_rows) >= 3:
    e1 = np.abs(np.array([r[0] for r in star_rows]) - c1_star)
    e2 = np.abs(np.array([r[1] for r in star_rows]) - c2_star)
    print()
    print(f"  worst error in the anchor c1: {e1.max():.2e}")
    print(f"  worst error in the target c2: {e2.max():.2e}")
    print(f"  the anchor is quieter by a factor of {e2.max() / e1.max():.0f}")
    print()
    print("  For comparison the real c1 moves by "
          f"{np.abs(c1_lad - C1_EXACT).max():.1e} across the ladder above.")
    print()
    print("  This is why the manufactured problem cannot score the method. A")
    print("  control variate needs its anchor to carry the same error as its")
    print("  target, and pinning the solution with a source removes the error")
    print("  at the smooth point while leaving it at the rough one. The same")
    print("  requirement that emptied the PV anchor empties this one. Parts 1")
    print("  to 3 are the test; this part is the record of an approach that")
    print("  looks right and is not.")
print()
