"""
Chase the +1.3e-3 residual between predicted and measured beta.

At a = 0.8 the clean band gives 3.00303 at N = 4096 and 3.00291 at N = 8192,
against a prediction of 3.001668. Same sign and size at both resolutions, so it
is systematic. Before attributing it to anything, ask how well the prediction
itself is known.

That is not a rhetorical question. mu2 came out 1.999354, 2.003911, 2.001668 at
N = 1024, 2048, 4096, bouncing by about 4e-3 and not monotonically. If the
prediction carries 2e-3 of its own uncertainty then a 1.3e-3 residual is not a
finding, it is noise, and chasing it means chasing the prediction rather than
the measurement.

Two error sources on the prediction side, worth separating.

  Locating y*. refine_zero used linear interpolation of a sign change, which is
  O(dx^2). Since U' = H f, the zero can instead be polished by Newton on the
  spectral interpolant to machine precision, which costs nothing. If the
  bouncing shrinks, it was this.

  The profile itself. f is only C^1,1, so everything read off it converges
  algebraically, and ||f||_inf already moves 1.5e-3 between N = 2048 and 4096.
  This is the floor and it cannot be removed by better root finding.

    python residual.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def zero_linear(g, U, i):
    x0, x1 = g.x[i], g.x[(i + 1) % g.N]
    if x1 < x0:
        x1 += 2.0 * np.pi
    u0, u1 = U[i], U[(i + 1) % g.N]
    return x0 + (x1 - x0) * u0 / (u0 - u1)


def zero_spectral(g, U, Hf, i, iters=60):
    """Polish the zero of U by Newton on the interpolant. U' = H f exactly."""
    x = zero_linear(g, U, i)
    for _ in range(iters):
        u = evaluate(g, U, x)
        du = evaluate(g, Hf, x)
        if du == 0.0:
            break
        step = u / du
        x -= step
        if abs(step) < 1e-15:
            break
    return x


def mus(g, f, a, spectral=True):
    U, Hf, _ = dg.fields(g, f)
    out = []
    for i in np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1))):
        x = (zero_spectral(g, U, Hf, i) if spectral
             else zero_linear(g, U, i)) % (2.0 * np.pi)
        c = evaluate(g, Hf, x)
        out.append((x, c, (c - 1.0) / (a * c), abs(evaluate(g, U, x))))
    return sorted(out, key=lambda r: r[2])


def beta_band(g, f, lo_frac=0.094, hi_frac=0.375):
    """Fitted exponent over the clean relative band, in units of k_cut."""
    amp = np.abs(g.fwd(f))
    k, floor = g.k, np.abs(g.fwd(f)).max() * 1e-13
    kc, vc = [], []
    lo = lo_frac * g.kcut
    while lo * np.sqrt(2.0) <= hi_frac * g.kcut:
        hi = lo * np.sqrt(2.0)
        sel = (k >= lo) & (k < hi) & (amp > floor)
        if sel.sum() >= 3:
            kc.append(np.sqrt(lo * hi))
            vc.append(np.sqrt((amp[sel] ** 2).mean()))
        lo = hi
    if len(kc) < 3:
        return np.nan
    x, y = np.log(np.array(kc)), np.log(np.array(vc))
    return -float(np.polyfit(x, y, 1)[0])


print()
print("=" * 76)
print("Part 1: does exact root finding stabilise the prediction?")
print("=" * 76)
print()
print(f"  {'N':>7}  {'mu2 linear':>12}  {'mu2 spectral':>13}  "
      f"{'shift':>10}  {'|U(y*)|':>10}  {'||f||_inf':>11}")
print("  " + "-" * 70)

solved = {}
for n in (1024, 2048, 4096):
    g, out = simulate(A, n=n, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    f, hist, ok = newton(g, f0, A)
    if not ok:
        print(f"  {n:7d}  Newton failed")
        continue
    solved[n] = (g, f)
    lin = mus(g, f, A, spectral=False)[1]
    spe = mus(g, f, A, spectral=True)[1]
    print(f"  {n:7d}  {lin[2]:12.7f}  {spe[2]:13.7f}  "
          f"{spe[2] - lin[2]:+10.2e}  {spe[3]:10.2e}  {sup(g, f):11.6f}")

print()
print("  If the shift column is far below the 4e-3 spread across N, then the")
print("  root finding was never the problem and the spread is the profile's")
print("  own algebraic convergence.")


print()
print("=" * 76)
print("Part 2: prediction and measurement on the same profile")
print("=" * 76)
print("  Both columns come from one object at one resolution, so the gap")
print("  between them is not contaminated by comparing different profiles.")
print()
print(f"  {'N':>7}  {'beta predicted':>15}  {'beta measured':>14}  "
      f"{'gap':>11}")
print("  " + "-" * 54)

gaps = {}
for n, (g, f) in solved.items():
    bp = mus(g, f, A)[1][2] + 1.0
    bm = beta_band(g, f)
    gaps[n] = bm - bp
    print(f"  {n:7d}  {bp:15.7f}  {bm:14.7f}  {bm - bp:+11.3e}")

# The simulated profile at N = 8192, where a dense Newton solve is out of
# reach. It satisfies the equation to about 1e-5, so treat it as indicative.
g8, out8 = simulate(A, n=8192, w_max=1e6)
f8, _ = guess_from_simulation(g8, out8)
res8 = float(np.abs(dg.rhs(g8, f8, A) - f8).max())
bp8 = mus(g8, f8, A)[1][2] + 1.0
bm8 = beta_band(g8, f8)
print(f"  {8192:7d}  {bp8:15.7f}  {bm8:14.7f}  {bm8 - bp8:+11.3e}   "
      f"(simulated profile, residual {res8:.1e})")

print()
print("  A gap that shrinks with N says the residual is discretisation on one")
print("  side or the other. A gap that holds says something systematic is")
print("  left. Compare it against the spread of the prediction itself.")

if len(gaps) >= 2:
    spread = max(abs(mus(*solved[n], A)[1][2] - mus(*solved[m], A)[1][2])
                 for n in solved for m in solved if n < m)
    print()
    print(f"  spread of the prediction across N: {spread:.3e}")
    print(f"  gap at the highest solved N:       "
          f"{abs(gaps[max(gaps)]):.3e}")
    if abs(gaps[max(gaps)]) < spread:
        print()
        print("  The gap is smaller than the prediction's own spread, so it is")
        print("  not resolvable at these resolutions and should not be quoted")
        print("  as a finding.")
