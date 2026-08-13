"""
Does any of this depend on starting from sin x?

Every result so far comes from omega_0 = sin x, which is exactly the ground
state of the a = 1 model. That is the largest caveat in the whole account: the
frozen profile, the selection window, and the constants could all be artefacts
of starting on a distinguished datum rather than facts about the model.

It is also a sharp test of the stability result rather than merely a robustness
check. Section 6 found the profile linearly stable, with no unstable directions
beyond the two forced by symmetry. That predicts that any datum in its basin
converges to the same profile, so the two findings, derived independently, must
agree here.

Quantities to compare, all translation invariant:

  ||f||_inf, the profile amplitude, which the fixed point equation determines
  outright and which is measurable from the reciprocal slope;
  c1, which Proposition 4 says is 1/(1-a) whatever the datum;
  c2, the global constant, and hence beta.

The data are sin x, a tilted version which is not the ground state, a one
signed datum with nonzero mean, and a datum built from higher modes only so
that it shares no Fourier content with the ground state at all.

    python generic.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, residual, simulate, sup

A = 0.8
N = 2048


def data(name, x):
    if name == "sin x":
        return np.sin(x)
    if name == "tilted":
        return np.sin(x) + 0.3 * np.sin(2.0 * x)
    if name == "one signed":
        return 1.0 - np.cos(x)
    if name == "high modes":
        return np.sin(3.0 * x) + 0.4 * np.sin(4.0 * x)
    if name == "asymmetric":
        return np.sin(x) + 0.5 * np.cos(2.0 * x) - 0.2 * np.sin(3.0 * x)
    raise ValueError(name)


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def constants(g, f, a):
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
            s = evaluate(g, U, x) / du
            x -= s
            if abs(s) < 1e-15:
                break
        c = evaluate(g, Hf, x)
        out.append((c, (c - 1.0) / (a * c)))
    return sorted(out, key=lambda r: r[1])


print()
print(f"  a = {A}, N = {N}, run to amplitude 1e6")
print()
print(f"  {'datum':>12}  {'mean':>7}  {'raw resid':>11}  {'||f||_inf':>11}  "
      f"{'c1':>10}  {'c2':>11}  {'beta':>9}")
print("  " + "-" * 80)

results = []
for name in ("sin x", "tilted", "one signed", "high modes", "asymmetric"):
    g = dg.Grid(N)
    w0 = data(name, g.x)
    out = dg.run(w0, a=A, grid=g, t_max=120.0, cfl=0.02, tail_tol=1e-9,
                 w_max=1e6)
    if out["reason"] != "amplitude cap":
        print(f"  {name:>12}  no blowup reached: {out['reason']} at "
              f"|w| = {out['winf_hist'][-1]:.2e}")
        continue

    f0, amp_sim = guess_from_simulation(g, out)
    if f0 is None:
        print(f"  {name:>12}  no usable growth")
        continue
    raw = float(np.abs(residual(g, f0, A)).max())
    f, hist, ok = newton(g, f0, A)
    if not ok or sup(g, f) < 1e-6:
        print(f"  {name:>12}  Newton failed or trivial, raw resid {raw:.2e}")
        continue

    cs = constants(g, f, A)
    c1, c2 = cs[0][0], cs[1][0]
    beta = cs[1][1] + 1.0
    results.append((name, sup(g, f), c1, c2, beta))
    print(f"  {name:>12}  {w0.mean():7.3f}  {raw:11.3e}  {sup(g, f):11.7f}  "
          f"{c1:10.6f}  {c2:11.7f}  {beta:9.6f}")

print()
print(f"  exact c1 = 1/(1-a) = {1.0 / (1.0 - A):.6f}")

if len(results) >= 2:
    amps = np.array([r[1] for r in results])
    c1s = np.array([r[2] for r in results])
    c2s = np.array([r[3] for r in results])
    print()
    print("  spread across data:")
    print(f"    ||f||_inf : {amps.max() - amps.min():.3e}   "
          f"(relative {(amps.max() - amps.min()) / amps.mean():.3e})")
    print(f"    c1        : {c1s.max() - c1s.min():.3e}   "
          f"max deviation from 1/(1-a): "
          f"{np.abs(c1s - 1.0 / (1.0 - A)).max():.3e}")
    print(f"    c2        : {c2s.max() - c2s.min():.3e}   "
          f"(relative {(c2s.max() - c2s.min()) / abs(c2s.mean()):.3e})")
    print()
    print("  A spread at the level of the N = 2048 discretisation error, which")
    print("  is about 2e-3 for these constants, means the profile is the same")
    print("  object reached from every datum, and the stability result of")
    print("  Section 6 predicts exactly that. A larger spread would mean the")
    print("  basin is narrow or the profile depends on the datum.")
