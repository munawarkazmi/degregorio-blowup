"""
Map the basin of the frozen profile at a = 0.8.

Three facts constrain the search before it starts. The nonlinearity is
quadratic, so replacing omega_0 by lambda omega_0 rescales time by 1/lambda and
changes nothing else: the basin is scale invariant and only shape matters.
Proposition on dilation says profiles come in a family f(nx), so "which
profile" is a question about which member. And one signed data did not blow up
at all, so the basin has a boundary somewhere between that and sin x.

Three experiments.

  A. The boundary. omega_0 = sin x + m interpolates from sign changing at
     m = 0 to one signed at m = 1, where the zero at x = -pi/2 becomes a
     touching point rather than a crossing. If sign change is what matters the
     transition sits exactly at m = 1.

  B. Which member. omega_0 = sin(kx) should land on the k-th member by
     dilation, and mixtures should land on whatever their structure dictates.
     The member is read off the number of stagnation points, which is 2n.

  C. Random smooth data, to see what a generic datum does rather than a
     designed one.

Classification is by the number of stagnation points of the converged profile
and by its constants, both of which are unchanged along the dilation family
except that the n-th member is represented on an effective grid of N/n.

    python basin.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, residual, simulate, sup

A = 0.8
N = 2048
WMAX = 1.0e4


def evaluate(g, field, x):
    fh = g.fwd(field)
    c = 2.0 * fh / g.N
    c[0] = fh[0] / g.N
    c[-1] = fh[-1] / g.N
    return float(np.real(np.sum(c * np.exp(1j * g.k * x))))


def stagnation_count(g, f, a):
    U, Hf, _ = dg.fields(g, f)
    idx = np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1)))
    mus = []
    for i in idx:
        x0, x1 = g.x[i], g.x[(i + 1) % g.N]
        if x1 < x0:
            x1 += 2.0 * np.pi
        u0, u1 = U[i], U[(i + 1) % g.N]
        x = x0 + (x1 - x0) * u0 / (u0 - u1)
        for _ in range(40):
            du = evaluate(g, Hf, x)
            if du == 0.0:
                break
            s = evaluate(g, U, x) / du
            x -= s
            if abs(s) < 1e-15:
                break
        c = evaluate(g, Hf, x)
        mus.append((c - 1.0) / (a * c))
    return len(idx), mus


def sign_changes(w):
    s = np.sign(w)
    s = s[s != 0]
    if len(s) == 0:
        return 0
    return int((s != np.roll(s, -1)).sum())


def classify(w0, n=N, w_max=WMAX, t_max=150.0):
    g = dg.Grid(n)
    w0 = w0 / np.abs(w0).max()
    out = dg.run(w0, a=A, grid=g, t_max=t_max, cfl=0.02, tail_tol=1e-9,
                 w_max=w_max)
    sc = sign_changes(w0)
    if out["reason"] != "amplitude cap":
        return dict(kind="no blowup", sc=sc, amp=out["winf_hist"][-1],
                    t=out["t"], reason=out["reason"])
    f0, _ = guess_from_simulation(g, out)
    if f0 is None:
        return dict(kind="unclassified", sc=sc, amp=out["winf_hist"][-1],
                    t=out["t"], reason=out["reason"])
    raw = float(np.abs(residual(g, f0, A)).max())
    f, hist, ok = newton(g, f0, A)
    if not ok or sup(g, f) < 1e-6:
        return dict(kind="newton failed", sc=sc, raw=raw, t=out["t"],
                    reason=out["reason"])
    nst, mus = stagnation_count(g, f, A)
    return dict(kind="blowup", sc=sc, raw=raw, member=nst // 2,
                amp_f=sup(g, f), mus=sorted(mus), t=out["t"],
                reason=out["reason"])


print()
print("=" * 74)
print("A. The boundary: omega_0 = sin x + m")
print("=" * 74)
print("  m = 1 is where the sign change becomes a touching zero.")
print()
print(f"  {'m':>6}  {'sign chgs':>10}  {'outcome':>12}  {'member':>7}  "
      f"{'||f||_inf':>11}  {'t':>8}")
print("  " + "-" * 62)
g = dg.Grid(N)
for m in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01, 1.05, 1.2):
    r = classify(np.sin(g.x) + m)
    if r["kind"] == "blowup":
        print(f"  {m:6.2f}  {r['sc']:10d}  {'blowup':>12}  {r['member']:7d}  "
              f"{r['amp_f']:11.6f}  {r['t']:8.3f}")
    else:
        print(f"  {m:6.2f}  {r['sc']:10d}  {r['kind']:>12}  {'-':>7}  "
              f"{'-':>11}  {r['t']:8.3f}")

print()
print("=" * 74)
print("B. Which member: omega_0 = sin(kx), and mixtures")
print("=" * 74)
print()
print(f"  {'datum':>22}  {'sign chgs':>10}  {'member':>7}  {'||f||_inf':>11}  "
      f"{'exponents':>20}")
print("  " + "-" * 76)
cases = [("sin x", np.sin(g.x)),
         ("sin 2x", np.sin(2 * g.x)),
         ("sin 3x", np.sin(3 * g.x)),
         ("sin 4x", np.sin(4 * g.x)),
         ("sin x + 0.5 sin 2x", np.sin(g.x) + 0.5 * np.sin(2 * g.x)),
         ("sin 2x + 0.5 sin x", np.sin(2 * g.x) + 0.5 * np.sin(g.x)),
         ("sin 2x + 0.1 sin x", np.sin(2 * g.x) + 0.1 * np.sin(g.x)),
         ("sin 3x + 0.4 sin 4x", np.sin(3 * g.x) + 0.4 * np.sin(4 * g.x))]
for name, w0 in cases:
    r = classify(w0)
    if r["kind"] == "blowup":
        mu = ", ".join(f"{m:.3f}" for m in r["mus"][:4])
        print(f"  {name:>22}  {r['sc']:10d}  {r['member']:7d}  "
              f"{r['amp_f']:11.6f}  {mu:>20}")
    else:
        print(f"  {name:>22}  {r['sc']:10d}  {r['kind']:>7}")

print()
print("=" * 74)
print("C. Random smooth data")
print("=" * 74)
print("  Modes 1 to 8 with amplitudes falling like 1/k and random phases.")
print()
print(f"  {'#':>3}  {'sign chgs':>10}  {'outcome':>12}  {'member':>7}  "
      f"{'||f||_inf':>11}")
print("  " + "-" * 50)
rng = np.random.default_rng(11)
tally = {}
for trial in range(16):
    coef = np.zeros(g.N // 2 + 1, dtype=complex)
    for k in range(1, 9):
        coef[k] = (rng.normal() + 1j * rng.normal()) / k
    w0 = np.fft.irfft(coef * g.N / 2.0, g.N)
    r = classify(w0)
    key = (r["kind"], r.get("member"))
    tally[key] = tally.get(key, 0) + 1
    if r["kind"] == "blowup":
        print(f"  {trial:3d}  {r['sc']:10d}  {'blowup':>12}  "
              f"{r['member']:7d}  {r['amp_f']:11.6f}")
    else:
        print(f"  {trial:3d}  {r['sc']:10d}  {r['kind']:>12}")

print()
print("  tally:")
for key, count in sorted(tally.items(), key=lambda kv: -kv[1]):
    label = f"{key[0]}" + (f", member {key[1]}" if key[1] else "")
    print(f"    {label:>28}  {count:3d}")
