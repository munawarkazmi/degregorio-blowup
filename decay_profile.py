"""
What is the decaying solution decaying to?

If w ~ g(x)/t as t grows, then w_t = -g/t^2 while R(w) = R(g)/t^2, so

    R(g) = -g

and since R is quadratic, R(-g) = R(g) = -g, so h = -g satisfies R(h) = h.
The decay attractor is a solution of the same profile equation, entered with
the opposite sign. Its amplitude is what ||w||_inf t converges to, measured at
2.97, and that matches neither member 1 at 4.148482 nor member 2 at 4.161220.
So it should be a profile not yet seen.

Its stability follows too, and inverts. Writing s = log t and W = t w gives
W_s = R(W) + W, whose linearisation about g is DR(g) + I. With h = -g and DR
linear in its argument, DR(g) = -DR(h), so

    J_decay = -( DR(h) - I ) = -J_blowup

The eigenvalues are those of the blowup problem with the sign flipped. For this
profile to attract decaying solutions, it must therefore be strongly unstable
as a blowup profile: every eigenvalue that is stable here has to be unstable
there, apart from the two forced modes at 1 and 0, which map to -1 and 0.

That is a sharp prediction and it is falsifiable. Extract h from the decaying
run, Newton it, and read the spectrum.

    python decay_profile.py
"""

import numpy as np

import dg
from profile_eq import newton, residual, sup

A = 0.8
C = 0.50
N = 2048


def decaying_run(c=C, n=N, t_max=400.0):
    g = dg.Grid(n)
    w0 = np.sin(g.x) + c * np.sin(2.0 * g.x)
    w0 = w0 / np.abs(w0).max()
    out = dg.run(w0, a=A, grid=g, t_max=t_max, cfl=0.02, tail_tol=1e-9,
                 w_max=1e4)
    return g, out


print()
print("=" * 74)
print("Part 1: extract h = -t w from the decaying solution")
print("=" * 74)

g, out = decaying_run()
print(f"  c = {C}: stop {out['reason']} at t = {out['t']:.2f}, "
      f"||w||_inf = {out['winf_hist'][-1]:.5e}")
print(f"  ||w||_inf * t = {out['winf_hist'][-1] * out['t']:.6f}")

h0 = -out["w"] * out["t"]
raw = float(np.abs(residual(g, h0, A)).max())
print()
print(f"  residual of h = -t w in R(h) = h, before Newton: {raw:.3e}")
print(f"  ||h||_inf = {sup(g, h0):.7f}")

h, hist, ok = newton(g, h0, A)
print()
print("  Newton residuals:", "  ".join(f"{r:.2e}" for r in hist))
print(f"  converged: {ok}")
if ok:
    print(f"  ||h||_inf after Newton = {sup(g, h):.7f}")
    print(f"  compare member 1 = 4.148482, member 2 = 4.161220")

if not ok or sup(g, h) < 1e-6:
    raise SystemExit("  Newton did not find a nontrivial profile.")

print()
print("=" * 74)
print("Part 2: how many stagnation points, and what exponents?")
print("=" * 74)


def evaluate(gr, field, x):
    fh = gr.fwd(field)
    c = 2.0 * fh / gr.N
    c[0] = fh[0] / gr.N
    c[-1] = fh[-1] / gr.N
    return float(np.real(np.sum(c * np.exp(1j * gr.k * x))))


U, Hf, _ = dg.fields(g, h)
idx = np.flatnonzero(np.sign(U) != np.sign(np.roll(U, -1)))
print(f"  {len(idx)} stagnation points")
print()
print(f"  {'y*':>10}  {'c':>12}  {'mu':>10}  {'beta':>10}")
print("  " + "-" * 48)
for i in idx:
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
    mu = (c - 1.0) / (A * c)
    print(f"  {x % (2 * np.pi):10.6f}  {c:12.6f}  {mu:10.6f}  {mu + 1:10.6f}")
print()
print(f"  exact c1 for a simple zero is 1/(1-a) = {1 / (1 - A):.6f}")

print()
print("=" * 74)
print("Part 3: the spectrum, and the predicted inversion")
print("=" * 74)

from profile_eq import jacobian  # noqa: E402

gs = dg.Grid(512)
# Carry h down to a grid where a dense eigensolve is cheap.
hh = np.fft.rfft(h)
small = np.zeros(gs.N // 2 + 1, dtype=complex)
m = min(len(hh), len(small))
small[:m] = hh[:m]
hs = np.fft.irfft(small * gs.N / g.N, gs.N)
hs, hist_s, ok_s = newton(gs, hs, A)
print(f"  re-solved at N = {gs.N}: converged {ok_s}, "
      f"residual {hist_s[-1]:.2e}, ||h||_inf = {sup(gs, hs):.6f}")

vals = np.linalg.eigvals(jacobian(gs, hs, A))
vals = vals[np.argsort(-vals.real)]
print()
print("  leading eigenvalues of J_blowup = D[R(.)-.](h):")
print(f"  {'Re':>12}  {'Im':>12}")
for v in vals[:10]:
    print(f"  {v.real:12.6f}  {v.imag:12.6f}")

forced = (np.abs(vals - 1.0) < 1e-6) | (np.abs(vals) < 1e-6)
rest = vals[~forced]
n_unstable = int((rest.real > 1e-6).sum())
print()
print(f"  eigenvalues at exactly 1 or 0 (forced): {int(forced.sum())}")
print(f"  unstable directions beyond those:       {n_unstable}")
print()
print("  For this profile to attract decaying solutions, J_decay = -J_blowup")
print("  must be stable, so J_blowup must have no eigenvalue with negative")
print("  real part apart from the forced ones. A count of stable directions")
print("  here is what would refute the picture.")
n_stable = int((rest.real < -1e-6).sum())
print(f"  stable directions of J_blowup beyond the forced ones: {n_stable}")
