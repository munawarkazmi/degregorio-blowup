"""
Confirm that the linearisation about the ground state is neutrally stable, and
draw the consequence for any reduction.

The dense eigensolve says the spectrum of M = DR_1(sin x) is purely imaginary,
with real parts at the 1e-5 roundoff level against imaginary parts up to 72.
That is a claim about a non-symmetric 512 by 512 matrix, so check it a second
way that does not go through eigenvalues at all: integrate v_t = M v and watch
the norm. Damping shows as decay, instability as growth, neutrality as neither
over long times.

The consequence matters more than the fact. A reduction of the escape from the
line of equilibria needs a spectral gap: fast modes slaved to slow ones, so
that the fast ones can be eliminated. Purely imaginary spectrum means there is
no gap anywhere, no mode is faster than any other in the sense that matters,
and nothing decays. Then

  - the quasi steady step v = eps A M^{-1} sin 2x / 2, which would give a
    Riccati equation and p = 1, has no justification, since the transient it
    neglects never dies;
  - truncation error does not relax away either, which is why cutting at m
    modes converges like a power of m rather than exponentially.

So the obstruction to a finite reduction is structural rather than technical.

    python neutral.py
"""

import numpy as np

import dg


def dR_apply(g, f, v, a):
    """Action of DR(f) on v, matching dg.rhs including the mask."""
    u_f, ux_f, fx_f = dg.fields(g, f)
    u_v, ux_v, vx = dg.fields(g, v)
    return g.bwd(g.fwd(v * ux_f + f * ux_v - a * (u_v * fx_f + u_f * vx)))


def rk4(g, f, v, dt, a):
    k1 = dR_apply(g, f, v, a)
    k2 = dR_apply(g, f, v + 0.5 * dt * k1, a)
    k3 = dR_apply(g, f, v + 0.5 * dt * k2, a)
    k4 = dR_apply(g, f, v + dt * k3, a)
    return v + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


g = dg.Grid(512)
f = np.sin(g.x)
rng = np.random.default_rng(3)

# Random initial perturbation, mean zero, orthogonal to the two kernel
# directions so that neither drifts the amplitude nor translates the state.
v = g.bwd(g.fwd(rng.normal(size=g.N)))
v -= v.mean()
for mode in (np.sin(g.x), np.cos(g.x)):
    v -= mode * (v @ mode) / (mode @ mode)
v /= np.abs(v).max()

print()
print("=" * 68)
print("Integrating v_t = M v from a random perturbation")
print("=" * 68)
print()
print(f"  {'t':>8}  {'L2 norm':>13}  {'sup norm':>13}  {'ratio to t=0':>13}")
print("  " + "-" * 54)

dt, t, n0 = 1.0e-3, 0.0, float(np.linalg.norm(v))
print(f"  {0.0:8.2f}  {n0:13.7f}  {np.abs(v).max():13.7f}  {1.0:13.7f}")
for target in (10.0, 50.0, 100.0, 200.0, 400.0):
    while t < target - 1e-12:
        step = min(dt, target - t)
        v = rk4(g, f, v, step, 1.0)
        t += step
    nrm = float(np.linalg.norm(v))
    print(f"  {t:8.2f}  {nrm:13.7f}  {np.abs(v).max():13.7f}  "
          f"{nrm / n0:13.7f}")

print()
print("  Bounded and non decaying over t = 400 is neutral stability. The")
print("  ground state is stable but not asymptotically stable, so there is no")
print("  spectral gap to build a reduction on.")

print()
print("=" * 68)
print("Is some quadratic form conserved?")
print("=" * 68)
print("  Neutral spectrum suggests M is skew in some inner product. Test the")
print("  weighted forms Q_s(v) = sum |k|^s |v_k|^2, which covers L2 at s = 0")
print("  and the homogeneous Sobolev norms either side of it.")
print()
print(f"  {'s':>6}  {'Q at t=0':>13}  {'Q at t=400':>13}  {'rel change':>12}")
print("  " + "-" * 50)

v0 = g.bwd(g.fwd(rng.normal(size=g.N)))
v0 -= v0.mean()
for mode in (np.sin(g.x), np.cos(g.x)):
    v0 -= mode * (v0 @ mode) / (mode @ mode)
v0 /= np.abs(v0).max()

vT, t = v0.copy(), 0.0
while t < 400.0 - 1e-12:
    step = min(dt, 400.0 - t)
    vT = rk4(g, f, vT, step, 1.0)
    t += step

k = g.k.astype(float)
for s in (-2.0, -1.0, 0.0, 1.0, 2.0):
    wgt = np.where(k > 0, k ** s, 0.0)
    q0 = float((wgt * np.abs(g.fwd(v0)) ** 2).sum())
    qT = float((wgt * np.abs(g.fwd(vT)) ** 2).sum())
    print(f"  {s:6.1f}  {q0:13.6e}  {qT:13.6e}  {(qT - q0) / q0:12.3e}")

print()
print("  A relative change at roundoff identifies the conserved form and with")
print("  it the inner product in which M is skew.")
