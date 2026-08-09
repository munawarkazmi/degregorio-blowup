"""
Solve the blowup profile equation directly, instead of inferring it from a
simulation that is running away.

Substituting w(x, t) = f(x - x_p) / (T - t) into w_t = w u_x - a u w_x gives
f / (T-t)^2 on the left and rhs(f) / (T-t)^2 on the right, so the profile
equation is simply

    rhs(f) = f,        rhs(f) = f H f - a U f',   U' = H f, mean U = 0

a fixed point of the very nonlinearity the solver already validates. Two
consequences make this worth doing.

  No free scaling. If f solves it then cf gives c^2 rhs(f) = c^2 f, which
  equals cf only when c = 1. The equation fixes the amplitude outright, unlike
  a linear eigenvalue problem. So ||f||_inf is a prediction, not a fitted
  constant.

  That amplitude is measurable independently. From ||w|| = ||f|| / (T - t),

      d/dt (1 / ||w||) = -1 / ||f||_inf

  so the slope of the reciprocal amplitude in a simulation must equal minus the
  reciprocal of the peak height of the solved profile. Nothing is shared
  between the two computations except the value of a.

The only degeneracy is translation, since the equation commutes with shifts,
which leaves the Jacobian with a null vector f'. Handled by regularising with
that direction rather than by imposing a phase condition.

    python profile_solve.py
"""

import numpy as np

import dg

N = 2048
A_REF = 0.8


def reciprocal_slope(out, decades=1.0):
    """
    Measured d/dt (1 / ||w||) over the last `decades` decades of growth. This
    is the quantity the profile equation predicts as -1 / ||f||_inf.
    """
    t, w = out["t_hist"], out["winf_hist"]
    thr = w[-1] * 10.0 ** (-decades)
    below = np.flatnonzero(w < thr)
    i0 = int(below[-1]) + 1 if below.size else 0
    return float(np.polyfit(t[i0:], 1.0 / w[i0:], 1)[0])


def simulate(a, n=N, w_max=1e6):
    g = dg.Grid(n)
    out = dg.run(np.sin(g.x), a=a, grid=g, t_max=60.0, cfl=0.02,
                 tail_tol=1e-9, w_max=w_max)
    return g, out


def guess_from_simulation(g, out):
    """
    f = ||f||_inf * (w / ||w||_inf), with ||f||_inf read off the reciprocal
    slope. Rolled so the peak sits at index 0, purely to keep successive
    continuation steps from sliding around.
    """
    w = out["w"]
    m = reciprocal_slope(out)
    if m >= 0:
        return None, np.nan
    f = w / np.abs(w).max() * (-1.0 / m)
    return np.roll(f, -int(np.argmax(np.abs(f)))), -1.0 / m


def residual(g, f, a):
    return dg.rhs(g, f, a) - f


def jacobian(g, f, a):
    """
    Exact Jacobian of residual(), assembled a column at a time through the same
    masked transforms rhs() uses, so the linearisation matches the residual
    including dealiasing. Six transforms per column beats forming the dense
    multiplier matrices and multiplying them.
    """
    u_f, ux_f, fx_f = dg.fields(g, f)
    J = np.empty((g.N, g.N))
    e = np.zeros(g.N)
    for j in range(g.N):
        e[j] = 1.0
        u_d, ux_d, dx_d = dg.fields(g, e)
        J[:, j] = g.bwd(g.fwd(e * ux_f + f * ux_d
                              - a * (u_d * fx_f + u_f * dx_d))) - e
        e[j] = 0.0
    return J


def newton(g, f, a, tol=1e-11, max_iter=25, verbose=False):
    """
    Newton on residual(f) = 0, regularised along the translation mode.

    The Jacobian is singular because shifting a solution gives another
    solution, so J f' = 0. Adding eps * p p^T with p the normalised f' moves
    that one eigenvalue off zero and leaves everything else alone, which is
    cheaper and better conditioned than a bordered system or an SVD.
    """
    hist = []
    for _ in range(max_iter):
        r = residual(g, f, a)
        rn = float(np.abs(r).max())
        hist.append(rn)
        if rn < tol:
            return f, hist, True
        if not np.isfinite(rn) or rn > 1e8:
            return f, hist, False

        J = jacobian(g, f, a)
        p = g.bwd(g.fwd(f) * g.dmul)
        pn = float(np.linalg.norm(p))
        if pn > 0:
            p = p / pn
            J += np.outer(p, p)
        try:
            f = f - np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return f, hist, False
    return f, hist, float(np.abs(residual(g, f, a)).max()) < tol


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print(f"Part 1: does the a = {A_REF} frozen profile solve rhs(f) = f?")
print("=" * 74)

g, out = simulate(A_REF)
f0, amp_sim = guess_from_simulation(g, out)
print(f"  simulation: ||w|| = {out['winf_hist'][-1]:.3e} at t = {out['t']:.6f}, "
      f"stop = {out['reason']}")
print(f"  reciprocal slope gives ||f||_inf = {amp_sim:.9f}")
print(f"  residual of the raw simulation profile: "
      f"{np.abs(residual(g, f0, A_REF)).max():.3e}")
print()

f, hist, ok = newton(g, f0, A_REF)
print("  Newton residual by iteration:")
for i, h in enumerate(hist):
    print(f"      {i}: {h:.3e}")
print(f"  converged: {ok}")

if ok:
    amp_prof = float(np.abs(f).max())
    drift = float(np.abs(f - f0).max() / np.abs(f).max())
    print()
    print(f"  ||f||_inf from the profile equation : {amp_prof:.9f}")
    print(f"  ||f||_inf from the simulation slope : {amp_sim:.9f}")
    print(f"  relative disagreement               : "
          f"{abs(amp_prof - amp_sim) / amp_prof:.3e}")
    print(f"  how far Newton moved the simulated profile: {drift:.3e}")


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("Part 2: continuation in a. Where does the frozen profile exist?")
print("=" * 74)
print("  Stepping away from a = 0.8 in both directions, each solve started")
print("  from the previous one. A profile that ceases to exist should show up")
print("  as Newton failing to converge, not as a bad number.")
print()
print(f"  {'a':>6}  {'converged':>10}  {'||f||_inf':>14}  {'iters':>6}  "
      f"{'residual':>11}")
print("  " + "-" * 56)

branch = {}
for direction in (-1, +1):
    f_c = f.copy()
    a_c = A_REF
    while True:
        a_c = round(a_c + direction * 0.05, 4)
        if not (0.0 <= a_c <= 1.0):
            break
        f_new, h, good = newton(g, f_c, a_c)
        print(f"  {a_c:6.2f}  {str(good):>10}  "
              f"{float(np.abs(f_new).max()) if good else float('nan'):14.6f}  "
              f"{len(h):6d}  {h[-1]:11.3e}")
        if not good:
            break
        branch[a_c] = float(np.abs(f_new).max())
        f_c = f_new
branch[A_REF] = float(np.abs(f).max())


# ---------------------------------------------------------------------------
print()
print("=" * 74)
print("Part 3: the falsifiable bit")
print("=" * 74)
print("  The profile equation predicts d/dt (1/||w||) = -1 / ||f||_inf.")
print("  Nothing but the value of a is shared between the two columns.")
print()
print(f"  {'a':>6}  {'predicted':>13}  {'measured':>13}  {'rel diff':>10}")
print("  " + "-" * 48)

for a_t in sorted(k for k in branch if abs(k * 100 - round(k * 100)) < 1e-6):
    if round(a_t * 100) % 10 != 0:
        continue
    _, out_t = simulate(a_t)
    m = reciprocal_slope(out_t)
    pred = -1.0 / branch[a_t]
    print(f"  {a_t:6.2f}  {pred:13.9f}  {m:13.9f}  "
          f"{abs(pred - m) / abs(pred):10.2e}")
