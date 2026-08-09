"""
The blowup profile equation and its Newton solver.

Substituting w(x, t) = f(x - x_p) / (T - t) into w_t = w u_x - a u w_x gives
f / (T-t)^2 on the left and rhs(f) / (T-t)^2 on the right, so the profile
equation is simply

    rhs(f) = f,        rhs(f) = f H f - a U f',   U' = H f, mean U = 0

a fixed point of the very nonlinearity dg.py already validates. Two
consequences make this worth solving directly.

  No free scaling. If f solves it then cf gives c^2 rhs(f) = c^2 f, which
  equals cf only when c = 1. The equation fixes the amplitude outright, unlike
  a linear eigenvalue problem, so ||f||_inf is a prediction rather than a
  fitted constant.

  That amplitude is measurable independently. From ||w|| = ||f|| / (T - t),

      d/dt (1 / ||w||) = -1 / ||f||_inf

  so the reciprocal amplitude slope of a simulation must equal minus the
  reciprocal peak height of the solved profile. The two computations share
  nothing but the value of a.

Caution: f = 0 solves the equation exactly and says nothing. Any use of these
routines has to screen for it.
"""

import numpy as np

import dg


def sup(g, f):
    """
    Sup of the interpolant. Use this everywhere a peak height is compared.

    The grid maximum is wrong by O(dx^2 |f''|), and these profiles are narrow:
    at a = 0.8 the strip width is about 0.019, so |f''| ~ ||f|| / delta^2 is of
    order 1e4 and the grid max is off by a part in 1e3 at N = 2048. Worse, the
    error depends on where the peak falls between grid points, so it does not
    shrink cleanly with N and it masquerades as the profile itself changing.
    """
    return dg.sup_norm(g, f)[0]


def reciprocal_slope(out, decades=1.0):
    """
    Measured d/dt (1 / ||w||) over the last `decades` decades of growth, the
    quantity the profile equation predicts as -1 / ||f||_inf.
    """
    t, w = out["t_hist"], out["winf_hist"]
    thr = w[-1] * 10.0 ** (-decades)
    below = np.flatnonzero(w < thr)
    i0 = int(below[-1]) + 1 if below.size else 0
    if len(t) - i0 < 8:
        return np.nan
    return float(np.polyfit(t[i0:], 1.0 / w[i0:], 1)[0])


def simulate(a, n=2048, w_max=1e6, t_max=60.0, cfl=0.02, tail_tol=1e-9):
    g = dg.Grid(n)
    out = dg.run(np.sin(g.x), a=a, grid=g, t_max=t_max, cfl=cfl,
                 tail_tol=tail_tol, w_max=w_max)
    return g, out


def guess_from_simulation(g, out):
    """
    f = ||f||_inf * (w / ||w||_inf), with ||f||_inf read off the reciprocal
    slope. Rolled so the peak sits at index 0, purely to stop successive solves
    sliding along the translation mode.
    """
    w = out["w"]
    m = reciprocal_slope(out)
    if not np.isfinite(m) or m >= 0:
        return None, np.nan
    # winf_hist is built from sup_norm, so normalise by the same measure or the
    # comparison mixes two different definitions of peak height.
    f = w / sup(g, w) * (-1.0 / m)
    return np.roll(f, -int(np.argmax(np.abs(f)))), -1.0 / m


def residual(g, f, a):
    return dg.rhs(g, f, a) - f


def jacobian(g, f, a):
    """
    Exact Jacobian of residual(), assembled a column at a time through the same
    masked transforms rhs() uses, so the linearisation matches the residual
    including dealiasing. Six transforms per column beats forming the dense
    multiplier matrices and multiplying them out.
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


def newton(g, f, a, tol=1e-11, max_iter=25):
    """
    Newton on residual(f) = 0, regularised along the translation mode.

    The Jacobian is singular because shifting a solution gives another
    solution, so J f' = 0. Adding eps * p p^T with p the normalised f' moves
    that one eigenvalue off zero and leaves the rest alone, which is cheaper
    and better conditioned than a bordered system or an SVD.

    Returns (f, residual_history, converged).
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
