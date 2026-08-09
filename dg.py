"""
Okamoto-Sakajo-Wunsch family of 1D models for the 3D Euler vorticity equation,
posed on the circle x in [0, 2*pi).

    w_t + a * u * w_x = w * u_x
    u_x = H w,   with u of zero mean

H is the Hilbert transform. The parameter a interpolates between two named
models:

    a = 0   Constantin-Lax-Majda (1985).  Exactly solvable, blows up.
    a = 1   De Gregorio (1990).  Advection restored at full strength.

Fourier conventions. We store real fields on N equispaced points and use
numpy's rfft, so wavenumbers run k = 0, 1, ..., N/2 and the negative half is
implied by conjugate symmetry. The three multipliers we need are

    (H w)^_k   = -i * sgn(k) * w^_k
    u^_k       = -w^_k / |k|          for k != 0,   u^_0 = 0
    (w_x)^_k   = i * k * w^_k

The velocity multiplier follows from u_x = H w: taking i*k*u^_k = -i*sgn(k)*w^_k
gives u^_k = -w^_k / |k|.

Sanity anchor for the conventions: with w = sin(x) we get H w = -cos(x),
u = -sin(x), and therefore u * w_x = w * u_x pointwise, so sin(x) is an exact
steady state of the a = 1 model. Test 1 in validate.py checks this numerically.

Nonlinear products are formed in physical space with 2/3 dealiasing. Time
stepping is classical RK4 with a step size tied to the current stretching rate,
which is what keeps the integration honest as the solution steepens.
"""

import numpy as np


class Grid:
    """Spectral operators on N equispaced points over [0, 2*pi)."""

    def __init__(self, N):
        if N % 2 != 0:
            raise ValueError("N must be even")
        self.N = N
        self.x = 2.0 * np.pi * np.arange(N) / N
        self.dx = 2.0 * np.pi / N

        k = np.arange(N // 2 + 1)
        self.k = k

        # d/dx
        self.dmul = 1j * k

        # Hilbert transform. Zero at k = 0 and at Nyquist, where sgn is
        # ambiguous for a real-valued field.
        self.hmul = -1j * np.ones(k.shape, dtype=complex)
        self.hmul[0] = 0.0
        self.hmul[-1] = 0.0

        # Velocity from vorticity
        self.umul = np.zeros(k.shape, dtype=complex)
        self.umul[1:] = -1.0 / k[1:]
        self.umul[-1] = 0.0

        # 2/3 dealiasing mask
        self.kcut = N // 3
        self.mask = (k <= self.kcut).astype(float)
        self.mask[-1] = 0.0

    def fwd(self, f):
        return np.fft.rfft(f) * self.mask

    def bwd(self, fh):
        return np.fft.irfft(fh, self.N)

    def hilbert(self, f):
        return self.bwd(self.fwd(f) * self.hmul)


def fields(g, w, wh=None):
    """Return (u, u_x, w_x) for a given vorticity field."""
    if wh is None:
        wh = g.fwd(w)
    u = g.bwd(wh * g.umul)
    ux = g.bwd(wh * g.hmul)
    wx = g.bwd(wh * g.dmul)
    return u, ux, wx


def rhs(g, w, a):
    """Right hand side of w_t = w * u_x - a * u * w_x."""
    u, ux, wx = fields(g, w)
    return g.bwd(g.fwd(w * ux - a * u * wx))


def rk4_step(g, w, dt, a):
    k1 = rhs(g, w, a)
    k2 = rhs(g, w + 0.5 * dt * k1, a)
    k3 = rhs(g, w + 0.5 * dt * k2, a)
    k4 = rhs(g, w + dt * k3, a)
    return w + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def sup_norm(g, w, iters=8, wh=None):
    """
    Sup norm of the spectral interpolant, not of the grid samples.

    The grid maximum is wrong by O(dx^2 * |w''|) because the peak generally
    falls between grid points, and near blowup the peak narrows like 1 / ||w||
    so that error grows exactly when we care most. Both the blowup time and the
    rate exponent inherit the bias.

    Fix: start at the grid argmax and Newton on w'(x) = 0, evaluating the
    trigonometric series directly at an arbitrary x. Costs O(N) per iteration
    for a single point, which is nothing next to the transforms.

    Returns (value, location).
    """
    if wh is None:
        wh = g.fwd(w)
    c = 2.0 * wh / g.N
    c[0] = wh[0] / g.N
    c[-1] = wh[-1] / g.N
    k = g.k.astype(float)

    x = float(g.x[int(np.argmax(np.abs(w)))])
    for _ in range(iters):
        e = np.exp(1j * k * x)
        d1 = float(np.real(np.sum(c * (1j * k) * e)))
        d2 = float(np.real(np.sum(c * (-k * k) * e)))
        if d2 == 0.0:
            break
        # Cap the step at one cell so Newton cannot desert the local peak.
        s = float(np.clip(d1 / d2, -g.dx, g.dx))
        x -= s
        if abs(s) < 1e-15:
            break

    val = abs(float(np.real(np.sum(c * np.exp(1j * k * x)))))
    grid_val = float(np.abs(w).max())
    return (val, x) if val >= grid_val else (grid_val, float(g.x[int(np.argmax(np.abs(w)))]))


def tail_energy_fraction(g, w, wh=None):
    """
    Fraction of spectral energy sitting in the top quarter of the retained
    band. This is the resolution alarm: once it climbs off the floor, the grid
    can no longer represent the solution and every number after that point is
    fiction.
    """
    if wh is None:
        wh = g.fwd(w)
    p = np.abs(wh) ** 2
    total = p.sum()
    if total == 0.0:
        return 0.0
    lo = int(0.75 * g.kcut)
    return float(p[lo:g.kcut + 1].sum() / total)


def strip_width(g, w, wh=None):
    """
    Half width of the analyticity strip, from the exponential decay rate of the
    spectrum: |w_k| ~ exp(-delta * k).

    This is the physically meaningful width of the peak, and the product
    ||w||_inf * delta is the self-similarity test. For the a = 0 exact solution
    delta = arccosh((4 + t^2) / (4t)), which behaves like (2 - t) / 2 near
    blowup while ||w||_inf behaves like 1 / (2 - t), so the product tends to
    1/2. A product that stays constant means the peak narrows in proportion to
    its height; a product that grows means it does not.

    The fit window matters. Fitting over the whole upper band measures the
    roundoff plateau rather than the decay whenever the field is well resolved,
    since |w_k| bottoms out near 1e-16 of the peak and a fit through a flat
    floor returns delta near zero. So restrict to the decades that are genuinely
    decaying: below 1e-3 of the peak, above 1e-12 of it.

    Know the assumption before quoting the number. This is a single exponential
    fit, exact when the nearest complex singularity dominates, which is the
    case at a = 0 where it reproduces arccosh((4 + t^2) / 4t) to 1e-8. When the
    spectrum is curved in log space, several singularities at comparable
    depths, the straight line splits the difference and can be off by a factor
    of a few: at a = 0.8 it returns 0.0065 where the measured spectral tail
    implies about 0.019. Use it for trends and for the a = 0 control; for
    profile questions compare normalised spectra directly, as profile.py does,
    since that assumes nothing.
    """
    if wh is None:
        wh = g.fwd(w)
    amp = np.abs(wh)
    peak = amp.max()
    if peak <= 0.0:
        return np.nan
    k = g.k
    sel = (amp > peak * 1e-12) & (amp < peak * 1e-3) & (k > 0) & (k <= g.kcut)
    n = int(sel.sum())
    if n < 10:
        return np.nan
    # Least squares slope written out rather than via polyfit, which is called
    # once per step and whose setup cost dominates the arithmetic here.
    kk = k[sel].astype(float)
    ly = np.log(amp[sel])
    sk, sy = kk.sum(), ly.sum()
    den = n * (kk * kk).sum() - sk * sk
    if den == 0.0:
        return np.nan
    return float(-(n * (kk * ly).sum() - sk * sy) / den)


def timestep(g, w, ux, u, a, cfl, c_adv=1.0):
    """
    Step size, from two separate constraints.

    Accuracy: the stretching term w * u_x drives the growth, so the step must
    shrink like 1 / ||u_x|| as the solution sharpens. This is the binding one
    near blowup and it is what cfl controls.

    Stability: the advective term a * u * w_x has eigenvalues i * a * u * k out
    to k = kcut, and RK4 is stable on the imaginary axis only for
    |dt * lambda| < 2.828. So dt < 2.828 / (kcut * |a| * umax), and we take
    c_adv = 1 for a safety factor of about 2.8.

    Getting this second bound wrong is expensive in a way that looks like
    nothing: a step size of cfl * dx / umax, which is the naive finite
    difference instinct, is smaller than necessary by a factor of order
    2.828 / (cfl * kcut * dx) = 67 at cfl = 0.02, and it applies a penalty at
    a = 0 where the advection term is not even present.
    """
    s = max(np.abs(ux).max(), np.abs(w).max(), 1e-14)
    dt = cfl / s
    adv = abs(a) * float(np.abs(u).max())
    if adv > 1e-14:
        dt = min(dt, c_adv / (g.kcut * adv))
    return dt


def run(w0, a, N=None, t_max=1e4, cfl=0.05, tail_tol=1e-8, w_max=1e8,
        max_steps=2_000_000, sample_every=1, c_adv=1.0, grid=None,
        verbose=False):
    """
    Integrate the model from initial vorticity w0.

    Stops on whichever comes first: t_max reached, spectral tail exceeding
    tail_tol (under-resolved), ||w||_inf exceeding w_max, or max_steps.

    On w_max. The default of 1e8 is not timidity, it is where the clock runs
    out of precision. Near a power law singularity dt ~ cfl / ||w||, while t
    itself is order T, so the update t + dt loses all meaning once
    cfl / ||w|| falls below eps * T, which is around ||w|| = 1e13 for T of
    order 1. Long before that the recorded times are too coarsely spaced to
    fit. Eight decades of growth is far more than any blowup fit needs, and
    pushing further buys noise.

    Returns a dict with the final state and the diagnostic history.
    """
    g = grid if grid is not None else Grid(N if N is not None else len(w0))
    w = np.array(w0, dtype=float)

    hist_t, hist_winf, hist_mean, hist_tail, hist_dt = [], [], [], [], []
    hist_xpeak, hist_delta = [], []
    t = 0.0
    step = 0
    reason = "max_steps"

    while step < max_steps:
        wh = g.fwd(w)                       # shared by the diagnostics below
        u, ux, _ = fields(g, w, wh)
        winf = float(np.abs(w).max())

        if step % sample_every == 0:
            sup, xpeak = sup_norm(g, w, wh=wh)
            hist_t.append(t)
            hist_winf.append(sup)
            hist_xpeak.append(xpeak)
            hist_mean.append(float(w.mean()))
            hist_delta.append(strip_width(g, w, wh))
            tail = tail_energy_fraction(g, w, wh)
            hist_tail.append(tail)
            if tail > tail_tol:
                reason = "under-resolved"
                break
            if verbose and step % (sample_every * 500) == 0:
                print(f"    t={t:10.5f}  |w|_inf={sup:12.5e}  tail={tail:8.2e}")

        if winf > w_max:
            reason = "amplitude cap"
            break
        if t >= t_max:
            reason = "t_max"
            break

        dt = timestep(g, w, ux, u, a, cfl, c_adv)
        dt = min(dt, t_max - t)
        hist_dt.append(dt)
        w = rk4_step(g, w, dt, a)
        t += dt
        step += 1

        if not np.isfinite(w).all():
            reason = "non-finite"
            break

    return {
        "grid": g,
        "a": a,
        "w": w,
        "t": t,
        "steps": step,
        "reason": reason,
        "t_hist": np.array(hist_t),
        "winf_hist": np.array(hist_winf),
        "xpeak_hist": np.array(hist_xpeak),
        "delta_hist": np.array(hist_delta),
        "mean_hist": np.array(hist_mean),
        "tail_hist": np.array(hist_tail),
        "dt_hist": np.array(hist_dt),
    }


def fit_blowup(t, winf, window_decades=1.0, min_growth=10.0):
    """
    Estimate the blowup time T and rate exponent p in ||w||_inf ~ C (T - t)^-p.

    The trick is to avoid fitting T and p separately. If w = C (T - t)^-p then

        w / (dw/dt) = (T - t) / p,

    a straight line in t with slope -1/p and root T. One linear regression
    therefore gives both numbers, and the residual says whether the power law
    fits at all.

    The window matters more than the regression. A power law is only the
    leading asymptotics, so any window of finite width in t carries an O(width)
    bias. Selecting by amplitude rather than by record fraction keeps the window
    anchored to the late-time regime: we take the last `window_decades` decades
    of growth. Shrinking that number reduces bias and increases noise, and the
    honest way to quote a result is to report the answer for two window sizes.

    Returns (T, p, r2), with T = inf when the record shows no real growth.
    """
    t = np.asarray(t, dtype=float)
    w = np.asarray(winf, dtype=float)
    if len(t) < 8 or w[-1] < min_growth * max(w[0], 1e-300):
        return np.inf, np.nan, np.nan

    threshold = w[-1] * 10.0 ** (-window_decades)
    below = np.flatnonzero(w < threshold)
    i0 = int(below[-1]) + 1 if below.size else 0
    t, w = t[i0:], w[i0:]
    if len(t) < 8:
        return np.inf, np.nan, np.nan

    dwdt = np.gradient(w, t)
    good = dwdt > 0
    if good.sum() < 8:
        return np.inf, np.nan, np.nan
    t, w, dwdt = t[good], w[good], dwdt[good]

    y = w / dwdt
    m, c = np.polyfit(t, y, 1)
    if m >= -1e-12:
        return np.inf, np.nan, np.nan

    resid = y - (m * t + c)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else np.nan
    return float(-c / m), float(-1.0 / m), float(r2)


# ---------------------------------------------------------------------------
# Exact solutions used as ground truth in validate.py
# ---------------------------------------------------------------------------

def clm_exact(g, w0, t):
    """
    Exact solution of the a = 0 (Constantin-Lax-Majda) model.

    Setting z = H w + i w turns w_t = w H w into the Riccati equation
    z_t = z^2 / 2, whose solution z = 2 z_0 / (2 - t z_0) has imaginary part

        w(x, t) = 4 w_0(x) / [ (2 - t H w_0(x))^2 + t^2 w_0(x)^2 ].

    Blowup happens at the first time the denominator vanishes, which needs
    w_0 = 0 and H w_0 = 2 / t at the same point.
    """
    h0 = g.hilbert(w0)
    return 4.0 * w0 / ((2.0 - t * h0) ** 2 + (t * w0) ** 2)


def clm_sine_exact(x, t):
    """
    The a = 0 solution from w_0 = sin(x), in closed form.

    With H w_0 = -cos(x) the denominator above collapses:
    (2 + t cos x)^2 + t^2 sin^2 x = 4 + 4 t cos x + t^2.
    """
    return 4.0 * np.sin(x) / (4.0 + 4.0 * t * np.cos(x) + t * t)


def clm_sine_sup(t):
    """
    Sup norm of the above, in closed form: ||w(t)||_inf = 4 / (4 - t^2).

    Maximising 4 sin x / (4 + t^2 + 4 t cos x) gives cos x = -4t / (4 + t^2),
    hence the result. It diverges like 1 / (2 - t) as t -> 2, so T = 2 and the
    blowup rate exponent is exactly 1.
    """
    return 4.0 / (4.0 - t * t)
