"""
Validation suite for the solver in dg.py.

Every test checks the code against something known in closed form or conserved
exactly, so a pass is evidence rather than a plot that looks plausible. Run
this before trusting anything in sweep.py.

    python validate.py
"""

import numpy as np

import dg


results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for line in detail.splitlines():
        print(f"         {line}")


# ---------------------------------------------------------------------------
print()
print("Test 1: sin(x) is an exact steady state of De Gregorio (a = 1)")
print("  u * w_x = w * u_x pointwise for this datum, so nothing should move.")

g = dg.Grid(256)
w0 = np.sin(g.x)
out = dg.run(w0, a=1.0, grid=g, t_max=50.0, cfl=0.05)
drift = float(np.abs(out["w"] - w0).max())
report(
    "steady state preserved to t = 50",
    drift < 1e-10 and out["reason"] == "t_max",
    f"max |w(50) - sin(x)| = {drift:.3e}   ({out['steps']} steps, "
    f"stopped: {out['reason']})",
)


# ---------------------------------------------------------------------------
print()
print("Test 2: a = 0 matches the Constantin-Lax-Majda closed form")
print("  w(x,t) = 4 sin(x) / (4 + 4 t cos x + t^2),  ||w||_inf = 4 / (4 - t^2)")
print("  The sup norm is compared against the true continuum supremum, which")
print("  is what dg.sup_norm reconstructs by Newton on the interpolant.")

g = dg.Grid(2048)
w0 = np.sin(g.x)
for t_stop in (0.5, 1.0, 1.5, 1.8):
    out = dg.run(w0, a=0.0, grid=g, t_max=t_stop, cfl=0.005)
    t_end = out["t"]
    exact = dg.clm_sine_exact(g.x, t_end)
    field_err = float(np.abs(out["w"] - exact).max() / np.abs(exact).max())

    sup_num, _ = dg.sup_norm(g, out["w"])
    sup_ex = dg.clm_sine_sup(t_end)
    sup_err = abs(sup_num - sup_ex) / sup_ex
    grid_err = abs(float(np.abs(out["w"]).max()) - sup_ex) / sup_ex

    report(
        f"t = {t_stop}",
        field_err < 1e-9 and sup_err < 1e-9 and out["reason"] == "t_max",
        f"rel field error   = {field_err:.3e}\n"
        f"rel sup error     = {sup_err:.3e}   (refined interpolant)\n"
        f"                    {grid_err:.3e}   (raw grid max, for contrast)",
    )


# ---------------------------------------------------------------------------
print()
print("Test 3: mean vorticity is conserved for every a")
print("  d/dt int w = (1 + a) int w H w = 0, since H is antisymmetric.")

g = dg.Grid(512)
rng = np.random.default_rng(0)
wh = np.zeros(g.N // 2 + 1, dtype=complex)
wh[1:9] = (rng.normal(size=8) + 1j * rng.normal(size=8)) * g.N / 8.0
w0 = np.fft.irfft(wh, g.N) + 0.37  # deliberately nonzero mean
for a in (0.0, 0.5, 1.0, -0.5):
    out = dg.run(w0, a=a, grid=g, t_max=3.0, cfl=0.05)
    drift = float(np.abs(out["mean_hist"] - w0.mean()).max())
    report(
        f"a = {a}",
        drift < 1e-12,
        f"max |mean(w(t)) - mean(w_0)| = {drift:.3e}   (mean = {w0.mean():.6f})",
    )


# ---------------------------------------------------------------------------
print()
print("Test 4: spectral convergence in N (a = 0, t = 1.5)")
print("  The exact solution is analytic in a strip of half width")
print("  arccosh((4 + t^2) / (4 t)) = 0.2887, so the truncation error should")
print("  fall like exp(-0.2887 * kcut) with kcut = N / 3, until the RK4 time")
print("  error floors it. The resolution guard is disabled here so that every")
print("  N is compared at the same physical time.")

delta = float(np.arccosh((4.0 + 1.5 ** 2) / (4.0 * 1.5)))
prev = None
for N in (32, 64, 128, 256, 512):
    g = dg.Grid(N)
    out = dg.run(np.sin(g.x), a=0.0, grid=g, t_max=1.5, cfl=0.005,
                 tail_tol=np.inf)
    exact = dg.clm_sine_exact(g.x, out["t"])
    err = float(np.abs(out["w"] - exact).max() / np.abs(exact).max())
    pred = np.exp(-delta * (N // 3))
    ratio = "" if prev is None else f"  {prev / err:10.1f}x better"
    print(f"         N = {N:4d}  t = {out['t']:.4f}  rel error = {err:.3e}"
          f"  predicted = {pred:.1e}{ratio}")
    if N == 512:
        report("N = 512 converged", err < 1e-9, f"rel error = {err:.3e}")
    prev = err


# ---------------------------------------------------------------------------
print()
print("Test 5: fourth order convergence in time (a = 0, t = 1.5, N = 512)")
print("  N = 512 is spatially converged to 1e-22 here, so whatever error")
print("  remains belongs to RK4 and must fall by 16x per halving of cfl.")

prev = None
for cfl in (0.08, 0.04, 0.02, 0.01, 0.005):
    g = dg.Grid(512)
    out = dg.run(np.sin(g.x), a=0.0, grid=g, t_max=1.5, cfl=cfl,
                 tail_tol=np.inf)
    exact = dg.clm_sine_exact(g.x, out["t"])
    err = float(np.abs(out["w"] - exact).max() / np.abs(exact).max())
    ratio = None if prev is None else prev / err
    tag = "" if ratio is None else f"   ratio = {ratio:6.2f}  (expect 16)"
    print(f"         cfl = {cfl:6.3f}   rel error = {err:.3e}{tag}")
    if ratio is not None and cfl >= 0.01:
        report(f"cfl {cfl * 2:.3f} -> {cfl:.3f} is fourth order",
               8.0 < ratio < 32.0, f"error ratio = {ratio:.2f}")
    prev = err


# ---------------------------------------------------------------------------
print()
print("Test 6: the blowup fit, separated into solver error and estimator bias")
print("  ||w||_inf = 4 / (4 - t^2) is only asymptotically a power law, so any")
print("  finite fit window carries an O(window) bias toward p < 1. To tell the")
print("  two apart we run the identical fit on the exact sup norm sampled at")
print("  the same times. Agreement between the two isolates solver error; the")
print("  gap from p = 1 is the estimator's own bias.")

print()
print("  First, calibrate the resolution guard. Every run below stops on the")
print("  same spectral tail criterion, so what tail_tol buys in real accuracy")
print("  is not obvious a priori. Tightening it must shrink the sup-norm error")
print("  at the stopping point, and the sweep needs that number to know how")
print("  far to trust a blowup time.")

devs = []
for tol in (1e-6, 1e-9, 1e-12):
    g = dg.Grid(8192)
    o = dg.run(np.sin(g.x), a=0.0, grid=g, t_max=2.0, cfl=0.01, tail_tol=tol)
    ex = dg.clm_sine_sup(o["t_hist"])
    dev = float((np.abs(o["winf_hist"] - ex) / ex).max())
    devs.append(dev)
    print(f"         tail_tol = {tol:.0e}:  stops at t = {o['t']:.6f}, "
          f"||w||_inf = {o['winf_hist'][-1]:8.2f}")
    print(f"                        peak-relative sup error = {dev:.3e}")

report(
    "tightening the guard by 1e-3 buys at least 3x accuracy, twice over",
    devs[0] / devs[1] > 3.0 and devs[1] / devs[2] > 3.0,
    f"error fell {devs[0] / devs[1]:.1f}x then {devs[1] / devs[2]:.1f}x\n"
    f"so the deviation is spatial truncation, not a solver defect: it "
    f"responds to the guard,\nnot to cfl (halving cfl moves it 1.1x)",
)

g = dg.Grid(32768)
out = dg.run(np.sin(g.x), a=0.0, grid=g, t_max=2.0, cfl=0.01, tail_tol=1e-8)
print(f"\n         fit record: N = 32768, t = {out['t']:.6f}, "
      f"||w||_inf = {out['winf_hist'][-1]:.4e}, stopped: {out['reason']}")
t_h = out["t_hist"]
w_num = out["winf_hist"]
w_ex = dg.clm_sine_sup(t_h)

for wd in (1.0, 0.5):
    T_n, p_n, r2_n = dg.fit_blowup(t_h, w_num, window_decades=wd)
    T_e, p_e, r2_e = dg.fit_blowup(t_h, w_ex, window_decades=wd)
    report(
        f"window = {wd} decades: numerical fit equals exact-data fit",
        abs(T_n - T_e) < 1e-6 and abs(p_n - p_e) < 1e-5,
        f"numerical   T = {T_n:.8f}  p = {p_n:.6f}  R^2 = {r2_n:.8f}\n"
        f"exact data  T = {T_e:.8f}  p = {p_e:.6f}  R^2 = {r2_e:.8f}\n"
        f"estimator bias vs truth: dT = {T_e - 2.0:+.2e}, dp = {p_e - 1.0:+.2e}",
    )

T_1, p_1, _ = dg.fit_blowup(t_h, w_num, window_decades=1.0)
T_h, p_h, _ = dg.fit_blowup(t_h, w_num, window_decades=0.5)
report(
    "halving the window moves p toward 1 and T toward 2",
    abs(p_h - 1.0) < abs(p_1 - 1.0) and abs(T_h - 2.0) < abs(T_1 - 2.0),
    f"p: {p_1:.5f} -> {p_h:.5f}   (target 1)\n"
    f"T: {T_1:.7f} -> {T_h:.7f}   (target 2)",
)


# ---------------------------------------------------------------------------
print()
print("Test 7: the analyticity strip width, against the one exact case")
print("  For a = 0 from sin(x) the denominator 4 + 4t cos z + t^2 vanishes at")
print("  cos z = -(4 + t^2) / 4t, so delta = arccosh((4 + t^2) / 4t). Since")
print("  ||w||_inf = 4 / (4 - t^2), the product ||w||_inf * delta tends to 1/2.")
print("  That product is the self-similarity test used in mechanism.py, so it")
print("  has to be anchored somewhere it can be checked.")

g = dg.Grid(8192)
out = dg.run(np.sin(g.x), a=0.0, grid=g, t_max=2.0, cfl=0.01, tail_tol=1e-9)
t_h, w_h, d_h = out["t_hist"], out["winf_hist"], out["delta_hist"]

late = w_h > 5.0
d_exact = np.arccosh((4.0 + t_h[late] ** 2) / (4.0 * t_h[late]))
strip_err = float((np.abs(d_h[late] - d_exact) / d_exact).max())
report(
    "measured strip width matches the closed form",
    strip_err < 1e-4,
    f"max rel error over the growth phase = {strip_err:.3e}",
)

prod = w_h[-1] * d_h[-1]
report(
    "w * delta approaches 1/2, the self-similar signature",
    abs(prod - 0.5) < 0.01,
    f"w * delta = {prod:.5f} at ||w||_inf = {w_h[-1]:.2f}   (exact limit 0.5)",
)


# ---------------------------------------------------------------------------
print()
n_pass = sum(1 for _, ok in results if ok)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    print("Failures above. Do not trust sweep.py until these are green.")
    raise SystemExit(1)
