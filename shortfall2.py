"""
The shortfall is a fit window artefact. This is the check that proves it.

shortfall.py found the local slope falls monotonically through the predicted
value rather than converging to it: 3.164, 3.125, 3.046, 3.030, 3.003, 2.973
over octave bands from k = 8 up to 1024, against a prediction of 3.001668. Low
k is biased steep and high k biased shallow, and a fit spanning both lands in
between, which is where 2.976 came from.

The low k bias is expected. f ~ C|y-y*|^mu times an analytic factor contributes
k^-(mu+2) and faster alongside the leading k^-(mu+1), and those steeper terms
pull a fitted slope up at the low end. Predicted in advance, and the sign is
right.

The high k bias needs explaining, and the k^-2 candidate is dead: f' is
continuous at both stagnation points to 4e-12 and 1.6e-11. The remaining
suspect is the discretisation. The profile solves the dealiased equation, so
its highest retained modes carry error from the mask, and dropping the top
octave did raise beta.

That has a sharp signature. If the droop belongs to the discretisation, it sits
at a fixed fraction of k_cut and moves when N changes. If it belongs to the
profile, it sits at a fixed absolute k and does not move. Measure the same
absolute bands at three resolutions.

    python shortfall2.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, simulate, sup

A = 0.8
PRED = 3.001668
BANDS = [(32, 128), (64, 256), (128, 512), (256, 1024), (512, 2048),
         (1024, 4096)]


def fit_band(k, amp, k_lo, k_hi, ratio=2.0 ** 0.5):
    floor = amp.max() * 1e-13
    kc, vc, lo = [], [], float(k_lo)
    while lo * ratio <= k_hi:
        hi = lo * ratio
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
print("  Simulated profiles are used here rather than Newton solved ones, so")
print("  that N = 8192 and 16384 are affordable. They agree with the solved")
print("  profile to about 1e-5, far below the effect being measured.")
print()

results = {}
for n in (4096, 8192, 16384):
    g, out = simulate(A, n=n, w_max=1e6)
    f0, _ = guess_from_simulation(g, out)
    amp = np.abs(g.fwd(f0))
    results[n] = [fit_band(g.k, amp, lo, hi) for lo, hi in BANDS]
    print(f"  N = {n:5d}: k_cut = {g.kcut:5d}, "
          f"||f||_inf = {sup(g, f0):.6f}, tail = {out['tail_hist'][-1]:.2e}")

print()
print(f"  {'band':>16}  " + "  ".join(f"{'N=' + str(n):>11}" for n in results))
print("  " + "-" * 62)
for i, (lo, hi) in enumerate(BANDS):
    cells = []
    for n in results:
        v = results[n][i]
        cells.append(f"{v:11.5f}" if np.isfinite(v) else f"{'-':>11}")
    print(f"  {lo:6d} to {hi:6d}  " + "  ".join(cells))

print()
print(f"  prediction from the stagnation point: {PRED:.6f}")
print()
print("  Read down each column. If the droop is discretisation it appears at")
print("  the same fraction of k_cut in every column, so it slides right as N")
print("  grows and the bands that were drooping become clean. If it is real")
print("  structure in the profile, each row agrees across columns instead.")

print()
print("  Same thing as a fraction of k_cut, which is the discretisation test:")
print()
print(f"  {'band centre / k_cut':>22}  " +
      "  ".join(f"{'N=' + str(n):>11}" for n in results))
print("  " + "-" * 62)
for n in results:
    g = dg.Grid(n)
    fracs = [np.sqrt(lo * hi) / g.kcut for lo, hi in BANDS]
    line = "  ".join(f"{f:11.3f}" if np.isfinite(results[n][i]) else f"{'-':>11}"
                     for i, f in enumerate(fracs))
    print(f"  {'N = ' + str(n):>22}  " + line)
