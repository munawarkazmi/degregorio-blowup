"""
Part 2 of mechanism.py reported ||w||_inf * delta growing in exact proportion
to ||w||_inf at a = 0.8, which says the strip width delta is constant at about
0.0125. That cannot be squared with the run finishing on the amplitude cap: a
spectrum decaying like exp(-0.0125 k) out to k_cut = 682 leaves 2e-4 of the
peak amplitude at the cutoff, hence about 2.5e-6 of the energy in the top
quarter of the band, which is three orders above the 1e-9 guard that never
fired. One of the two numbers is lying.

So stop inferring the shape from a fitted slope and look at the spectrum.

The question underneath is the one that matters. If the solution approaches
A(t) f(x - x_p(t)) with f fixed, the blowup is pure amplification of a frozen
profile and the normalised spectrum |w_k| / max|w_k| is independent of time.
If instead the peak narrows, that normalised spectrum spreads steadily toward
high k. Taking the modulus removes the translation, so this comparison does not
need the peak locations to be aligned.

    python profile.py
"""

import numpy as np

import dg

LEVELS = (1e2, 1e3, 1e4, 1e5, 1e6)


def normalised_spectrum(g, w):
    amp = np.abs(g.fwd(w))
    return amp / amp.max()


for a in (0.0, 0.4, 0.8):
    print()
    print("=" * 78)
    print(f"a = {a}")
    print("=" * 78)

    g = dg.Grid(4096)
    w = np.sin(g.x)
    t0 = 0.0
    specs, snaps = [], []

    for lvl in LEVELS:
        out = dg.run(w, a=a, grid=g, t_max=60.0, cfl=0.02, tail_tol=1e-11,
                     w_max=lvl)
        w, t0 = out["w"], t0 + out["t"]
        peak, xp = dg.sup_norm(g, w)
        if out["reason"] not in ("amplitude cap",):
            print(f"  stopped early at ||w|| = {peak:.3e}: {out['reason']}")
            break
        s = normalised_spectrum(g, w)
        specs.append(s)
        snaps.append((peak, xp, s))

        # Where does the normalised spectrum cross given thresholds? This is
        # the honest version of "how wide is the peak in Fourier space".
        def crossing(thr):
            # Search from the peak mode outward, not from k = 0. The mean
            # vorticity is conserved at zero, so s[0] is zero and a naive
            # search reports every threshold as crossed at k = 0.
            k0 = int(np.argmax(s))
            below = np.flatnonzero(s[k0:] < thr)
            return int(below[0]) + k0 if below.size else -1

        print(f"  ||w|| = {peak:10.3e}  x_peak = {xp:8.5f}  "
              f"t = {t0:8.5f}  tail = {out['tail_hist'][-1]:.2e}")
        print(f"      k where spectrum falls below 1e-3: {crossing(1e-3):5d}   "
              f"1e-6: {crossing(1e-6):5d}   1e-9: {crossing(1e-9):5d}   "
              f"(k_cut = {g.kcut})")
        print(f"      strip_width says delta = {dg.strip_width(g, w):.6f}, so "
              f"exp(-delta*k_cut) = {np.exp(-dg.strip_width(g, w) * g.kcut):.2e}")

    if len(specs) >= 2:
        print()
        print("  normalised spectra compared against the first snapshot:")
        ref = specs[0]
        for (peak, _, s) in snaps[1:]:
            n = min(len(ref), len(s))
            d = float(np.abs(s[:n] - ref[:n]).max())
            print(f"      ||w|| = {peak:9.2e}:  max |spectrum - reference| "
                  f"= {d:.3e}")
        print("  near zero means a frozen profile; growth means narrowing")
