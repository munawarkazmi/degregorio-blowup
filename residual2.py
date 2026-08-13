"""
Was the +1.3e-3 residual an artefact of comparing two different objects?

shortfall2.py measured beta on the *simulated* profile and got 3.00303 at
N = 4096 over the band 128 to 512. residual.py measured it on the *Newton*
profile over the same relative band and got 2.98626. Those should not differ by
1.7e-2 when the two profiles agree to about 1e-5 in sup norm, so one of two
things is happening.

  The objects genuinely differ where it counts. Sup norms agreeing to 1e-5 says
  nothing about the high k tail, where the fit lives. The simulated profile is
  a snapshot still approaching the fixed point; the Newton profile is the exact
  fixed point of the discrete equation.

  Or the binning differs. The two callers construct bins slightly differently,
  and one ends with four bins over 128 to 512 while the other ends with three.
  Dropping a bin from a four point fit can move the slope a lot.

These have different fixes, so separate them: measure both profiles with both
binnings, all four combinations.

The self consistency point matters regardless. The prediction uses
c = H f(y*) read off a particular profile, so the measurement has to come from
that same profile or the comparison is meaningless.

    python residual2.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

A = 0.8


def fit(k, amp, k_lo, k_hi, ratio=2.0 ** 0.5):
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
        return np.nan, len(kc)
    x, y = np.log(np.array(kc)), np.log(np.array(vc))
    return -float(np.polyfit(x, y, 1)[0]), len(kc)


g, out = simulate(A, n=4096, w_max=1e6)
f_sim, _ = guess_from_simulation(g, out)
f_new, hist, ok = newton(g, f_sim.copy(), A)
print()
print(f"  simulated profile: ||f|| = {sup(g, f_sim):.7f}, "
      f"equation residual {np.abs(dg.rhs(g, f_sim, A) - f_sim).max():.2e}")
print(f"  Newton profile:    ||f|| = {sup(g, f_new):.7f}, "
      f"equation residual {hist[-1]:.2e}")
print(f"  difference in sup norm: "
      f"{abs(sup(g, f_sim) - sup(g, f_new)):.2e}")

a_sim, a_new = np.abs(g.fwd(f_sim)), np.abs(g.fwd(f_new))
print()
print("  relative difference of the two spectra, by band:")
for lo, hi in ((16, 64), (64, 256), (128, 512), (256, 1024), (512, 1365)):
    sel = (g.k >= lo) & (g.k < hi)
    rel = np.abs(a_sim[sel] - a_new[sel]) / a_new[sel]
    print(f"    {lo:5d} to {hi:5d}:  median {np.median(rel):.3e}   "
          f"max {rel.max():.3e}")

print()
print("=" * 74)
print("All four combinations")
print("=" * 74)
print(f"  {'profile':>10}  {'window':>18}  {'bins':>5}  {'beta':>11}")
print("  " + "-" * 50)
for name, amp in (("simulated", a_sim), ("Newton", a_new)):
    for label, lo, hi in (("128 to 512", 128, 512),
                          ("0.094 to 0.375 kcut", 0.094 * g.kcut,
                           0.375 * g.kcut)):
        b, nb = fit(g.k, amp, lo, hi)
        print(f"  {name:>10}  {label:>18}  {nb:5d}  {b:11.6f}")

print()
print("=" * 74)
print("Sensitivity of the Newton profile fit to the window")
print("=" * 74)
print("  If a single bin moves the answer by 1e-2, the measurement was never")
print("  good to 1e-3 and the original residual was below its own noise.")
print()
print(f"  {'window':>16}  {'bins':>5}  {'beta':>11}")
print("  " + "-" * 36)
for lo, hi in ((96, 512), (128, 512), (128, 724), (96, 724), (128, 384),
               (181, 724), (128, 1024), (64, 512)):
    b, nb = fit(g.k, a_new, lo, hi)
    print(f"  {lo:6d} to {hi:6d}  {nb:5d}  {b:11.6f}")

bs = [fit(g.k, a_new, lo, hi)[0] for lo, hi in
      ((96, 512), (128, 512), (128, 724), (96, 724), (128, 384),
       (181, 724), (128, 1024), (64, 512))]
bs = [b for b in bs if np.isfinite(b)]
print()
print(f"  spread over these windows: {max(bs) - min(bs):.3e}")
print("  prediction from this profile: 3.0016682")
