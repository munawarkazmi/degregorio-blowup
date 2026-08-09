"""
Is the profile solution unique, or does Newton land wherever its guess points?

At a = 0.8 the solved peak height came out as 4.161220 at N = 1024, 4.148482
at N = 2048 and 4.154737 at N = 4096, non-monotonically. The N = 1024 offset is
explainable: the strip width there is about 0.019, so exp(-delta k_cut) is
1.5e-3 and truncation alone accounts for it. The other two are not. At N = 2048
truncation is 2e-6 and at N = 4096 it is 5e-12, yet they disagree by 1.5e-3,
three orders too much, with both Newton residuals below 1e-11.

Two explanations, and they have opposite consequences.

  Guess dependence. Each solve started from its own simulation, and those
  differ slightly. If the equation has several nearby solutions, Newton finds
  whichever is closest to where it started, and the spread across N is not a
  discretisation error at all.

  Genuine discretisation effect, in which case the profile is only pinned to
  about three digits and something about the problem is being missed.

Distinguish them by removing the variable: solve at both resolutions from the
same profile, spectrally interpolated between grids. Zero padding an rfft is
exact interpolation for a band limited field, so nothing is lost going up, and
going down only truncates modes that are already at 1e-12.

    python profile_unique.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, residual, simulate, sup

A = 0.8


def regrid(g_from, f, g_to):
    """Spectral interpolation between grids by zero padding or truncation."""
    fh = np.fft.rfft(f)
    out = np.zeros(g_to.N // 2 + 1, dtype=complex)
    n = min(len(fh), len(out))
    out[:n] = fh[:n]
    out *= g_to.N / g_from.N          # rfft scales with the point count
    return np.fft.irfft(out, g_to.N)


g2, g4 = dg.Grid(2048), dg.Grid(4096)

print()
print("Reference solve at N = 2048 from its own simulation:")
_, out2 = simulate(A, n=2048)
f0, amp_sim2 = guess_from_simulation(g2, out2)
f2, h2, ok2 = newton(g2, f0, A)
print(f"  converged {ok2} in {len(h2)} iters, residual {h2[-1]:.2e}")
print(f"  ||f||_inf = {sup(g2, f2):.9f}")

print()
print("Same solution carried to N = 4096 and re-solved there:")
f4_guess = regrid(g2, f2, g4)
print(f"  interpolation residual before Newton: "
      f"{np.abs(residual(g4, f4_guess, A)).max():.3e}")
f4, h4, ok4 = newton(g4, f4_guess, A)
print(f"  converged {ok4} in {len(h4)} iters, residual {h4[-1]:.2e}")
print(f"  ||f||_inf = {sup(g4, f4):.9f}")

print()
print("And carried back down to N = 2048:")
f2b_guess = regrid(g4, f4, g2)
f2b, h2b, ok2b = newton(g2, f2b_guess, A)
print(f"  converged {ok2b} in {len(h2b)} iters, residual {h2b[-1]:.2e}")
print(f"  ||f||_inf = {sup(g2, f2b):.9f}")

print()
print("-" * 62)
if ok2 and ok4 and ok2b:
    up = abs(sup(g4, f4) - sup(g2, f2)) / sup(g2, f2)
    back = abs(sup(g2, f2b) - sup(g2, f2)) / sup(g2, f2)
    print(f"  2048 -> 4096 changes ||f||_inf by {up:.3e}")
    print(f"  round trip back to 2048 changes it by {back:.3e}")
    print()
    print("  Small numbers here mean the solution is grid independent and the")
    print("  earlier spread came from each solve starting somewhere different,")
    print("  so the equation has more than one nearby solution. Large numbers")
    print("  mean the discretisation itself is moving the answer.")

    # If the solution is grid independent, the two simulations' own predictions
    # should also agree once compared against the same profile.
    _, out4 = simulate(A, n=4096)
    _, amp_sim4 = guess_from_simulation(g4, out4)
    print()
    print(f"  peak height from the N = 2048 simulation slope: {amp_sim2:.9f}")
    print(f"  peak height from the N = 4096 simulation slope: {amp_sim4:.9f}")
    print(f"  peak height from the equation, grid independent: "
          f"{sup(g4, f4):.9f}")
