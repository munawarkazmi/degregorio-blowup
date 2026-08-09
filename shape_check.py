"""
What does the a = 0.8 profile actually look like?

The figure drew a broad smooth hump, but fig_frozen shows its spectrum decaying
only to 1e-10 by k = 900, which needs a steep feature somewhere. Those two
readings cannot both be right, so measure the shape instead of eyeballing it.
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, newton, simulate, sup

g, out = simulate(0.8, n=1024)
f0, _ = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, 0.8)
print(f"converged {ok}, residual {hist[-1]:.2e}")

fx = g.bwd(g.fwd(f) * g.dmul)
y = g.x

print()
print(f"  max f          = {f.max():10.5f} at y = {y[int(np.argmax(f))]:.5f}")
print(f"  min f          = {f.min():10.5f} at y = {y[int(np.argmin(f))]:.5f}")
print(f"  ||f||_inf      = {sup(g, f):10.5f}")
print(f"  max |f'|       = {np.abs(fx).max():10.5f} at "
      f"y = {y[int(np.argmax(np.abs(fx)))]:.5f}")
print(f"  |f'| / ||f||   = {np.abs(fx).max() / sup(g, f):10.5f}   "
      f"(1 / this is the width of the steepest feature)")

print()
print("  f sampled near the steepest point:")
j = int(np.argmax(np.abs(fx)))
for d in range(-6, 7):
    i = (j + d * 3) % g.N
    print(f"    y = {y[i]:8.5f}   f = {f[i]:10.5f}")
