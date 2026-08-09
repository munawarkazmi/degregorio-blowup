"""
The sweep returned T(0.5) = 3.1416 for w_0 = sin x, which is pi to the five
digits the run could resolve. At those settings the a = 0 control came out as
1.99853 against an exact 2, so the accuracy was only about 1.5e-3 relative and
the agreement means very little on its own.

Worth one focused test. Refine N and watch T(0.5) converge, carrying the a = 0
run alongside at identical settings as the error bar: whatever |T(0) - 2| is
at a given N is roughly what T(0.5) is worth at that N.

    python pi_check.py
"""

import numpy as np

import dg

print()
print(f"{'N':>7}  {'T(a=0)':>12}  {'err vs 2':>10}  {'T(a=0.5)':>12}  "
      f"{'err vs pi':>10}")
print("-" * 60)

for N in (2048, 4096, 8192, 16384):
    row = []
    for a in (0.0, 0.5):
        g = dg.Grid(N)
        out = dg.run(np.sin(g.x), a=a, grid=g, t_max=10.0, cfl=0.01,
                     tail_tol=1e-9, w_max=1e8)
        T, _, _ = dg.fit_blowup(out["t_hist"], out["winf_hist"],
                                window_decades=0.5)
        row.append(T)
    print(f"{N:7d}  {row[0]:12.7f}  {abs(row[0] - 2.0):10.2e}  "
          f"{row[1]:12.7f}  {abs(row[1] - np.pi):10.2e}")

print()
print("If the a = 0.5 error against pi shrinks in step with the a = 0 error")
print("against 2, T(1/2) = pi is worth chasing analytically. If it stalls at")
print("some fixed value while the control keeps improving, it is a coincidence")
print("at the third decimal and nothing more.")
