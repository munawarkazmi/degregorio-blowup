"""
Is the decaying case a separatrix or a third outcome with its own open set?

In omega_0 = sin x + c sin 2x, admixtures up to c = 0.45 reach the member 1
profile, c = 0.6 and above reach member 2, and c = 0.5 does neither: its
amplitude falls monotonically to 7.4e-3 by t = 400, with |w| t settling near
3.0, so it decays like 1/t.

Those two readings differ in kind. If decay happens only at an isolated c, it
is the separatrix between the two basins and a set of measure zero. If it
happens over an interval, there is a third outcome with an open basin of its
own, and the model admits global solutions from sign changing data, which no
result here predicts.

Resolve it by bisecting between 0.45 and 0.6.

    python basin_decay.py
"""

import numpy as np

import dg

A = 0.8


def outcome(c, n=1024, t_max=300.0, w_max=1e4):
    g = dg.Grid(n)
    w0 = np.sin(g.x) + c * np.sin(2.0 * g.x)
    w0 = w0 / np.abs(w0).max()
    out = dg.run(w0, a=A, grid=g, t_max=t_max, cfl=0.02, tail_tol=1e-9,
                 w_max=w_max)
    amp = float(out["winf_hist"][-1])
    if out["reason"] == "amplitude cap":
        return "blowup", amp, out["t"], out
    return "decay" if amp < 0.5 else "unclear", amp, out["t"], out


print()
print(f"  {'c':>7}  {'outcome':>9}  {'final |w|':>12}  {'t':>8}  "
      f"{'|w| t':>9}")
print("  " + "-" * 52)

for c in (0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60):
    kind, amp, t, out = outcome(c)
    prod = f"{amp * t:9.3f}" if kind == "decay" else f"{'-':>9}"
    print(f"  {c:7.2f}  {kind:>9}  {amp:12.4e}  {t:8.2f}  {prod}")

print()
print("  A single c decaying between two blowing up is a separatrix. A run of")
print("  them is a third outcome with an open basin.")

print()
print("=" * 66)
print("Which member does each side reach, and how does T behave?")
print("=" * 66)
print(f"  {'c':>7}  {'outcome':>9}  {'T':>9}")
print("  " + "-" * 30)
for c in (0.40, 0.44, 0.46, 0.56, 0.60, 0.70):
    kind, amp, t, out = outcome(c)
    print(f"  {c:7.2f}  {kind:>9}  {t:9.3f}")
print()
print("  T diverging on approach from both sides would confirm a boundary")
print("  between the two basins, whatever sits on it.")
