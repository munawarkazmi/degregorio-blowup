"""
Stability of the blowup profile, from the spectrum of the Jacobian.

Renormalise. With s = -ln(T - t) and W(y, s) = (T - t) w, we have w = e^s W and
ds/dt = e^s, so w_t = e^(2s) (W + W_s), while rhs is quadratic and gives
rhs(w) = e^(2s) rhs(W). The e^(2s) cancels and the equation becomes

    W_s = rhs(W) - W = residual(W)

so the profile is precisely a fixed point of this flow, and linearising about it
gives d(delta)/ds = J delta with J = D[residual](f). The eigenvalues of J are
therefore growth rates in renormalised time, directly: Re(lambda) > 0 is a
direction the blowup does not survive, Re(lambda) < 0 is one it damps out.

Two eigenvalues are forced and are not instabilities.

  lambda = 1, eigenvector f. Because rhs is quadratic, D[rhs](f) f = 2 rhs(f)
  = 2f, so J f = 2f - f = f exactly. This is the freedom to move the blowup
  time: replacing T by T + eps gives W = f tau / (tau + eps), which to first
  order is f - eps e^s f, growing like e^s. Nothing to do with stability, and
  it is an exact check on the assembled Jacobian.

  lambda = 0, eigenvector f'. Translation invariance, since shifting a solution
  gives another solution.

So the profile is stable, and can be what the dynamics selects, exactly when
every other eigenvalue has negative real part. If an eigenvalue crosses zero
somewhere in a, that crossing should coincide with where profile_branch.py
found the dynamics stops selecting the profile, around a = 0.7.

A third family is an artefact worth recognising: for k above the dealiasing
cutoff the residual is just -f_k, so J is -I there and contributes eigenvalue
-1 with high multiplicity. Harmless, and stable.

    python spectrum.py
"""

import numpy as np

import dg
from profile_eq import guess_from_simulation, jacobian, newton, simulate, sup

A_REF = 0.8
N_REF = 1024


def spectrum(g, f, a, k=10):
    """
    Eigenvalues of the unregularised Jacobian, sorted by real part, with each
    eigenvector's overlap against the two symmetry modes so they can be
    identified rather than assumed.
    """
    J = jacobian(g, f, a)
    vals, vecs = np.linalg.eig(J)
    order = np.argsort(-vals.real)
    vals, vecs = vals[order], vecs[:, order]

    fp = g.bwd(g.fwd(f) * g.dmul)
    fn, fpn = f / np.linalg.norm(f), fp / np.linalg.norm(fp)

    rows = []
    for i in range(min(k, len(vals))):
        v = vecs[:, i]
        v = v / np.linalg.norm(v)
        rows.append((vals[i],
                     abs(complex(np.vdot(fn, v))),
                     abs(complex(np.vdot(fpn, v)))))
    return vals, rows


def label(ov_f, ov_fp):
    if ov_f > 0.9:
        return "shift of T"
    if ov_fp > 0.9:
        return "translation"
    return ""


# ---------------------------------------------------------------------------
print()
print("=" * 76)
print(f"Part 1: the two forced eigenvalues, as an exactness check (a = {A_REF})")
print("=" * 76)

g, out = simulate(A_REF, n=N_REF)
f0, amp_sim = guess_from_simulation(g, out)
f, hist, ok = newton(g, f0, A_REF)
print(f"  profile solved: converged {ok}, residual {hist[-1]:.2e}, "
      f"||f||_inf = {sup(g, f):.9f}")

J = jacobian(g, f, A_REF)
fp = g.bwd(g.fwd(f) * g.dmul)
print()
print("  J f  should equal f exactly, since D[rhs](f) f = 2 rhs(f) = 2f:")
print(f"      max |J f - f|  = {np.abs(J @ f - f).max():.3e}   "
      f"(relative {np.abs(J @ f - f).max() / np.abs(f).max():.3e})")
print("  J f' should vanish, by translation invariance:")
print(f"      max |J f'|     = {np.abs(J @ fp).max():.3e}   "
      f"(relative {np.abs(J @ fp).max() / np.abs(fp).max():.3e})")


# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("Part 2: the leading spectrum, and whether it converges in N")
print("=" * 76)

for n in (512, 1024, 2048):
    g_n = dg.Grid(n)
    _, out_n = simulate(A_REF, n=n)
    f0_n, _ = guess_from_simulation(g_n, out_n)
    f_n, h_n, ok_n = newton(g_n, f0_n, A_REF)
    if not ok_n:
        print(f"  N = {n}: Newton failed")
        continue
    vals, rows = spectrum(g_n, f_n, A_REF, k=8)
    print()
    print(f"  N = {n}   (||f||_inf = {sup(g_n, f_n):.6f})")
    print(f"    {'Re':>12}  {'Im':>12}  {'overlap f':>10}  "
          f"{'overlap f_x':>11}  meaning")
    for lam, of, ofp in rows:
        print(f"    {lam.real:12.7f}  {lam.imag:12.7f}  {of:10.4f}  "
              f"{ofp:11.4f}  {label(of, ofp)}")
    nontrivial = [v for (v, of, ofp) in rows
                  if of < 0.9 and ofp < 0.9 and v.real > 1e-6]
    print(f"    unstable directions beyond the two forced modes: "
          f"{len(nontrivial)}")


# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("Part 3: stability along the branch. Where does an eigenvalue cross?")
print("=" * 76)
print("  Each profile is solved from its own simulation, so the branch tested")
print("  is the one that run converged to. Continuing in a instead does not")
print("  work here: stepping from a = 0.8 to 0.85 lands on a branch with")
print("  ||f|| = 6.249 carrying an unstable eigenvalue at +2.077, whereas the")
print("  a = 0.85 simulation matches a branch with ||f|| = 5.668 to 8e-6. The")
print("  continuation jumps branches, and the branch it jumps to is not the")
print("  one the dynamics picks.")
print()
print("  The forced modes are identified by their exact values, 1 and 0, which")
print("  Part 1 confirms to 1e-12, rather than by eigenvector overlap. The")
print("  overlap heuristic used above mislabels: at N = 1024 it tagged a mode")
print("  at -0.146 as translation on a 0.9988 overlap when the true")
print("  translation mode is the one sitting at exactly zero.")
print()
print(f"  {'a':>6}  {'||f||_inf':>11}  {'profile residual':>17}  "
      f"{'leading nontrivial':>19}  {'verdict':>9}")
print("  " + "-" * 74)


def leading_nontrivial(vals, tol=1e-6):
    """
    Largest real part excluding the two forced eigenvalues, which sit at
    exactly 1 (shift of the blowup time) and exactly 0 (translation).
    """
    keep = [v for v in vals if abs(v - 1.0) > tol and abs(v) > tol]
    return max(keep, key=lambda v: v.real) if keep else None


for a in (0.65, 0.70, 0.75, 0.80, 0.85):
    g_a = dg.Grid(N_REF)
    _, out_a = simulate(a, n=N_REF)
    f0_a, _ = guess_from_simulation(g_a, out_a)
    if f0_a is None:
        print(f"  {a:6.2f}  no usable growth")
        continue
    raw = float(np.abs(dg.rhs(g_a, f0_a, a) - f0_a).max())
    f_a, h_a, ok_a = newton(g_a, f0_a, a)
    if not ok_a or sup(g_a, f_a) < 1e-6:
        why = "trivial f=0" if ok_a else "Newton failed"
        print(f"  {a:6.2f}  {why:>11}  {raw:17.3e}")
        continue
    vals, _ = spectrum(g_a, f_a, a, k=1)
    lam = leading_nontrivial(vals)
    verdict = "unstable" if lam.real > 1e-6 else "stable"
    print(f"  {a:6.2f}  {sup(g_a, f_a):11.6f}  {raw:17.3e}  "
          f"{lam.real:19.7f}  {verdict:>9}")

print()
print("  A stable verdict alongside a small profile residual is the whole")
print("  claim: the equation has a solution there, it is linearly stable, and")
print("  the simulation independently walks onto it.")
