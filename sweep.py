"""
The headline experiment: how does the fate of sin(x) depend on a?

This datum is chosen because both ends of the sweep are known exactly.

    a = 0  (Constantin-Lax-Majda)  blows up at T = 2, with rate exponent 1.
    a = 1  (De Gregorio)           sin(x) is a steady state, so T = infinity.

So we already know the answer at both endpoints and can watch T(a) climb
between them. Anything that fails to reproduce T(0) = 2 is broken.

Caveat worth keeping in front of you: sin(x) is not a generic datum for the
a = 1 model, it is exactly that model's ground state. What this sweep measures
is how the ground state destabilises as advection is weakened, which is a
sharper and narrower question than "when does this family blow up". Other data
can and will behave differently, which is what --ic is for.

    python sweep.py                     default scan, a = 0 to 1
    python sweep.py --na 21 --tmax 40   finer, run longer
    python sweep.py --ic tilted         a datum that is not the ground state
"""

import argparse
import csv

import numpy as np

import dg


def initial_condition(name, x):
    if name == "sine":
        return np.sin(x)
    if name == "tilted":
        # Deliberately not the a = 1 steady state, still zero mean and
        # sign changing.
        return np.sin(x) + 0.3 * np.sin(2.0 * x)
    if name == "onesigned":
        # Lei, Liu and Ren proved a = 1 stays global for one signed data on
        # the circle, so this row should never blow up at a = 1.
        return 1.0 - np.cos(x)
    raise ValueError(f"unknown initial condition: {name}")


def classify(out, growth, T, r2):
    if out["reason"] == "non-finite":
        return "diverged"
    if growth > 20.0 and np.isfinite(T) and r2 > 0.999:
        return "blowup"
    if growth < 3.0 and out["reason"] == "t_max":
        return "bounded"
    return "unresolved"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--amin", type=float, default=0.0)
    p.add_argument("--amax", type=float, default=1.0)
    p.add_argument("--na", type=int, default=11)
    p.add_argument("--tmax", type=float, default=20.0)
    p.add_argument("--cfl", type=float, default=0.02)
    p.add_argument("--tail-tol", type=float, default=1e-9)
    p.add_argument("--ic", default="sine",
                   choices=["sine", "tilted", "onesigned"])
    p.add_argument("--out", default="sweep.csv")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    g = dg.Grid(args.N)
    w0 = initial_condition(args.ic, g.x)
    avals = np.linspace(args.amin, args.amax, args.na)

    print(f"N = {args.N}  (kcut = {g.kcut})   initial condition = {args.ic}")
    print(f"cfl = {args.cfl}   tail_tol = {args.tail_tol:.0e}   "
          f"t_max = {args.tmax}")
    print("The resolution guard at this tail_tol holds the peak to roughly")
    print("1e-6 relative, per Test 6 of validate.py.")
    print()
    header = (f"{'a':>6}  {'verdict':>10}  {'T':>10}  {'p':>8}  {'R^2':>9}  "
              f"{'growth':>10}  {'t_end':>8}  {'stop':>15}")
    print(header)
    print("-" * len(header))

    rows = []
    records = []
    for a in avals:
        out = dg.run(w0, a=float(a), grid=g, t_max=args.tmax, cfl=args.cfl,
                     tail_tol=args.tail_tol)
        wh = out["winf_hist"]
        growth = float(wh[-1] / wh[0])
        T, pexp, r2 = dg.fit_blowup(out["t_hist"], wh, window_decades=0.5)
        verdict = classify(out, growth, T, r2)

        tstr = f"{T:10.5f}" if np.isfinite(T) else f"{'inf':>10}"
        pstr = f"{pexp:8.4f}" if np.isfinite(pexp) else f"{'-':>8}"
        rstr = f"{r2:9.6f}" if np.isfinite(r2) else f"{'-':>9}"
        print(f"{a:6.3f}  {verdict:>10}  {tstr}  {pstr}  {rstr}  "
              f"{growth:10.2f}  {out['t']:8.4f}  {out['reason']:>15}")

        rows.append({
            "a": float(a), "verdict": verdict, "T": T, "p": pexp, "r2": r2,
            "growth": growth, "t_end": out["t"], "stop": out["reason"],
            "w_end": float(wh[-1]), "steps": out["steps"],
        })
        records.append((float(a), out["t_hist"], wh))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {args.out}")

    exact = [r for r in rows if r["a"] == 0.0]
    if exact:
        T0 = exact[0]["T"]
        print(f"consistency check at a = 0: T = {T0:.6f}, exact value is 2, "
              f"error {abs(T0 - 2.0):.2e}")

    blow = [r for r in rows if r["verdict"] == "blowup"]
    bounded = [r for r in rows if r["verdict"] == "bounded"]
    if blow and bounded:
        print(f"transition is between a = {max(r['a'] for r in blow):.3f} "
              f"(blowup) and a = {min(r['a'] for r in bounded):.3f} (bounded); "
              f"rerun with --amin/--amax/--na to bracket it more tightly")
    elif not blow:
        print("no blowup detected anywhere in this range")
    elif not bounded:
        print(f"blowup everywhere in this range; note that 'bounded' requires "
              f"surviving to t_max = {args.tmax} with growth under 3x")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        cmap = plt.get_cmap("viridis")
        for a, th, wh in records:
            ax1.plot(th, wh[0] / wh, color=cmap(a / max(avals.max(), 1e-12)),
                     label=f"a = {a:.2f}")
        ax1.set_xlabel("t")
        ax1.set_ylabel(r"$\|w_0\|_\infty / \|w(t)\|_\infty$")
        ax1.set_title("reciprocal amplitude: a straight line hitting zero\n"
                      "is blowup at rate 1")
        ax1.set_ylim(0, 1.05)
        ax1.legend(fontsize=7, ncol=2)

        ab = [r["a"] for r in rows if np.isfinite(r["T"])]
        Tb = [r["T"] for r in rows if np.isfinite(r["T"])]
        ax2.plot(ab, Tb, "o-")
        ax2.axhline(2.0, ls=":", lw=1, color="k")
        ax2.annotate("exact: T(0) = 2", (0.0, 2.0), textcoords="offset points",
                     xytext=(8, 8), fontsize=8)
        ax2.set_xlabel("a")
        ax2.set_ylabel("estimated blowup time T")
        ax2.set_title("T(a) for $w_0 = \\sin x$")
        ax2.set_yscale("log")
        fig.tight_layout()
        fig.savefig("sweep.png", dpi=140)
        print("wrote sweep.png")


if __name__ == "__main__":
    main()
