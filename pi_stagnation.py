"""
The stagnation point relation, tested against the exact a = 1/2 solution.

Finding 8 derives, at a stagnation point where the profile has a simple zero,

    a c f' (y - y1) = f (c - 1)      hence      c = 1 / (1 - a)

and finding 3 now gives an exact solution at a = 1/2, so the obvious test is
whether c = 1/(1 - 1/2) = 2 there. It does not, and the reason is worth more
than the test was.

Finding 8 assumes a *frozen* profile, w = f(y)/(T - t), where the only
transport is a U. At a = 1/2 the blowup is not frozen. Finding 2 already
placed a = 0.5 in the narrowing regime with profile residual 0.49, and the
exact solution now says exactly how much it narrows.

Take v -> infinity in the closed form of finding 3 near the blowup point
x = pi, with z = x - pi and zeta = v z / 2. Since X = tan(x/2) = -2/z + O(z),
X = -v/zeta, and the two terms of w combine to

    w = (v^3 / 2) * (-zeta) / (1 + zeta^2)^2 .

The pole ODE gives dv/dt -> v^4/32 as v -> infinity, so v^3 = 32/(3(T - t)),
which is the narrowing rate:

    lambda = 1/3     exactly,     w = (T - t)^-1 F(zeta),   zeta = z/(T-t)^(1/3) * const

    F(zeta)  = -(16/3) zeta / (1 + zeta^2)^2
    HF(zeta) = -(8/3) (zeta^2 - 1) / (zeta^2 + 1)^2
    U(zeta)  =  (8/3) zeta / (zeta^2 + 1),        U' = HF, U(0) = 0

For a narrowing profile the self similar equation carries an extra transport
term from the rescaling, and reads

    F + lambda zeta F' + a U F' = F HF

so the transport velocity in the moving frame is not a U but

    V(zeta) = lambda zeta + a U(zeta) = (zeta/3)(zeta^2 + 5)/(zeta^2 + 1)

whose only zero is zeta = 0, the blowup point itself. Redoing finding 8's
leading order match there, with F ~ A zeta and V ~ (lambda + a c) zeta:

    A zeta + (lambda + a c) zeta A = A zeta c      hence      1 + lambda + a c = c

    c = (1 + lambda) / (1 - a)

At a = 1/2, lambda = 1/3, so c = (4/3)/(1/2) = 8/3 = 2.6666..., not 2.

Same for the general exponent: F ~ C zeta^mu gives 1 + (lambda + a c) mu = c,
so mu = (c - 1)/(lambda + a c), which is finding 8's mu = (c - 1)/(a c) at
lambda = 0.

**None of this is new to the paper.** Remark rem:general already states the
full self similar ansatz w = (T-t)^(c_w) Om(x/(T-t)^(c_l)), its profile
equation (c_l X + a U) Om_X = (c_w + U_X) Om, the exponent

    nu = (c_w + h) / (c_l + a h),      h = H Om(X*)

and that a simple zero forces h = (1 + c_l)/(1 - a), with Proposition
prop:c1 the c_l = 0 case. The dictionary is c_l = lambda = 1/3,
c_w = -1, h = c = 8/3.

So this is not a correction. It is the first *exact* check of that remark. Up
to now the whole local analysis was tested only against frozen profiles, where
c_l = 0 and the two formulas coincide, so the c_l dependence was the one part
of the framework carrying no evidence at all. At a = 1/2 there is no frozen
profile, c_l = 1/3, and the remark is verified against a closed form solution
rather than against a simulation.

The test also confirms nu = 1 at the simple zero, since
(c_w + h)/(c_l + a h) = (8/3 - 1)/(1/3 + 4/3) = (5/3)/(5/3) = 1.

    python pi_stagnation.py
"""

import numpy as np

import dg

A_PARAM = 0.5
LAMBDA = 1.0 / 3.0


def G(v):
    return v * (v * v - 1.0) / (v * v + 1.0) ** 2 + np.arctan(v)


def time_left(v):
    """T - t as a function of v_c, exact: T - t = 4[G(inf) - G(v)]."""
    return 4.0 * (0.5 * np.pi - G(v))


def w_of_v(x, v):
    """The exact a = 1/2 solution of finding 3, parametrised by v_c."""
    w1 = (v * v + 1.0) ** 2 / 4.0
    w2i = (v * v - 1.0) * (v * v + 1.0) ** 2 / (8.0 * v)
    X = np.tan(0.5 * x)
    d = X * X + v * v
    return 2.0 * w1 * X / d - 4.0 * w2i * v * X / (d * d)


def F(z):
    return -(16.0 / 3.0) * z / (1.0 + z * z) ** 2


def dF(z):
    return -(16.0 / 3.0) * (1.0 - 3.0 * z * z) / (1.0 + z * z) ** 3


def HF(z):
    return -(8.0 / 3.0) * (z * z - 1.0) / (z * z + 1.0) ** 2


def U(z):
    return (8.0 / 3.0) * z / (z * z + 1.0)


print(__doc__.split("    python")[0])
print("=" * 72)

# --------------------------------------------- 1. the narrowing rate is 1/3
print("\n  1. v^3 (T - t) -> 32/3, so lambda = 1/3\n")
print(f"     {'v':>10}  {'v^3 (T-t)':>16}  {'error vs 32/3':>14}")
for v in (10.0, 30.0, 100.0, 300.0, 1000.0, 1e4):
    q = v ** 3 * time_left(v)
    print(f"     {v:10.0f}  {q:16.10f}  {abs(q - 32.0 / 3.0):14.2e}")

# ------------------------------------- 2. the rescaled solution tends to F
print("\n  2. (T - t) w(x, t) against F(zeta), on |zeta| <= 8\n")
print(f"     {'v':>10}  {'N':>8}  {'max abs diff':>14}  {'relative':>10}")
for v, N in ((10.0, 1 << 14), (30.0, 1 << 15), (100.0, 1 << 16),
             (300.0, 1 << 18), (1000.0, 1 << 20)):
    g = dg.Grid(N)
    z = g.x - np.pi
    zeta = 0.5 * v * z
    m = np.abs(zeta) <= 8.0
    lhs = time_left(v) * w_of_v(g.x, v)[m]
    rhs = F(zeta[m])
    e = np.abs(lhs - rhs).max()
    print(f"     {v:10.0f}  {N:8d}  {e:14.3e}  {e / np.abs(rhs).max():10.2e}")

# -------------------------------- 3. c = H F(0), read off the exact solution
print("\n  3. c = (T - t) Hw(pi, t), by dg's own Hilbert transform\n")
print(f"     {'v':>10}  {'N':>8}  {'c measured':>16}  {'vs 8/3':>10}  {'vs 2':>8}")
for v, N in ((10.0, 1 << 14), (30.0, 1 << 15), (100.0, 1 << 16),
             (300.0, 1 << 18), (1000.0, 1 << 20)):
    g = dg.Grid(N)
    hw = g.hilbert(w_of_v(g.x, v))
    c = time_left(v) * hw[N // 2]          # x[N//2] is exactly pi
    print(f"     {v:10.0f}  {N:8d}  {c:16.10f}  {abs(c - 8.0/3.0):10.2e}  "
          f"{abs(c - 2.0):8.2e}")

# ------------------------------------ 4. the self similar equation holds
zz = np.linspace(-40.0, 40.0, 200001)
res = F(zz) + LAMBDA * zz * dF(zz) + A_PARAM * U(zz) * dF(zz) - F(zz) * HF(zz)
print(f"\n  4. residual of F + lambda z F' + a U F' - F HF   "
      f"max {np.abs(res).max():.2e}")

# The closed form for HF is checked against a numerical transform on the line,
# since everything above leans on it.
L, M = 4000.0, 1 << 22
zl = -L + 2.0 * L * np.arange(M) / M
k = np.fft.fftfreq(M, d=2.0 * L / M) * 2.0 * np.pi
num = np.fft.ifft(np.fft.fft(F(zl)) * (-1j * np.sign(k))).real
core = np.abs(zl) <= 8.0
print(f"     HF closed form against FFT on the line, |z| <= 8  "
      f"max {np.abs(num[core] - HF(zl[core])).max():.2e}")

# ------------------------------------------------- 5. what the relation says
c_exact = 8.0 / 3.0
print("\n  5. The relation, and its scope\n")
print(f"     stagnation points of V = lambda z + a U      only z = 0")
print(f"     F has a simple zero there, F'(0)             {dF(0.0):.6f}")
print(f"     c = HF(0)                                    {HF(0.0):.10f}")
print(f"     (1 + lambda)/(1 - a) = (4/3)/(1/2)           {(1+LAMBDA)/(1-A_PARAM):.10f}")
print(f"     finding 8's 1/(1 - a)                        {1.0/(1-A_PARAM):.10f}")
print(f"     local balance 1 + lambda + a c - c           "
      f"{1.0 + LAMBDA + A_PARAM * c_exact - c_exact:.2e}")
print(f"     V'(0) = lambda + a c                         "
      f"{LAMBDA + A_PARAM * c_exact:.10f}")

print(f"     nu = (c_w + h)/(c_l + a h), c_w = -1         "
      f"{(-1.0 + c_exact) / (LAMBDA + A_PARAM * c_exact):.10f}   (simple zero)")

print("""
  c = 2 is refuted at a = 1/2, but it was never the prediction there.
  Proposition prop:c1 requires c_l = 0, and a = 1/2 has c_l = 1/3. The right
  prediction is Remark rem:general's h = (1 + c_l)/(1 - a) = 8/3, and that is
  what the exact solution returns.

  The remark was previously supported only by frozen profiles, where c_l = 0
  and it is indistinguishable from prop:c1. This is the first test that can
  tell them apart, and it is exact rather than numerical.
""")
