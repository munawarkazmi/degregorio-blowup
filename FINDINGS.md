# Findings

All from `w_0 = sin x` on the circle. Everything below is floating point
numerics with convergence studies, not proof. Reproduce with `mechanism.py`,
`profile.py` and `pi_check.py`.

## 1. Blowup for every `a < 1`, with `T(a)` diverging as `a -> 1`

| `a` | `T` | rate exponent `p` | growth reached |
| --- | --- | --- | --- |
| 0 | 2 (exact) | 1 (exact) | - |
| 0.2 | 2.3101 | 0.9971 | 2.5e2 |
| 0.4 | 2.7858 | 1.0000 | 8.4e3 |
| 0.5 | 3.1416 | 1.0005 | 5.9e5 |
| 0.6 | 3.6500 | 1.0005 | 1e8 |
| 0.7 | 4.4641 | 0.9986 | 1e8 |
| 0.8 | 6.0719 | 1.0000 | 1e8 |
| 0.9 | 10.7569 | 1.0487 | 1e8 |
| 0.95 | 17.7129 | 0.4813 (unreliable) | 1e8 |
| 1 | infinity (exact) | - | steady |

The rate exponent is 1 across the range, the same as the exactly solvable
`a = 0` case. No transition to global existence appears at any `a < 1`; what
happens instead is that `T` runs away as the advection term is restored. That
is consistent with `a = 1` being critical, and with `sin x` being exactly the
`a = 1` steady state.

The `a = 0.95` exponent of 0.48 is a fit failure, not a physical result. Its
`R^2` is 0.999 rather than the 0.99999 seen elsewhere, and the growth phase is
compressed against the end of the run.

## 2. Two structurally different blowups, not one

This is the part worth following up. Compare normalised spectra
`|w_k| / max|w_k|` at successive decades of amplitude, which is translation
invariant and assumes nothing about the profile (`profile.py`).

**Large `a` freezes.** At `a = 0.8` the normalised spectrum stops changing:
deviation from the first snapshot goes 3.303e-3, 3.676e-3, 3.762e-3, 3.779e-3
across four decades of growth, converging rather than drifting. The peak
location locks at `x = 2.16699` to five decimals. The spectral tail sits at
2.7e-17 the whole way, so a 4096 point grid carries an amplitude of 1e6 without
strain.

**Small `a` narrows.** At `a = 0.4` the same comparison moves by 0.72 over a
single decade, the fitted strip width falls from 0.099 to 0.032, and the run is
under-resolved by `||w|| = 5.8e3`. At `a = 0` the strip width obeys the exact
`delta ~ (2 - t) / 2`, so `||w|| * delta -> 1/2`, and resolution is exhausted by
`||w|| = 41`.

So at large `a` the solution approaches

```
w(x, t) -> f(x - x_p) / (T - t)
```

with `f` a fixed profile, and substituting that ansatz means `f` satisfies

```
f = f H f - a F f',      F' = H f
```

which is an ordinary eigenvalue-type problem on the circle, not a PDE. That is
a concrete, checkable object.

Where the changeover happens is not resolved here. `a = 0.6` sits in between,
with the strip width falling like `w^-0.17` and still drifting, which may be
transient rather than a genuine intermediate scaling.

## 3. `T(1/2) = pi`, to eleven digits

`pi_check.py` refines `N` with the `a = 0` case carried alongside at identical
settings as the error bar:

| `N` | `T(a=0)` | error vs 2 | `T(a=1/2)` | error vs `pi` |
| --- | --- | --- | --- | --- |
| 2048 | 1.9984616 | 1.5e-3 | 3.1415927 | 3.6e-9 |
| 4096 | 1.9996348 | 3.7e-4 | 3.1415927 | 7.1e-10 |
| 8192 | 1.9999116 | 8.8e-5 | 3.1415927 | 9.6e-11 |
| 16384 | 1.9999794 | 2.1e-5 | 3.1415927 | 1.2e-11 |

The `a = 1/2` column is six orders of magnitude closer to its target than the
control is to its own exactly known answer, which needs explaining before it
can be believed.

**The explanation offered here first was wrong.** It said `a = 1/2` is a frozen
profile blowup, giving `||w|| = ||f|| / (T - t)` as an exact power law and so an
unbiased estimator, against an `a = 0` control whose `4 / (4 - t^2)` carries a
biasing `(2 + t)` factor. The second half of that still holds, but the first
half does not: finding 4 measures the residual of the profile equation directly
and `a = 0.5` comes out at 0.49, nowhere near a frozen profile. It is in the
narrowing regime.

**Settled: it is exact, and it is a theorem.** `a = 1/2` is one of the two
exactly solvable cases of this family on the circle, the other being `a = 0`.
Silantyev, Lushnikov, Siegel and Ambrose, arXiv:2411.01891, solve both by pole
dynamics. Our datum lies in their class. In their variable `X = tan(x/2)`,

```
sin x = 2X / (1 + X^2) = 1/(X - i) + 1/(X + i)
```

a single conjugate pole pair at `v_c = 1`, which is their `v_c -> 1` limit where
the complex singularity sits at infinite height, the statement that `sin x` is
entire. At `a = 1/2` advection makes logarithms out of simple poles, so their
invariant class is a double pole plus a simple pole tied by their (42),
`w_1 = 2 i v_c w_2 / (1 - v_c^2)`. At `v_c = 1` that reads `1 = (2i/0) * 0`, so
`sin x` is the one point where the parametrisation degenerates. The degeneracy
is removable: (42) forces `w_2i(0) = (v_c^2(0) - 1)/(2 v_c(0))`, and that
vanishing amplitude cancels the vanishing denominator in the coefficient of
their (59), leaving `K = 1/[2(v_c^2(0) + 1)^2]`, which is `1/8` at `v_c(0) = 1`.
So the flow through `sin x` is regular and

```
dv_c/dt = (v_c^2 + 1)^3 / (32 v_c^2),      v_c(0) = 1.
```

With `G(x) = x(x^2 - 1)/(x^2 + 1)^2 + arctan x`, which has
`G'(x) = 8x^2/(x^2 + 1)^3`, this integrates to `G(v_c(t)) = G(1) + t/4`. `K > 0`
so `v_c` increases, and blowup is `v_c -> infinity`, their type B, the complex
singularity reaching the real axis at `x = +-pi`. Since `G(1) = pi/4` and
`G(infinity) = pi/2`,

```
T = 4 [G(infinity) - G(1)] = 4 [pi/2 - pi/4] = pi     exactly.
```

The whole solution is closed form, with `v = v_c(t)` and `X = tan(x/2)`:

```
w_1(t)  = (v^2 + 1)^2 / 4
w_2i(t) = (v^2 - 1)(v^2 + 1)^2 / (8v)
w(x, t) = 2 w_1 X / (X^2 + v^2)  -  4 w_2i v X / (X^2 + v^2)^2
```

`pi_exact.py` checks this against the solver. Agreement is at machine precision
and tightens with `N`:

| `t` | `N` | max abs error | relative |
| --- | --- | --- | --- |
| 0.50 | 2048 | 2.8e-14 | 2.7e-14 |
| 0.50 | 8192 | 7.6e-15 | 7.3e-15 |
| 1.00 | 8192 | 8.0e-15 | 7.0e-15 |
| 2.00 | 8192 | 1.9e-14 | 1.1e-14 |
| 2.50 | 8192 | 3.4e-14 | 1.1e-14 |

The same script carries `a = 0` as a control, where switching the double pole
off and taking `nu -> 0` gives `v_c = (2 + t)/(2 - t)` and `w_1 = (v + 1)^2/4`,
which reduces by hand to `4 sin x / (4 + 4t cos x + t^2)`, the
Constantin-Lax-Majda solution, and returns `T = 2`. That control earned its
place: it caught a wrong amplitude in the first draft of the script, which had
`w_1 = (v^2 + 1)/2`, right at `t = 0` by coincidence and wrong after. Note also
that the biasing `(2 + t)` blamed above is exactly this pole trajectory.

So the eleven digit agreement was neither a coincidence nor a lucky fit, and the
two decimals the fit could not reach were real.

**This was a literature miss, not a discovery.** The overlap list had settled at
seven papers, all from the Hou, Chen, Huang, Zheng and Okamoto line. The pole
dynamics school, Lushnikov, Silantyev, Siegel, Ambrose and Schochet, was absent
from it entirely, and it is the one that owns exact solutions of this family.
Searching by topic and by author within one school does not surface the other.

## 4. The blowup profile solves a fixed point equation, confirmed to 1e-5

Substituting `w = f(x - x_p) / (T - t)` into the PDE gives `f / (T-t)^2` on the
left and `rhs(f) / (T-t)^2` on the right, so the profile equation is

```
rhs(f) = f,     rhs(f) = f H f - a U f',   U' = H f
```

a fixed point of the same nonlinearity `dg.py` already validates. It has no
free scaling: if `f` solves it then `cf` gives `c^2 rhs(f) = c^2 f`, which
equals `cf` only at `c = 1`. So the equation *fixes the amplitude*, and since
`||w|| = ||f|| / (T - t)` gives

```
d/dt (1 / ||w||) = -1 / ||f||_inf
```

the peak height is a falsifiable prediction. The two computations, time
marching a PDE and Newton on an algebraic fixed point, share nothing but `a`.

At `a = 0.8` the simulated profile satisfies `rhs(f) = f` to 1.3e-5 *before*
Newton runs. Newton then converges quadratically, 1.3e-5 to 4.1e-11 to
1.8e-12, and:

| `N` | simulation slope | profile equation | agreement |
| --- | --- | --- | --- |
| 2048 | 4.148459148 | 4.148481857 | 5.5e-6 |
| 4096 | 4.154779433 | 4.154737187 | 1.0e-5 |

Note what does and does not converge. The two methods agree to 1e-5 at matched
resolution, but the shared value moves by 1.5e-3 between `N = 2048` and
`N = 4096`. That is not Newton landing on different branches: re-solving at
4096 from the interpolated 2048 solution gives the same 4.154737187, and
carrying it back down returns 4.148481857 to 2e-12. Each grid has one solution
and the simulation finds that grid's solution, not the continuum's.

**Do not quote nine digits for the continuum value.** Finding 6 shows the
profile is only about `C^2`, so its Fourier series converges algebraically and
so does everything read off it. The honest figure is `||f||_inf = 4.15` with
the third decimal uncertain, from 4.1485 at `N = 2048` and 4.1547 at
`N = 4096`. What is pinned to 1e-5 is the *agreement between two independent
methods at matched resolution*, which is the actual claim being made.

### Selection, not existence

Solutions of `rhs(f) = f` exist over a wide range of `a`, but the dynamics only
converges to them in a window. Running Newton from each simulation's own
profile, and reading the residual *before* Newton moves anything:

| `a` | residual of the simulated profile | predicted vs measured rate |
| --- | --- | --- |
| 0.3 to 0.6 | 0.33 to 0.56 | no agreement, or no solution found |
| 0.65 | 0.19 | 6.1e-2 |
| 0.70 | 5.0e-2 | 3.0e-2 |
| 0.75 | 2.6e-3 | 1.4e-3 |
| 0.80 | 1.3e-5 | 4.1e-6 |
| 0.85 | 5.5e-5 | 8.1e-6 |
| 0.90 | see below | 1.5e-1 |

So the frozen regime is roughly `a` in `[0.75, 0.85]` at the amplitudes
reached, and it degrades smoothly rather than switching off. Below about 0.6
the ansatz simply does not apply, whatever solutions the equation admits, which
is the same boundary finding 2 saw from the spectra.

`a = 0.9` is unresolved. Its residual does not fall monotonically with the
amplitude cap (0.78, 2.93, 1.77 at 1e4, 1e6, 1e8), and the reciprocal slope it
implies wanders with it, so that row is measuring the fit window rather than the
physics. `T(0.9)` is 10.8 against 6.1 at `a = 0.8`, so it has had less time to
settle at any fixed cap; whether the profile survives to `a = 0.9` is open.

Beware the trivial solution. `f = 0` satisfies `rhs(f) = f` exactly, and a
naive continuation in `a` falls into it: the first attempt reported
`||f|| = 0` at `a = 0` with residual 1e-10 and looked like a converged result.
Anything using these routines has to screen for it.

## 5. The profile is linearly stable, with exactly two forced eigenvalues

Renormalise with `s = -ln(T - t)` and `W = (T - t) w`. Since `rhs` is quadratic,
the `e^(2s)` factors cancel and the equation becomes

```
W_s = rhs(W) - W = residual(W)
```

so the profile is a fixed point of that flow and the Jacobian eigenvalues are
growth rates in renormalised time, directly. Two are forced and are not
instabilities:

| eigenvalue | eigenvector | meaning | verified to |
| --- | --- | --- | --- |
| exactly 1 | `f` | freedom to shift the blowup time `T` | 1.3e-12 |
| exactly 0 | `f'` | translation invariance | 4.0e-11 |

The `lambda = 1` mode is provable rather than fitted: `rhs` quadratic gives
`D[rhs](f) f = 2 rhs(f) = 2f`, hence `J f = 2f - f = f`. It is a sharp check on
the assembled Jacobian, and it passes.

Everything else at `a = 0.8` lies in the left half plane, at `N = 512`, 1024
and 2048 alike. **Zero unstable directions beyond the two forced modes.** That
is what "the dynamics selects this profile" means precisely, and it is why a
simulation from `sin x` walks onto it without any tuning.

The same test at profiles taken from each simulation gives `stable` at
`a = 0.65, 0.70, 0.75, 0.80`, alongside profile residuals falling
0.19, 0.050, 2.6e-3, 4.6e-5. Stability and selection agree.

Two things here are not settled. The leading *stable* eigenvalue does not
converge in `N` (-0.54, -0.146, -0.41 at 512, 1024, 2048), so the decay rate
onto the profile is not determined, only its sign. And `a = 0.85` at `N = 1024`
lands on a different branch, `||f|| = 6.249`, which is genuinely unstable at
+2.077, whereas at `N = 2048` the same value of `a` gives `||f|| = 5.668`
matching its simulation to 8e-6. Continuation in `a` is worse still: stepping
from 0.8 to 0.85 jumps branches outright. Branch structure here is real and
unmapped.

## 6. The profile is not analytic. Its spectrum decays like `k^-3.05`

The shape looks harmless: broad, gentle, exactly odd, with `max |f'| = 7.04`
against `||f||_inf = 4.16`, so the steepest feature is about 0.59 wide. Yet its
spectrum is still at 1e-10 by `k = 900`, which no analytic function that
smooth would do. Fitting both laws over the same window:

| `N` | exponential `exp(-delta k)` | `R^2` | algebraic `k^-beta` | `R^2` |
| --- | --- | --- | --- | --- |
| 1024 | `delta` = 0.0248 | 0.827 | `beta` = 3.044 | 0.981 |
| 2048 | `delta` = 0.0128 | 0.805 | `beta` = 3.048 | 0.983 |
| 4096 | `delta` = 0.0065 | 0.788 | `beta` = 3.045 | 0.984 |

The algebraic law wins at every resolution, `beta` holds at 3.045 across a 4x
range in `N`, and the fitted `delta` halves every time `N` doubles, which is
what fitting an exponential to a power law always looks like: the apparent rate
is just a constant over `k_cut`.

So the blowup profile has finite regularity, roughly `C^2`. Three consequences:

- It explains finding 4's slow convergence. Nothing read off this profile
  converges spectrally, so `N = 4096` buys far less than it would for an
  analytic profile.
- It retires `dg.strip_width` for `a > 0`. That function fits a single
  exponential, and for these profiles there is no exponential to fit. The
  freezing evidence in finding 2 stands because it compares normalised spectra
  directly and assumes nothing; the `delta` values quoted alongside it do not.
- `beta` near 3 looked close enough to exactly 3 to be worth testing. It is
  not 3. See finding 7, which replaces the 3.045 quoted above: that value was
  biased, and the true exponent at `a = 0.8` is 3.00, but for a reason that
  makes 3 uninteresting.

## 7. `beta` is not a constant. It is `1 + (c-1)/(ac)`, set by one local number

Rearranging the profile equation,

```
f' / f = (H f - 1) / (a U)
```

so `f` can only be singular where `U` vanishes, at a stagnation point of the
profile's own velocity. Setting `U(y*) = 0` in the equation forces `f(y*) = 0`
too, confirmed to 5e-15. Expanding `U(y) = U'(y*)(y - y*)` and using `U' = H f`,
with `c = H f(y*)`:

```
f'/f = mu / (y - y*),     mu = (c - 1) / (a c),     f ~ C (y - y*)^mu
```

and a `|y|^mu` singularity has Fourier coefficients decaying like `k^-(mu+1)`,
so `beta = 1 + (c - 1) / (a c)`. **One local number predicts the entire
spectral decay rate**, with no fitting window and no bins.

At `a = 0.8` there are two stagnation points, and they are exactly `pi` apart:

| `y*` | `c = H f(y*)` | `f(y*)` | `mu` | `beta` |
| --- | --- | --- | --- | --- |
| 0.974078 | 5.001701 | 4.9e-15 | 1.000085 | 2.000085 |
| 4.115670 | -1.662968 | -7.1e-16 | 2.001668 | 3.001668 |

The equation is first order, so there is one exponent and the general solution
is `C (y - y*)^mu` with `C` free to differ on the two sides. With `mu = 2` and
`C+ != C-`, `f` is `C^1,1` but not `C^2` and `f''` jumps. Measured: `f''` is
exactly odd about `y*` to 1e-9, bounded at 2.27045 on one side and -2.27045 on
the other, so the jump is 4.54089 and `C+ = -C- = 1.13522`. A jump in the
second derivative gives `k^-3` exactly. Not a logarithm: fitting
`f'' = 2D log|y-y*| + const` returns a slope of 0.043 with `R^2 = 0.46`,
consistent with zero.

The two singular points being exactly `pi` apart also explains the beating that
biased the original fit. Two singularities separated by `pi` make `|f_k|`
alternate with period 2 in `k`, which is the band visible at high `k` in
`fig_profile`. Binning geometrically and taking the RMS averages over it, and
the local slopes then scatter around 3 with no drift: 2.91, 2.87, 3.07, 2.96,
2.98, 2.97, 3.01, 3.00, 2.98, 2.98, 2.96, 2.92 across `k` from 16 to 724.

### Then the test that settles it

`mu = 2` requires `c = 1/(1 - 2a)` exactly, and nothing in the equation puts
`c` there. So predict `beta` at other values of `a` and measure it
independently. The predictions span 4.2 to 2.5, far too spread to agree by
luck:

| `a` | `beta` predicted | `beta` measured | rel diff | `R^2` |
| --- | --- | --- | --- | --- |
| 0.72 | 6.795 | 8.292 | 0.220 | 0.953 |
| 0.75 | 4.213 | 4.167 | 0.011 | 0.99988 |
| 0.78 | 3.329 | 3.305 | 0.007 | 0.99999 |
| 0.80 | 3.002 | 2.976 | 0.009 | 0.99999 |
| 0.82 | 2.770 | 2.741 | 0.011 | 0.99998 |
| 0.85 | 2.520 | 2.483 | 0.015 | 0.99997 |

The prediction tracks the measurement to about 1 percent across the range. The
`a = 0.72` row fails exactly where the caveat says it should: once `beta` is
large the spectrum reaches the roundoff floor within a couple of octaves and
there is no range left to fit, which `R^2 = 0.95` flags.

**So `beta` is not 3.** The regularity of the blowup profile varies
continuously with `a`, and `beta = 3` is simply where that curve crosses while
passing through. The apparent 3 at `a = 0.8` came from having measured at one
value of `a` only. The measured column sits about 1 percent below the predicted
one throughout, a systematic bias rather than scatter, most likely residual
beating inside the fit window.

Whether `mu(0.8)` is exactly 2, which would make `a = 4/5` special, is not
resolved: `mu - 2` reads -6.5e-4, 3.9e-3, 1.7e-3 at `N` = 1024, 2048, 4096,
hovering around 1e-3 without converging. Given `dmu/da` is about -17 near
`a = 0.8`, `mu = 2` lands at `a = 0.8001`, which is not distinguishable from
0.8 at this accuracy. There is no evident reason for 4/5 to be special, so the
default reading is coincidence.

## 8. Half of `c` is exact: `c = 1/(1-a)` at the smooth stagnation point

`beta = 1 + (c-1)/(ac)` moves the question to what sets `c = H f(y*)`, which is
a global quantity. For one of the two stagnation points the answer is closed
form, and it needs three lines rather than a computation.

Where `f` has a *simple* zero at a stagnation point, write `U ~ c1 (y - y1)`,
`f ~ A (y - y1)` with `A = f'(y1)` nonzero, and `H f ~ c1`. The profile
equation `a U f' = f (H f - 1)` becomes, at leading order in `(y - y1)`,

```
a c1 (y - y1) A  =  A (y - y1) (c1 - 1)
```

and dividing by `A (y - y1)`:

```
a c1 = c1 - 1        hence        c1 = 1 / (1 - a)
```

Equivalently `1/c1 = 1 - a`, and since `1/c = 1 - a mu` in general, this is
exactly `mu1 = 1`. Measured deviations of `mu1` from 1:

| `a` | 0.70 | 0.72 | 0.74 | 0.76 | 0.78 | 0.80 | 0.82 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mu1 - 1` | -1.7e-15 | -4.6e-14 | -2.0e-10 | 3.5e-8 | 5.6e-6 | 8.5e-5 | 5.3e-4 |

Machine precision at the low end. The growth toward `a = 0.82` is the profile's
own accuracy degrading, since `mu2` is falling toward 1 there and the profile
is getting rougher, not the identity failing.

The same balance at the other stagnation point, with `f ~ C (y - y2)^mu`,
reproduces `a c2 mu = c2 - 1` and hence the general formula. So the whole
structure follows from one leading-order match, and `mu = 1` is simply the case
where `f` is smooth with a nonvanishing derivative.

**What is still not closed.** This pins the stagnation point that carries no
singularity. The one that actually sets `beta` is the other one, and `c2`
remains genuinely global: it is not determined by any local balance, and no
closed form for `mu2(a)` came out of the data.

| `a` | 0.70 | 0.72 | 0.74 | 0.76 | 0.78 | 0.80 | 0.82 | 0.84 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mu2` | 15.362 | 5.795 | 3.737 | 2.836 | 2.329 | 2.002 | 1.770 | 1.594 |
| `1/mu2` | 0.0651 | 0.1726 | 0.2676 | 0.3526 | 0.4294 | 0.4996 | 0.5648 | 0.6272 |

`1/mu2` is smooth and monotone but convex, its slope falling steadily from 5.37
to 3.12. A straight line through it returns `R^2 = 0.991`, which looks
respectable and is worthless: the residuals arch cleanly through
-0.029, -0.001, +0.014, +0.020, +0.017, +0.008, -0.006, -0.023, which is
curvature and not scatter. Extrapolating from two different windows disagrees:

| window | `1/mu2 = 0`, profile turns analytic | `1/mu2 = 1`, profile leaves `C^1` |
| --- | --- | --- |
| `a >= 0.74` | 0.6623 | 0.9418 |
| `a >= 0.78` | 0.6490 | 0.9526 |

So neither endpoint is located. This is the same trap that produced the biased
`beta` in finding 6, a high `R^2` on a model of the wrong shape, and it is
worth stating twice.

### A hypothesis that failed

The measured `beta` sat about 1 percent below prediction at every value of `a`,
systematically rather than randomly, which suggested a cause. The proposal was
that the even `k` modes are far smaller than the odd ones, so that binning by
RMS averages real signal together with near-zeros and drags the exponent down.
That would also have explained the beating.

It is wrong. Over `100 < k < k_cut` the RMS amplitudes are 3.386e-4 on even `k`
against 3.865e-4 on odd, a ratio of 0.876. The even modes are not small, just
12 percent below the odd ones, which is enough to show as a visible band in a
log plot but nowhere near vanishing. And fitting the parities separately moves
the answer the wrong way:

| fit | `beta` | `R^2` |
| --- | --- | --- |
| predicted from `mu2 + 1` | 3.001668 | |
| all modes | 2.975974 | 0.999986 |
| odd `k` only | 2.935612 | 0.999838 |
| even `k` only | 3.059236 | 0.999617 |

Restricting to odd `k` moves *away* from the prediction, not toward it. So the
1 percent shortfall is not parity contamination. It is resolved in finding 9.

## 9. The shortfall was the fit window. In a clean band the agreement is 5e-4

Fitting octave by octave rather than over one wide window shows the local slope
does not converge to the predicted 3.001668. It falls straight through it:

| band | 8 to 32 | 16 to 64 | 32 to 128 | 64 to 256 | **128 to 512** | 256 to 1024 |
| --- | --- | --- | --- | --- | --- | --- |
| `beta` | 3.164 | 3.125 | 3.046 | 3.030 | **3.00315** | 2.973 |
| error | +0.162 | +0.123 | +0.045 | +0.028 | **+0.0015** | -0.029 |

The two ends are biased in *opposite* directions, so a fit spanning both lands
somewhere in between, and 2.976 was that compromise. Raising the lower edge of
the window walks `beta` down through the prediction (3.021, 3.011, 3.002,
2.994, 2.984 for `k_lo` = 8, 16, 32, 64, 128); dropping the top octave walks it
up (3.011, 3.020, 3.028 for `k_hi` = `k_cut`, `k_cut/2`, `k_cut/4`). Neither
end is trustworthy and both were in the original window.

**In the clean band `128 < k < 512`, measured `beta` is 3.00315 against a
predicted 3.001668, an error of 5e-4.** That is a factor of 18 better than the
0.9 percent the wide window reported, and it means the stagnation point formula
is correct to well under a tenth of a percent.

The low `k` bias was predicted in advance and has the right sign. Writing
`f ~ C|y-y*|^mu` times an analytic factor, the analytic factor contributes
`k^-(mu+2)` and faster alongside the leading `k^-(mu+1)`, and those steeper
terms pull a fitted slope upward at small `k`.

The high `k` droop is not what it looked like. The natural suspect was a `k^-2`
component from a jump in `f'` at the `mu = 1` stagnation point, since a
shallower term drags a fit down exactly as observed. That is dead: `f'` is
continuous at **both** stagnation points to 4e-12 and 1.6e-11 against `|f'|` of
7.04 and 0.062 respectively. Neither point contributes `k^-2`.

### Separating the two ends by refinement

The two biases have opposite fingerprints under a change of `N`. Discretisation
lives at a fixed fraction of `k_cut` and slides right as `N` grows; real
structure in the profile lives at a fixed absolute `k` and does not move.
Measuring the same bands at `N = 4096` and `N = 8192` settles both at once.

Aligned by **fraction of `k_cut`**, the high end matches almost exactly:

| `k / k_cut` | 0.094 | 0.188 | 0.375 | 0.750 |
| --- | --- | --- | --- | --- |
| `N = 4096` | 3.02942 | **3.00303** | 2.97238 | 2.78870 |
| `N = 8192` | 3.01507 | **3.00291** | 2.97596 | 2.79399 |

Aligned by **absolute `k`**, the same band cleans up as the mask moves away:

| band | 128 to 512 | 256 to 1024 | 512 to 2048 |
| --- | --- | --- | --- |
| `N = 4096` | 3.00303 | 2.97238 | 2.78870 |
| `N = 8192` | 3.01507 | 3.00291 | 2.97596 |

So the droop is the dealiasing mask, confirmed. The low end goes the other way:
band 32 to 128 gives 3.04635 at `N = 4096` and 3.05277 at `N = 8192`, pinned to
absolute `k`, which is the analytic correction terms and is real.

The clean window is therefore where both are small: far enough above the
corrections in absolute `k`, and far enough below `k_cut` in relative terms.
That is `k / k_cut` near 0.19 at both resolutions, and both give the same
answer:

| | `beta` | error vs 3.001668 |
| --- | --- | --- |
| `N = 4096`, 128 to 512 | 3.00303 | +1.4e-3 |
| `N = 8192`, 256 to 1024 | 3.00291 | +1.2e-3 |

Two resolutions agreeing with the prediction to 0.04 percent, and with each
other to 1e-4.

**That reading was wrong, and finding 10 retracts it.** The agreement is
reproducibility, not accuracy: the same deterministic window bias applied to
two similar spectra returns the same biased answer. The measurement is not
good to 1e-4, or to 1e-3, and the residual quoted here is not a real quantity.

## 10. Retraction: the measurement is only good to about 2e-2

Chasing the `+1.3e-3` residual established that it does not exist to be chased.

Three checks, in order of how much they moved.

**Root finding was never the issue.** Polishing the zero of `U` by Newton on
the spectral interpolant, rather than linear interpolation of the sign change,
changes `mu2` by 0 to 4e-16. The linear estimate was already landing on the
zero to machine precision, `|U(y*)| = 5.7e-16`. So the 4.6e-3 spread of `mu2`
across `N` is the profile's own algebraic convergence and cannot be reduced
this way.

**Which profile is measured barely matters.** Simulated against Newton solved
at `N = 4096` gives 3.003025 against 3.003150 over the same band, a difference
of 1.3e-4, and their spectra differ by 2e-4 relative. Worth checking, since the
prediction reads `c` off one particular profile, but not the explanation.

**The estimator is the whole story.** The window `128 to 512` gives 3.003150.
The window `0.094 to 0.375 k_cut`, which is 128.31 to 511.88 and the same
window to three digits, gives 2.986260. With three or four bins in a fit, which
modes land in which bin dominates the slope. Across ten reasonable windows:

| estimator | range | spread |
| --- | --- | --- |
| binned, RMS per bin | 2.9829 to 3.0072 | 2.4e-2 |
| raw, least squares over every mode | 3.0132 to 3.0510 | 3.8e-2 |

Unbinning does not help, it hurts, and it biases the other way: the binned RMS
is energy weighted while an unbinned log fit is a geometric mean, and where the
spectrum has scatter those two diverge. The ranges barely overlap. The raw
estimator sits `+2.25e-2` above the prediction on average, and its gap is flat
across a 4x range in `N`:

| `N` | 2048 | 4096 | 8192 |
| --- | --- | --- | --- |
| gap | +1.68e-2 | +1.54e-2 | +1.64e-2 |

A discretisation error shrinks under refinement. This does not, which is what
makes it the estimator rather than the grid.

### The honest numbers

| quantity | value | limited by |
| --- | --- | --- |
| `beta` predicted | 3.002 +/- 0.005 | resolution, since `f` is only `C^1,1` |
| `beta` measured | 3.00 +/- 0.02 | choice of estimator and window |

They agree, and nothing finer is resolvable with this machinery. The `+1.3e-3`
residual was more than an order of magnitude below the measurement's own noise
floor. The prediction is the better estimate of the two, by a factor of four,
which inverts how findings 6 and 7 read: the formula is not being checked
against a more reliable measurement, it *is* the more reliable number, and the
spectral fits merely fail to contradict it.

### Practical consequence

Every spectral exponent quoted anywhere in this repository should be read as a
band measurement, not a global fit, and the band has to sit clear of both ends.
The wide window fits in findings 6 and 7 are all biased for this reason, which
is why the measured column there sits about 1 percent low at every value of
`a`, systematically rather than randomly. The prediction was right and the
measurement was wrong, which is the opposite of how it looked.

## 11. The `pi` separation is exact structure, not a coincidence

The two stagnation points came out exactly `pi` apart at every `a` tested, to
ten digits, which was noted in finding 7 and left unexplained. It is forced,
and the reason is a parity statement.

If `U(y1) = 0` and `U(y1 + pi) = 0`, expand both in Fourier modes. The second
condition is `sum_k U_k e^(i k y1) (-1)^k = 0`. Adding it to the first kills the
odd modes and leaves `sum_{k even} U_k e^(i k y1) = 0`; subtracting kills the
even ones. So the even mode part and the odd mode part of `U` must each vanish
at `y1` **separately**. That is two conditions where a generic zero of `U`
satisfies one, so it cannot happen by accident.

Checked directly at `N = 4096`, `a = 0.8`, against a scale of `|U| = 3.48`:

| point | `U_even` | `U_odd` |
| --- | --- | --- |
| `y1 = 0.974077800` | -1.0e-15 | +1.1e-15 |
| `y2 = 4.115670454` | -2.9e-15 | +2.5e-15 |

Both halves vanish at both points, at machine precision. The separation is
`pi` to within 9e-16.

This is not the same as `f` having only odd modes, which it does not: `|f_2|` is
3315 against `|f_1|` of 6520, so the mode 2 content is comparable to mode 1. The
symmetry lives in where `U` vanishes, not in the spectrum.

## 12. `c2` to five digits, using `c1` as a control variate

No closed form for `c2` is likely. Writing `f = i(psi' - psi~')` and
`U = psi + psi~` with `psi` the positive frequency part, the equation reads

```
a (psi + psi~)(psi'' - psi~'') = (psi' - psi~')(psi' + psi~' - 1)
```

and the mixed products `psi psi~''` and `psi~ psi''` straddle both halves of
the spectrum. That refusal to separate is the whole difficulty of the model,
and it is why `a = 0` collapses to a Riccati equation and `a != 0` does not.

Extrapolating in `N` fails too. Seven resolutions give `c2` bouncing between
-1.6486 and -1.6681 with no monotone trend, so a three parameter fit returns
exponent 3.3, amplitude 1.1e7 and residual 4.2e-3, which is a fit to noise.

The lever is finding 8. `c1 = 1/(1-a)` **exactly**, so its measured wander is a
direct gauge of the discretisation error in reading anything off the profile.
`c2` comes off the same profile and carries the same error, entering linearly,
so `c2` is a linear function of `c1` across resolutions and evaluating that
line at the exact `c1` cancels the error without needing its size or its rate.

At `a = 0.8`, regressing over `N` = 512 to 4096:

```
c2 = -0.980361 c1 + 3.240505,     R^2 = 0.99999744
```

| `N` | `c1 - 5` | `c2` raw | `c2` corrected |
| --- | --- | --- | --- |
| 512 | -1.30e-2 | -1.64861379 | -1.66131558 |
| 1024 | +6.93e-3 | -1.66810312 | -1.66131304 |
| 2048 | -3.34e-3 | -1.65801989 | -1.66129143 |
| 4096 | +1.70e-3 | -1.66296777 | -1.66130056 |

The raw spread of 1.95e-2 collapses to 2.43e-5, a factor of **801**. Even
`N = 512` lands within 1.5e-5 of the value from `N = 4096` once corrected.

```
c2   = -1.66130027 +/- 1.0e-05
mu2  =  2.00242268 +/- 4.6e-06
beta =  3.00242268
```

### `beta = 3` is now excluded

`mu2 = 2` exactly would require `c2 = 1/(1-2a) = -1.66666667`. The corrected
value misses that by 5.37e-3, which is 500 times the scatter of the corrected
points. `mu2 - 2 = +2.42e-3` against a scatter of 4.6e-6.

Treat that factor as a margin rather than a p value: the scatter measures how
consistently the correction reproduces itself across resolutions, not how much
residual systematic it leaves behind. But the raw error was 1e-2 and the
correction removed 800 times it, so even allowing a residual systematic ten
times the observed scatter, `mu2 = 2` stays excluded by two orders of
magnitude.

This closes the question finding 7 opened and finding 10 could not settle. The
prediction is now 3.0024227 rather than 3.002 +/- 0.005, a thousandfold
improvement, and it comes from seven cheap solves rather than one expensive
one. The spectral measurement of `beta` remains good only to 2e-2 and is now
irrelevant to the answer.

### It transfers, and it matters most where the profile is roughest

`mu1 = 1` is exact for every `a`, so the correction is not special to 0.8:

| `a` | `c2` corrected | scatter | `mu2` | `beta` | `beta` before |
| --- | --- | --- | --- | --- | --- |
| 0.75 | -0.70940840 | 1.4e-10 | 3.2128337 | 4.2128337 | 4.212834 |
| 0.78 | -1.22477829 | 1.1e-07 | 2.3288132 | 3.3288132 | 3.328739 |
| 0.80 | -1.66130027 | 9.5e-06 | 2.0024227 | 3.0024227 | 3.001668 |
| 0.82 | -2.20020937 | 7.2e-06 | 1.7737832 | 2.7737832 | 2.770433 |

The scatter degrades with `a` for a reason that checks out: at `a = 0.75` the
profile has `beta = 4.2`, so its own spectrum decays fast enough that the
discrete solve nearly converges spectrally, while by `a = 0.82` everything is
algebraic. The size of the correction follows the same pattern, moving the
seventh digit at `a = 0.75` and the third at `a = 0.82`.

Read the scatter as a consistency check rather than a full error bar. It
measures how well the correction reproduces itself across resolutions, and a
residual nonlinear dependence of the error on `c1` would not show up in it. The
1.4e-10 at `a = 0.75` should not be taken at face value as ten digit accuracy.

**`beta(a)` is now known to between five and ten digits at four values of `a`,
which is precise enough to test a closed form decisively rather than
suggestively.** None of the obvious candidates fits. That is the sharpest form
of the open question in finding 8: `c1` is exact, `c2` is now merely very well
measured, and the gap between those two statuses is where the remaining
mathematics sits.

## 13. No closed form for `c2`, but the equation integrates once and a global identity falls out

**The closed form was not found, and the simplest candidates are now excluded
rather than merely unconfirmed.** What the search produced instead is worth
more than another fitted constant.

### The equation integrates exactly once

From `f'/f = (U' - 1)/(a U)`,

```
ln|f| = (1/a) [ ln|U| - integral dy/U ]
```

so on each interval between zeros of `U`,

```
f = C |U|^(1/a) exp( -(1/a) integral dy/U )
```

Near a simple zero with `U ~ c (y - y*)` this gives
`|y - y*|^(1/a) |y - y*|^(-1/(ac)) = |y - y*|^mu` with `mu = (c-1)/(ac)`,
reproducing finding 7 from a different direction.

### Hence an exact global constraint

Going once around the circle, both `f` and `U` return to themselves, so the
loop integral must vanish:

```
PV integral of dy/U over the circle = 0
```

This needs `ln|f|` not to jump at either zero, which holds: at `y1` the profile
is smooth with `mu = 1`, and at `y2` the one sided constants satisfy
`C+ = -C-`, so `|C+| = |C-|`.

Verified by subtracting both simple poles with `cot`, which carries the same
residue and whose own principal value on the circle is zero:

| `a` | 0.75 | 0.78 | 0.80 | 0.82 |
| --- | --- | --- | --- | --- |
| PV / scale | 2.2e-12 | 6.7e-13 | 9.8e-13 | 1.4e-12 |

Stable across three independent grid shifts. The first attempt read 1e-2 to 1
and looked like a refutation; it was catastrophic cancellation, because the
stagnation points land on grid points to within 1e-11 of a cell and the
subtraction was differencing two numbers of size 1e8. Evaluating on a half
shifted grid, which is exact for a band limited field, fixes it.

### What is excluded

`c2` is now mapped across eleven values of `a` by the control variate, with
scatter from 1.4e-14 at `a = 0.72` to 9.5e-5 at `a = 0.83`:

| `a` | 0.72 | 0.74 | 0.76 | 0.78 | 0.80 | 0.82 |
| --- | --- | --- | --- | --- | --- | --- |
| `c2` | -0.31519119 | -0.56651088 | -0.86560395 | -1.22477889 | -1.66130509 | -2.20024724 |
| `beta` | 6.7953854 | 4.7367448 | 3.8358720 | 3.3288127 | 3.0024205 | 2.7737736 |

Against that, fitting candidate shapes by least squares:

| form | rms residual | verdict |
| --- | --- | --- |
| `mu2 = (p + qa)/(1 + ra)` | 1.9e-4 | rejected |
| `c2 = (p + qa)/(1 + ra)` | 3.0e-4 | rejected |
| `mu2 = (p + qb)/(1 + rb)`, `b = 1/(1-a)` | 6.0e-4 | rejected |
| `mu2` or `c2` = quadratic / quadratic | 2.6e-7 to 1.7e-6 | unconstrained |

Every Mobius form is out, in both natural variables, by one to two orders of
magnitude above the data's own precision. The quadratic over quadratic fits are
not evidence of anything: five parameters against eleven points, with residuals
below the data scatter at the rough end of the range, which means they are
absorbing the smooth error rather than the signal.

### Why this is probably the wrong thing to look for

`c1` is exact because it comes from a **local** balance at a point where `f` is
smooth. `c2` has no such balance available: the exponent there is not fixed by
the leading order match, so its value is set by the global solution. A closed
form for it would amount to solving the nonlocal problem, and the obstruction
to that is explicit. With `psi` the positive frequency part,

```
a (psi + psi~)(psi'' - psi~'') = (psi' - psi~')(psi' + psi~' - 1)
```

and the mixed products `psi psi~''` and `psi~ psi''` straddle both halves of the
spectrum. That is exactly the non separation which makes `a = 0` a Riccati
equation and `a != 0` an open problem. If `c2` had an elementary closed form,
De Gregorio would not be hard.

## 14. The basin at `a = 0.8`, and a third outcome

Three facts shrink the search before it starts. The nonlinearity is quadratic,
so `w_0 -> lambda w_0` rescales time and nothing else: the basin is scale
invariant and only shape matters. Profiles come in a family `f(nx)`, so "which
profile" means "which member", read off as half the number of stagnation
points. And one signed data does not blow up at all, so a boundary exists.

### Member 1 is the generic attractor

Sixteen random smooth data, modes 1 to 8 with amplitudes falling like `1/k` and
random phases, between 4 and 10 sign changes: **all sixteen reach member 1**,
with `||f||_inf = 4.148482` in every case, identical to seven digits. The
number of sign changes of the datum does not select the member.

Higher members are reached only from data with the matching exact symmetry:
`sin(kx)` gives member `k` for `k = 1,2,3,4`, with `2k` stagnation points whose
exponents alternate 1 and 2. Since `sin(kx)` is exactly `2 pi / k` periodic, it
stays in that symmetric subspace forever. Break the symmetry and it falls back
to member 1, which is what all the random data do.

### The offset boundary is exactly at a sign change

For `w_0 = sin x + m`, which is sign changing for `m < 1` and one signed at
`m = 1` where the zero at `-pi/2` becomes a touching point:

| `m` | 0 | 0.5 | 0.8 | 0.9 | 0.95 | 0.99 | 1.0 | 1.2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| outcome | member 1 | member 1 | member 1 | member 1 | member 1 | member 1 | none | none |
| `T` | 6.07 | 10.07 | 15.52 | 20.28 | 25.88 | 43.81 | - | - |

The profile is completely insensitive to `m`: `||f||_inf = 4.148482` for every
value that blows up. The blowup time diverges as `m -> 1` and blowup stops
exactly where the sign change does, so this boundary is topological. We fitted
the divergence and report no law: neither `-log(1-m)` nor `(1-m)^-p` is clean,
their implied constants drifting monotonically, and this repository has a poor
record with extrapolations of that kind.

### A third outcome, on an open set

For `w_0 = sin x + c\,sin 2x` the two members compete, and between them is
something neither:

| `c` | 0.46 | 0.475 | 0.48 | 0.50 | 0.52 | 0.56 | 0.58 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outcome | member 1 | member 1 | member 1 | decay | decay | member 2 | member 2 |
| `T` | 61.4 | 90.4 | 108.2 | - | - | 54.9 | 44.0 |

`T` diverges on approach from both sides. In the window the solution does not
blow up: at `c = 0.5` the amplitude falls monotonically to `7.4e-3` by
`t = 400`, with `||w||_inf \cdot t` settling on 3.09, 3.03, 2.99, 2.97, so it
**decays like `1/t`**. That is the blowup rate with the sign reversed, which is
what the same profile family gives when the singularity sits in the past rather
than the future.

The window is open, not a separatrix: `c = 0.5` and `0.52` both fail to blow up
while `0.48` and `0.56` both do. So sign changing data need not blow up, which
nothing else here predicts, and the model admits global decaying solutions from
an open set. The window is narrow, roughly `0.49 < c < 0.55`.

### The interior is not self-similar

Decay like `1/t` invites an obvious guess. If `w ~ g/t` then `w_t = -g/t^2`
while `R(w) = R(g)/t^2`, so `R(g) = -g`, and since `R` is quadratic
`R(-g) = R(g) = -g`, meaning `h = -g` solves `R(h) = h`. The decay attractor
would be a profile entered with the opposite sign, and `||w||_inf t = 2.95`
would be its amplitude, matching neither member 1 at 4.148482 nor member 2 at
4.161220. The stability inverts too: with `s = log t` and `W = tw` the flow is
`W_s = R(W) + W`, whose linearisation is `-J` for the blowup `J`, so such a
profile would have to be strongly unstable as a blowup profile.

**That is wrong, on two independent tests.**

Newton started from `h = -tw` at `t=400` walks to member 1, returning
`||h||_inf = 4.1484819` with member 1's constants and spectrum. The guess
carried an 8 percent residual in `R(h) = h`, so it was never near a solution
and Newton simply found the nearest one.

Running to `t = 2000` shows why. The amplitude law is clean and converging:

| `t` | 100 | 400 | 1400 | 2000 |
| --- | --- | --- | --- | --- |
| `\|\|w\|\| t` | 3.02517 | 2.97152 | 2.95307 | 2.95045 |
| decay exponent | - | 1.0102 | 1.0036 | 1.0025 |
| residual of `-tw` | 6.4e-2 | 2.4e-1 | 2.2 | 4.9 |
| spectrum drift | - | 4.7e-3 | 1.9e-2 | 4.6e-2 |

The exponent converges to 1 and `||w||_inf t` converges to about 2.950, so the
amplitude really does follow `1/t`. But the shape does not freeze. The
normalised spectrum drifts steadily and the residual of `-tw` grows by two
orders while its sup norm stays fixed, which means fine structure accumulating
at high wavenumber rather than a profile settling.

So the decay is asymptotically `1/t` in amplitude and not self-similar in
shape. Whatever organises the interior of the window is not a fixed profile,
and we do not identify it. The same mechanism explains why `c = 0.52` went
under resolved at `t = 42`: it is developing the same fine structure faster.

An earlier pass at `N = 1024` put the window at `0.48` to `0.56`, wrongly:
every run there stopped near `t = 17.5` rather than at the time limit, for a
reason the script never printed, and the classification rested on the amplitude
at that moment rather than on an outcome.

### What this settles and what it does not

The genericity objection is largely answered. The profile is reached from
sixteen random data, from a continuous family of offsets, and from three
designed data, always with the same constants to seven digits. It is not an
artefact of `sin x`.

What is not answered is the shape of the basin boundary. Two families were
probed, one at a time, in a space of functions. The decay window shows the
boundary is not simply "does the datum change sign", and nothing here maps it
in more than one dimension.

## 15. The critical limit `a -> 1`: an exact reduction, and why it is not enough

At `a = 1` every multiple `A sin x` is steady, since
`R_1(A sin) = A^2 sin(-cos) - A^2(-sin)(cos) = 0`. The critical case has a
**line** of equilibria, destroyed for `a < 1`. Measuring `T` by
`T = t + ||f||/||w||`, exact for a frozen profile and needing no fit:

| `a` | 0.80 | 0.90 | 0.93 | 0.95 | 0.96 | 0.97 | 0.98 | 0.99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `T` | 6.0719 | 10.7569 | 14.0844 | 17.7129 | 20.4591 | 24.5830 | 32.1786 | 56.9742 |
| local `p` | - | 0.828 | 0.755 | 0.681 | 0.646 | 0.638 | 0.664 | **0.824** |

**The exponent does not converge.** It falls from 0.83 to 0.64 and then climbs
back to 0.82 at the smallest `eps`. An intermediate reading of "about 0.65"
taken from the middle of that range was premature; the last point overturns it.
All that survives is `T eps` decreasing monotonically throughout, 1.214 down to
0.570, so `T` grows strictly slower than `1/eps`, which already excludes the
naive drift argument.

No functional form fits. Over the last points, `C eps^-p` gives rms 0.99,
`C/eps + D` gives 0.71, `C log(1/eps) + D` gives 4.22, and
`A/eps + B log(1/eps) + C` gives 0.38, all large against `T` values of 6 to 57.
The asymptotic regime has not been reached by `eps = 0.01`, and reaching
`eps = 0.001` would need roughly 30 times the 866,000 steps that point already
took.

### The parity hypothesis, refuted

The guess was that mode 2 feeds back into mode 1 only at quadratic order, the
linear term vanishing by parity, which would give `p = 2/3`. Doing the
calculation kills it. With `omega = A sin x + B sin 2x`,

```
omega H omega = -(A^2/2) sin2x - AB sin3x - (B^2/2) sin4x
u omega_x     = -(A^2/2) sin2x - (5AB/4) sin3x + (3AB/4) sin x - (B^2/2) sin4x
```

The `sin x` terms cancel in the first but not the second, so
`R_a = omega H omega - a u omega_x` has coefficients

| mode | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- |
| coefficient | `-(3a/4)AB` | `-(eps/2)A^2` | `(5a/4-1)AB` | `-(eps/2)B^2` |

verified against the code to `1e-17`. **The linear feedback is present**, with
coefficient `-3a/4`. There is no parity cancellation and no route to 2/3.

### What the reduction does give

The two mode system `A' = -(3a/4)AB`, `B' = -(eps/2)A^2` integrates exactly.
With `L = log A` at `a = 1`, `L' = -(3/4)B` and `L'' = (3 eps/8) e^{2L}`, a
Liouville equation. Multiplying by `L'` with `L(0)=L'(0)=0` gives
`L'^2 = (3 eps/8)(e^{2L}-1)`, and since
`integral dL / sqrt(e^{2L}-1) = arccos(e^{-L}) -> pi/2`,

```
T = (pi/2) sqrt(8 / (3 eps)) = 2.56510 eps^(-1/2)
```

Checked against a two mode Galerkin run, the ratio `T / (2.5651 eps^(-1/2))`
reads 1.11803, 1.05409, 1.02598, 1.01015, 1.00504 at
`eps = 0.2, 0.1, 0.05, 0.02, 0.01`. The reduction is exact and gives
`p = 1/2`.

### The exponent is a cascade effect

Neither 1 nor 1/2 nor 2/3 is the answer, and the reason is visible in how the
truncated exponent depends on how many modes are retained:

| modes | 2 | 4 | 6 | 8 | 12 | 24 | full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `p` | 0.474 | 0.506 | 0.518 | 0.525 | 0.536 | 0.554 | ~0.65 |

The exponent climbs steadily with mode count and is still climbing at 24 modes.
So `p` is set by the cascade past mode 2, no finite reduction produces it, and
the two mode Liouville result is a lower bound on the mechanism rather than the
mechanism.

The three mode truncation is degenerate and should not be read as part of that
trend: it gives `T = 405` at `eps = 0.1` and no blowup at all at 0.05 or 0.02,
so adding exactly mode 3 suppresses the instability that modes 2 and 4 sustain.

**Status of `p`.** Not determined. The local exponent oscillates between 0.64
and 0.83 across `0.01 <= eps <= 0.15` with no trend, so no value should be
quoted for the full equation.

What is settled is narrower and worth separating from what is not:

- `p = 1`, from the naive drift argument, is excluded. `T eps` falls
  monotonically by a factor of two across the range.
- `p = 2/3` is excluded by the calculation itself, not by numerics: the linear
  feedback coefficient is `-(3a/4)AB` and does not vanish.
- `p = 1/2` is **exact** for the two mode system, with the constant
  `(pi/2)sqrt(8/3) = 2.56510`, confirmed to 0.5 percent.
- The full exponent exceeds 1/2 and is set by the cascade, since the truncated
  value climbs monotonically with mode count and is still climbing at 24 modes.

The gap between the last two is the open question. A reduction that captures
the cascade, rather than a truncation that discards it, is what would close it.

## 16. Why no finite reduction exists: the linearisation has no spectral gap

A reduction of the escape from the line of equilibria needs a spectral gap:
fast modes slaved to slow ones so the fast ones can be eliminated. Writing
`omega = A sin x + v` and using that `R` is quadratic, so
`DR_1(A sin) = A M` with `M = DR_1(sin x)` fixed,

```
v_t = A M v - (eps/2) A^2 sin 2x,      A' = -(3/4) A b_2
```

If `M` were strictly stable, `v` would saturate at `eps A M^{-1} sin 2x / 2`,
giving `A' = kappa eps A^2`, a Riccati equation, and `p = 1`. The measured `p`
is below 1, so that step must fail. It does, and structurally.

### `M` has purely imaginary spectrum

Computed densely at `N` = 128, 256 and 512: every eigenvalue has real part at
the `1e-5` roundoff level against imaginary parts reaching 72, and at `N = 256`
there are **no** eigenvalues with positive real part at all. The spectrum is
`+/- 0.799i`, `+/- 2.12i`, `+/- 2.89i`, `+/- 22.1i`, `+/- 39.0i`, `+/- 72.2i`
and so on. The kernel is confirmed exactly: `M sin x` and `M cos x` are
`8e-14` and `6e-14`, these being the tangent directions to the manifold,
amplitude and translation. The 170 further near-zero eigenvalues at `N = 512`
are the dealiasing mask, since `k_cut = N/3 = 170`, not physics.

### It is not diagonalisable, and the growth is drift

Integrating `v_t = M v` from a random perturbation, `||v||` grows **linearly**:
ratios to the initial norm of 187, 736, 1392, 2873, 5738 at
`t = 10, 50, 100, 200, 400`, doubling as `t` doubles. Linear growth with
imaginary spectrum means a Jordan block at zero.

Splitting the state confirms what generates it:

| `t` | 0 | 10 | 50 | 100 | 200 | 400 |
| --- | --- | --- | --- | --- | --- | --- |
| along `sin x` | 0 | 86.6 | 341.2 | 646.2 | 1333.6 | 2663.0 |
| along `cos x` | 0 | -27.4 | -106.5 | -204.8 | -421.1 | -844.8 |
| remainder | 7.79 | 139.2 | 346.5 | 149.5 | 466.8 | 135.2 |

The kernel components grow linearly while the remainder stays bounded and
oscillates. So the secular growth is **drift along the manifold of equilibria**
rather than instability: a perturbation changes the amplitude and phase at a
constant rate, moving the state to a different steady state rather than away
from all of them. The Jordan block is generated by the symmetries themselves.
This is consistent with the stability theorem of Jia, Stewart and Sverak, which
is stability modulo exactly these symmetries.

Worth noting separately: `M` is strongly non-normal. The remainder is amplified
from 7.79 to 467 before returning to 135, a transient factor of about 60 with
no eigenvalue to account for it. Transient amplification of that size is
dynamically relevant even though the spectrum is neutral.

### The consequence

There is no spectral gap anywhere. Modulo the kernel every mode oscillates
without decay, and the kernel modes drift linearly. Therefore:

- **quasi steady elimination is invalid**, since the transient it discards
  never dies, which is why the Riccati argument giving `p = 1` fails;
- **mode truncation is not a reduction**, since the discarded modes never relax
  either, which is why the truncated exponent converges only like `m^(-1/2)`
  and was still climbing at 24 modes;
- **there is no centre manifold to reduce onto**, because the centre subspace
  is the whole space.

So the failure to build a finite reduction is structural, not a matter of
finding the right ansatz. The cascade cannot be eliminated because nothing in
it is fast relative to anything else. Any successful treatment of the
`a -> 1` limit will have to handle the full spectrum at once, which is what the
two mode Liouville result, exact and wrong, was always going to miss.

## 17. The exact solution tests the general self similar remark, and it holds

Finding 3 gives an exact solution at `a = 1/2`, so the obvious move is to test
`c = 1/(1-a) = 2` there. It fails, and the failure is the informative kind: `2`
was never the prediction at `a = 1/2`.

Taking `v -> infinity` in the closed form near the blowup point `x = pi`, with
`z = x - pi` and `zeta = v z / 2`, and using `dv/dt -> v^4/32` so that
`v^3 = 32/(3(T - t))`:

```
lambda = 1/3   exactly,      w = (T - t)^-1 F(zeta)

F(zeta)  = -(16/3) zeta / (1 + zeta^2)^2
HF(zeta) = -(8/3) (zeta^2 - 1) / (zeta^2 + 1)^2
U(zeta)  =  (8/3) zeta / (zeta^2 + 1)
```

so `a = 1/2` narrows with exponent `1/3` and is nowhere near frozen, which
finding 2 already said qualitatively with its 0.49 profile residual. The
transport velocity in the moving frame is `V = lambda zeta + a U`, not `a U`,
and `V = (zeta/3)(zeta^2 + 5)/(zeta^2 + 1)` has its only zero at `zeta = 0`,
the blowup point itself, where `F` has a simple zero. The leading order match
there gives `1 + lambda + a c = c`, hence

```
c = (1 + lambda) / (1 - a) = (4/3) / (1/2) = 8/3
```

`pi_stagnation.py` measures `c = (T - t) Hw(pi, t)` off the exact solution with
the solver's own Hilbert transform:

| `v` | `N` | `c` measured | vs `8/3` | vs `2` |
| --- | --- | --- | --- | --- |
| 10 | 2^14 | 2.6719924062 | 5.3e-3 | 6.7e-1 |
| 100 | 2^16 | 2.6667199992 | 5.3e-5 | 6.7e-1 |
| 1000 | 2^20 | 2.6666671136 | 4.5e-7 | 6.7e-1 |

Converging on `8/3` and standing off `2` by two thirds at every resolution. The
self similar equation `F + lambda zeta F' + a U F' = F HF` holds to 1.3e-15,
and `v^3 (T - t) -> 32/3` confirms `lambda = 1/3`.

**This is not a correction to the paper.** Remark `rem:general` already carries
the full ansatz `w = (T-t)^{c_w} Om(x/(T-t)^{c_l})`, its profile equation
`(c_l X + a U) Om_X = (c_w + U_X) Om`, the exponent
`nu = (c_w + h)/(c_l + a h)`, and the statement that a simple zero forces
`h = (1 + c_l)/(1 - a)` with Proposition `prop:c1` as the `c_l = 0` case. The
dictionary is `c_l = lambda = 1/3`, `c_w = -1`, `h = c = 8/3`, and
`nu = (5/3)/(5/3) = 1` as a simple zero requires.

What is new is that the remark now has evidence. Every measurement behind
findings 7, 8, 10 and 12 was taken on frozen profiles, where `c_l = 0` and the
general formula is indistinguishable from `prop:c1`, so the `c_l` dependence
was the one part of the local analysis carrying no support whatsoever. At
`a = 1/2` no frozen profile exists, `c_l = 1/3`, the two formulas differ by
33 percent, and the exact solution picks the general one. That is a stronger
check than anything in the frozen regime, because it is against a closed form
rather than against a simulation.

Worth noting separately: `F` is analytic on the real line, with its nearest
singularities at `zeta = +-i`. So the narrowing profile is smooth where the
frozen profiles of finding 6 are only `C^{1,1}` with `k^-3.05` spectra. The
roughness is a feature of the frozen regime, not of blowup in this family.

## 18. The overlap search, and what it cost

Findings 1 to 17 were recorded before the pole dynamics school was found. That
school works on the circle, owns the exact solutions, and searches organised
around the other seven papers do not surface it. Everything below is now cited
in the paper.

**Lushnikov, Silantyev and Siegel, arXiv:2010.01201, contains:** the frozen
ansatz `w = f(x)/(T-t)`, their Eq. (9), as `alpha = 0`, for
`a_c < a <= 0.95` on the circle; its numerical profile for `a_c < a <= 0.85`,
solved by Petviashvili iteration; the jump in `w_xx` at `x = +-pi`, antipodal
to the singularity, at `a = 0.8`; the resulting algebraic spectral decay, whose
exponent they call `p_b` and we call `beta`, with `p_b -> infinity` as
`a -> a_c^+`; `gamma = 1/(1-a)` for the leading complex singularity, from
`a gamma = gamma - 1`, the same algebra as finding 8; the width exponent
`alpha(a)`, our `c_l`, to 5 to 8 digits, with
`a_c = 0.6890665337007457`; `alpha = 1/3` at `a = 1/2`; and nonlinear stability
of the frozen profile, inferred from convergence in simulations.

**Xu, arXiv:2607.19762, 22 July 2026, contains:** the general self similar
profile equation, same `c_l` symbol; `H Om(0) = (1 + c_l)/(1 - a)` derived by
differentiating it at the stagnation point under `Om'(0) != 0`, which is
finding 17's relation argument for argument; and the spectrum of the
linearisation at `a = 0` on the line, point spectrum `{0,1}` of symmetry modes.
He states the formula in passing and claims no novelty for it.

**Silantyev, Lushnikov, Siegel and Ambrose, arXiv:2411.01891:** the exact
`a = 1/2` solution of finding 3, whose Remark 2 flags the `v_c(0) = 1`
degeneracy that finding 3 clears.

### Retraction

The paper claimed the narrowing to frozen changeover satisfies `a_c >~ 0.795`,
from `c_l = 0.004536 +/- 0.000543` at `a = 0.751` being eight standard errors
from zero. **Withdrawn.** Against the published branch the width measurement
reads 1.058 vs 1.0000 at `a = 0`, 0.748 vs 0.7474 at 0.2, 0.512 vs 0.4809 at
0.4, 0.325 vs the exact 1/3 at 0.5, and 0.179 vs 0.1691 at 0.6. A few percent
method cannot resolve 0.0045. The true value is 0.68907, confirmed
independently by Xu's Newton continuation crossing zero at 0.6888.

### What survives, and the number that sharpens

Finding 8 at the **non-simple** zero, which Xu does not cover and which turns
their fitted `p_b` into a predicted quantity; the parity proof of finding 11;
the control variate of finding 12 and the exclusion of `beta = 3`; the linear
stability spectrum of finding 5; the PV identity; `T = pi`.

`beta_lss.py` quantifies the correction. Their one value recoverable from
running text is `p_b = 9.32592` at `a = 0.71`, from a two domain spectral fit.
The local relation gives:

| `a` | 0.70 | 0.71 | 0.72 | 0.75 |
| --- | --- | --- | --- | --- |
| `beta` | 16.36200 | 9.29546 | 6.79539 | 4.21283 |

so **9.29546 against their 9.32592, lower by 0.33 percent**. The run reproduces
finding 12 at `a = 0.75` to 3e-8 and finding 8's `mu2` at 0.70 and 0.72 to five
decimals, so the new rows are worth the same as the old ones. Their Table 2
holds the rest of `p_b(a)` but does not render in the arXiv PDF and ar5iv
truncates before it.

Their `p_b` diverges as `a -> a_c^+` and so does this `beta`. Extrapolating
`1/beta` to zero from the pairs (0.70, 0.71) and (0.71, 0.72) gives 0.6868 and
0.6828, the closer pair larger, pointing at 0.68907 and not at 0.795. Convex
data again, so direction only, but it is a second and independent route to the
retraction above.

## Caveats

`sin x` is not generic. It is exactly the `a = 1` ground state, so all of this
describes how that ground state destabilises when advection is weakened, which
is narrower than "when does this family blow up". `sweep.py --ic tilted` and
`--ic onesigned` exist to test how much survives a different datum, and neither
has been run past a smoke test.

Nothing here has been checked against the literature. Okamoto, Sakajo and
Wunsch studied this family numerically in 2008 and Chen, Hou and Huang proved
finite time blowup for De Gregorio on the line with low regularity data in
2021. Findings 1 and 2 may well be known; finding 3 should be searched for
before any effort is spent deriving it.

## Next

1. Done, see finding 4. The profile equation is solved and confirmed.
2. Push the continuum limit. `N = 4096` pins `||f||_inf` to about 1e-5 and
   `N = 8192` would confirm it. Worth doing before quoting the number anywhere,
   since the `N = 2048` value is wrong in the third decimal.
3. Settle `a = 0.9`: run to a fixed fraction of `T` rather than a fixed
   amplitude, so runs at different `a` are compared at the same stage of their
   approach rather than the same height.
4. Bracket the narrowing to frozen changeover between `a = 0.6` and `a = 0.75`,
   and find out whether it is sharp. The profile residual is a much better
   order parameter for this than the spectral measurements in finding 2, since
   it needs no fitted decay rate.
5. Compute the spectrum of the Jacobian at the solved profile. It is already
   assembled in `profile_eq.jacobian`, and its eigenvalues decide stability,
   which is what "the dynamics selects this profile" means precisely. That is
   also the quantity a computer assisted proof would need to enclose.
6. Done, see finding 3. `T(1/2) = pi` is exact, and the literature search is
   what settled it: `a = 1/2` is exactly solvable and the closed form gives
   `T = pi` in three lines. `pi_exact.py` verifies it against the solver at
   machine precision.
7. Done, see finding 17. `c = 2` is refuted and Remark `rem:general` is
   confirmed instead, with `c_l = 1/3` and `c = 8/3` to 4.5e-7. Carried into
   the paper: the remark is now Proposition `prop:general` with a proof,
   `T = pi` is Proposition `prop:pi`, and Section `sec:exacthalf` is the exact
   test. 25 pages, compiles clean.
8. Done, see finding 18. arXiv:2010.01201 read in full, arXiv:2607.19762 in
   part. Still unread: Schochet, arXiv:2207.07548, and Xu's sections 3 to 7.
   The paper now cites ten related works and states plainly what is not ours.
9. Get LSS Table 2. It holds `p_b(a)` across the frozen window and would let
   the correction in finding 18 be quoted at more than one value of `a`. It
   does not render in the arXiv PDF and ar5iv truncates; the published version
   or a request to the authors is the way in.
