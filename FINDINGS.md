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
narrowing regime. Why its amplitude nonetheless follows a power law clean
enough to place `T` to eleven digits is unexplained.

The observation is unaffected by the failed explanation. It converges across an
8x range in `N`, which is evidence that `T = pi` exactly for `a = 1/2` from
`w_0 = sin x`. It is not a proof and no derivation is offered. It either falls
out of an exact solution at `a = 1/2` or is a coincidence at the eleventh
decimal, and finding out which is a well posed question.

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
6. Settle `T(1/2) = pi`, starting with a literature search. The explanation
   originally offered for its precision is refuted, so it is now purely an
   observation.
