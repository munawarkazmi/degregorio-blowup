# De Gregorio blowup

> Everything here blows up in finite time, including the author. Fittingly,
> most of the damage came from fitting: twice this repo handed me
> `R^2 = 0.99` on a model of the wrong shape, and twice I fell for it. I also
> refuted one of my own exact identities before noticing the refutation was a
> cancellation error. Finding 10 retracts finding 9.

A pseudospectral solver and profile analysis for the Okamoto, Sakajo and Wunsch
family of 1D models for the 3D Euler vorticity equation, on the circle:

```
w_t + a u w_x = w u_x
u_x = H w,   mean(u) = 0
```

`H` is the Hilbert transform, and `a` tunes how much advection fights vortex
stretching. This family isolates the one thing that makes 3D Euler hard.
Stretching (`w u_x`) drives blowup; advection (`a u w_x`) fights it by carrying
the growing peak out of the straining region. At `a = 0` the fight is over
before it starts and there is a closed-form singularity. At `a = 1` the two
terms are in real competition, and that competition is the 1D shadow of the
Euler problem.

**Both endpoints are known exactly**, which is what makes everything here
checkable:

| `a` | model | exact fact |
| --- | --- | --- |
| 0 | Constantin, Lax and Majda (1985) | `w = 4 w_0 / [(2 - t H w_0)^2 + t^2 w_0^2]`, so from `sin x` the blowup time is `T = 2` and the rate exponent is 1 |
| 1 | De Gregorio (1990) | `sin x` is a steady state, since `u w_x = w u_x` pointwise, so `T` is infinite |

## Quick start

```bash
pip install numpy matplotlib
```

```bash
python validate.py
```

19 checks in about 35 seconds, every one against a closed form or a conserved
quantity. Then:

```bash
python mechanism.py && python profile_branch.py && python spectrum.py
```

## Results at a glance

![blowup times](fig_blowup.png)

Blowup for every `a < 1` with rate exponent 1, and `T(a)` running away as
`a` approaches 1. No transition to global existence appears below `a = 1`.

| `a` | 0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `T` | 2 exact | 2.3101 | 2.7858 | 3.1416 | 3.6500 | 4.4641 | 6.0719 | 10.757 | 17.713 | inf exact |
| `p` | 1 exact | 0.997 | 1.000 | 1.001 | 1.001 | 0.999 | 1.000 | 1.049 | unreliable | n/a |

### Two structurally different blowups

![frozen versus narrowing spectra](fig_frozen.png)

Comparing normalised spectra `|w_k| / max|w_k|` across decades of amplitude
growth is translation invariant and assumes nothing about the shape. At
`a = 0.8` the curves lie on top of one another: the profile is frozen and only
its height diverges. At `a = 0.4` they spread: the peak is narrowing.

| | small `a` | large `a` |
| --- | --- | --- |
| behaviour | peak narrows | profile freezes |
| spectra across 4 decades | move by 0.72 in one decade | converge to 3.8e-3 and stop |
| peak location | drifts | locks to `x = 2.16699` |
| resolvability | exhausted by `\|\|w\|\| = 5.8e3` | tail stays at 2.7e-17 to `\|\|w\|\| = 1e6` |

### The blowup profile solves a fixed point equation

Substituting `w = f(x - x_p) / (T - t)` gives `f / (T-t)^2` on the left and
`rhs(f) / (T-t)^2` on the right, so the profile equation is just

```
rhs(f) = f
```

a fixed point of the same nonlinearity the solver already validates. It has no
free scaling, since `cf` gives `c^2 rhs(f) = c^2 f`, which equals `cf` only at
`c = 1`. **The equation fixes the amplitude**, and because
`d/dt (1/||w||) = -1/||f||_inf`, that amplitude is independently measurable
from a simulation.

![profile, convergence and spectrum](fig_profile.png)

At `a = 0.8` the simulated profile satisfies `rhs(f) = f` to 1.3e-5 before
Newton runs at all. The two computations, time marching a PDE and Newton on an
algebraic fixed point, share nothing but the value of `a`:

| `N` | simulation slope | profile equation | agreement |
| --- | --- | --- | --- |
| 2048 | 4.148459148 | 4.148481857 | 5.5e-6 |
| 4096 | 4.154779433 | 4.154737187 | 1.0e-5 |

### The profile is linearly stable, and that is why it is selected

Renormalising with `s = -ln(T - t)` and `W = (T - t) w` turns the equation into
`W_s = rhs(W) - W`, so the profile is a fixed point and **the Jacobian
eigenvalues are growth rates in renormalised time, directly**.

![Jacobian spectrum](fig_spectrum.png)

Two eigenvalues are forced by symmetry. The `lambda = 1` mode is provable, not
fitted: `rhs` quadratic gives `D[rhs](f) f = 2 rhs(f) = 2f`, hence `J f = f`.

| eigenvalue | eigenvector | meaning | verified to |
| --- | --- | --- | --- |
| exactly 1 | `f` | freedom to shift the blowup time | 1.3e-12 |
| exactly 0 | `f'` | translation invariance | 4.0e-11 |

Everything else lies in the left half plane at `N = 512`, 1024 and 2048 alike.
Zero unstable directions beyond the two forced modes.

### The profile has finite regularity

The shape looks harmless: broad, exactly odd, `max |f'| = 7.04`. But its
spectrum is still at 1e-10 by `k = 900`, which no analytic function that smooth
would do. Fitting both laws over the same window:

| `N` | exponential `R^2` | algebraic `R^2` | `beta` |
| --- | --- | --- | --- |
| 1024 | 0.827 | **0.981** | 3.044 |
| 2048 | 0.805 | **0.983** | 3.048 |
| 4096 | 0.788 | **0.984** | 3.045 |

The algebraic law wins at every resolution, so the profile has finite
regularity and nothing read off it converges spectrally.

### And `beta` is not a constant. It is predicted by one local number

Rearranging the profile equation gives `f'/f = (Hf - 1)/(a U)`, so `f` can only
be singular where the velocity `U` vanishes. Setting `U(y*) = 0` forces
`f(y*) = 0` too, confirmed to 5e-15. Expanding around such a point with
`c = Hf(y*)`:

```
f ~ C (y - y*)^mu,   mu = (c - 1) / (a c),   so   beta = 1 + (c - 1) / (a c)
```

![beta against a](fig_beta.png)

At `a = 0.8` this gives `mu = 2.0017`, so `f` is `C^1,1` but not `C^2` and
`f''` jumps. Measured: `f''` is exactly odd about `y*` to 1e-9, bounded at
`+2.27045` and `-2.27045`, a jump of 4.54089. A jump in the second derivative
gives `k^-3` exactly.

Predicting `beta` from that one number and measuring it independently from the
spectrum, across a range where the predictions span 4.2 to 2.5:

| `a` | 0.75 | 0.78 | 0.80 | 0.82 | 0.85 |
| --- | --- | --- | --- | --- | --- |
| predicted | 4.213 | 3.329 | 3.002 | 2.770 | 2.520 |
| measured | 4.167 | 3.305 | 2.976 | 2.741 | 2.483 |
| rel diff | 0.011 | 0.007 | 0.009 | 0.011 | 0.015 |

Those measurements are about 1 percent low at every `a`, systematically rather
than randomly, and the cause is the fit window rather than the formula. Fitting
octave by octave, the local slope falls straight through the prediction instead
of converging to it:

| band | 8 to 32 | 32 to 128 | 64 to 256 | **128 to 512** | 256 to 1024 |
| --- | --- | --- | --- | --- | --- |
| `beta` | 3.164 | 3.046 | 3.030 | **3.00315** | 2.973 |

The two ends are biased in opposite directions, so any window spanning both
lands in between. Refining `N` separates them, because discretisation sits at a
fixed fraction of `k_cut` and slides right as `N` grows while real structure
sits at a fixed absolute `k`. Aligned by `k / k_cut` the high end matches
across resolutions, 2.789 against 2.794 at 0.75 and 2.972 against 2.976 at
0.375, so the droop is the dealiasing mask. Aligned by absolute `k` the low end
matches instead, 3.046 against 3.053 over 32 to 128, so that bias is the
analytic factor multiplying `|y-y*|^mu`, which contributes `k^-(mu+2)` and
faster. Both effects identified, neither is the `k^-2` term that looked
obvious: `f'` is continuous at both stagnation points to 4e-12.

The clean window is where both are small, near `k / k_cut = 0.19`, and there
`N = 4096` and `N = 8192` give 3.00303 and 3.00291 against a predicted
3.001668.

**Do not read that as agreement to 0.04 percent.** It is reproducibility, not
accuracy. Applying the same window to the same estimator on two similar spectra
returns the same biased answer, which says nothing about how close either is to
the truth. Shifting the window from `128 to 512` across to
`0.094 to 0.375 k_cut`, the same window to three digits, moves the answer from
3.00315 to 2.98626. Across ten reasonable windows:

| estimator | range | spread |
| --- | --- | --- |
| binned, RMS per bin | 2.9829 to 3.0072 | 2.4e-2 |
| raw, every mode | 3.0132 to 3.0510 | 3.8e-2 |

Unbinning hurts rather than helps, and biases the other way, since RMS binning
is energy weighted while an unbinned log fit is a geometric mean. So the honest
figures are:

| quantity | value | limited by |
| --- | --- | --- |
| `beta` predicted | 3.002 +/- 0.005 | resolution, `f` is only `C^1,1` |
| `beta` measured | 3.00 +/- 0.02 | choice of estimator and window |

They agree, and nothing finer is resolvable here. The prediction is the better
number of the two by a factor of four, so the spectral fits do not confirm the
formula so much as fail to contradict it.

So the regularity of the blowup profile varies continuously with `a`, and
`beta = 3` is just where that curve happens to cross. The earlier reading of a
constant 3.045 came from measuring at one value of `a`, with a wide fit window
biased by beating: the two stagnation points sit exactly `pi` apart, which makes
`|f_k|` alternate with period 2 in `k`.

### Half of `c` is exact, and needs three lines rather than a computation

There are two stagnation points. Where `f` has a **simple** zero at one, put
`U ~ c(y-y_1)`, `f ~ A(y-y_1)` with `A` nonzero, and `Hf ~ c`. Then
`a U f' = f(Hf - 1)` reads `a c A (y-y_1) = A (y-y_1)(c-1)` at leading order,
so dividing by `A(y-y_1)`:

```
a c = c - 1        hence        c = 1 / (1 - a)
```

which is `mu = 1`. Measured deviations of `mu` from 1 at that point:

| `a` | 0.70 | 0.72 | 0.74 | 0.76 | 0.78 | 0.80 |
| --- | --- | --- | --- | --- | --- | --- |
| `mu - 1` | -1.7e-15 | -4.6e-14 | -2.0e-10 | 3.5e-8 | 5.6e-6 | 8.5e-5 |

Machine precision. The same balance at the other stagnation point reproduces
the general formula, so `mu = 1` is just the case where `f` is smooth there.
That point therefore carries no singularity, and `beta` is set entirely by the
other one, whose `c` stays genuinely global.

No closed form for that one came out of the data. `1/mu2` runs 0.0651, 0.1726,
0.2676, 0.3526, 0.4294, 0.4996, 0.5648, 0.6272 across `a` from 0.70 to 0.84,
smooth and monotone but convex, with its slope falling from 5.37 to 3.12. A
straight line fits at `R^2 = 0.991` and extrapolates uselessly: two different
windows put the point where the profile turns analytic at `a = 0.662` and
`a = 0.649`. Same trap as the biased `beta` above, a high `R^2` on a model of
the wrong shape.

### Getting `c2` to five digits anyway

`c1 = 1/(1-a)` exactly, so its measured wander across resolutions is a direct
gauge of the discretisation error in reading anything off the profile. `c2`
comes off the same profile and carries the same error linearly, so regressing
`c2` against `c1` and evaluating at the exact `c1` cancels it. At `a = 0.8`
over `N` = 512 to 4096 the regression is `c2 = -0.980361 c1 + 3.240505` at
`R^2 = 0.99999744`, and the raw spread of 1.95e-2 collapses to 2.43e-5, a
factor of **801**. Even `N = 512` then lands within 1.5e-5 of `N = 4096`.

```
c2   = -1.66130027 +/- 1.0e-05
mu2  =  2.00242268 +/- 4.6e-06
beta =  3.00242268
```

**So `beta = 3` is excluded.** `mu2 = 2` would need `c2 = 1/(1-2a) =
-1.66666667`, and the corrected value misses it by 5.37e-3, some 500 times the
scatter of the corrected points. The spectral measurement of `beta` is good
only to 2e-2 and no longer matters to the answer.

### The equation integrates once, and `c2` has no closed form

From `f'/f = (U' - 1)/(aU)`, on each interval between zeros of `U`,

```
f = C |U|^(1/a) exp( -(1/a) integral dy/U )
```

Requiring `f` to close up around the circle then forces an exact constraint,
`PV integral of dy/U = 0`, verified to 1e-12 at four values of `a` and three
independent grid shifts.

No closed form for `c2` emerged. With `c2` mapped across eleven values of `a`
it is now precise enough to **exclude** shapes rather than fail to confirm
them: every Mobius form, in `a` and in `1/(1-a)`, for both `c2` and `mu2`, is
out by one to two orders of magnitude above the data's precision.

That is probably the right answer rather than a gap. `c1` is exact because it
comes from a local balance where `f` is smooth; `c2` has no such balance, so
its value is set by the global solution. Pinning it in closed form would mean
solving the nonlocal problem, whose obstruction is explicit: with `psi` the
positive frequency part the equation reads
`a(psi + psi~)(psi'' - psi~'') = (psi' - psi~')(psi' + psi~' - 1)`, and the
mixed products straddle both halves of the spectrum. If `c2` were elementary,
De Gregorio would not be hard.

Full detail, including what is not settled, is in [FINDINGS.md](FINDINGS.md).
The technical write up is in [paper/](paper/degregorio.pdf).

If none of the above meant anything to you, start here instead:
**[Sharper, or taller?](docs/explainer/degregorio-blowup-explained.pdf)**, six
pages, no mathematics assumed. It covers the same results and the same
retractions, and explains what a blowup is before claiming to have measured
one.
A paper draft written up from those findings is in
[paper/degregorio.tex](paper/degregorio.tex).

```bash
cd paper && pdflatex degregorio.tex && pdflatex degregorio.tex
```

## How the pieces fit together

```mermaid
flowchart TD
    A["dg.py<br/>pseudospectral solver"] --> B["validate.py<br/>19 checks vs closed forms"]
    A --> C["mechanism.py<br/>power law or exponential?"]
    C --> D["profile.py<br/>normalised spectra:<br/>frozen or narrowing?"]
    D --> E["profile_eq.py<br/>solve rhs(f) = f<br/>by Newton"]
    E --> F["profile_branch.py<br/>which a does the<br/>dynamics select?"]
    E --> G["spectrum.py<br/>Jacobian eigenvalues<br/>= stability"]
    E --> H["regularity.py<br/>is f analytic?"]
    A --> I["diagnose.py<br/>grid refinement,<br/>the arbiter"]
```

## Files

| file | what it does |
| --- | --- |
| `dg.py` | solver, spectral operators, diagnostics, exact solutions |
| `validate.py` | 19 checks against closed forms and conserved quantities |
| `profile_eq.py` | the profile equation `rhs(f) = f` and its Newton solver |
| `sweep.py` | the `T(a)` experiment, writes `sweep.csv` |
| `mechanism.py` | growth law: power law versus exponential |
| `profile.py` | normalised spectra, the frozen versus narrowing test |
| `profile_branch.py` | which `a` the dynamics selects a profile for |
| `profile_unique.py` | one solution per grid, or several |
| `spectrum.py` | Jacobian eigenvalues, linear stability |
| `regularity.py` | is the profile analytic, and how smooth is it |
| `beta.py` | local spectral slope by octave, binned over the beating |
| `beta_theory.py` | derives `beta` from the stagnation points |
| `beta_jump.py` | the jump in `f''`, and whether `mu = 2` is structural |
| `beta_predict.py` | predicted against measured `beta` across `a` |
| `pi_check.py` | whether `T(1/2) = pi` survives refinement |
| `diagnose.py` | grid refinement study, for when a result looks too good |
| `figures.py` | regenerates the figures above |
| `explainer_figure.py`, `explainer_figure2.py` | the two figures used only by the plain-language guide |
| `docs/explainer/` | the plain-language guide, source and PDF |
| `paper/` | the technical write up, source and PDF |

`profile_solve.py`, `profile_refine.py` and `shape_check.py` are earlier passes,
kept because their working parts stand and their failures are instructive: the
first fell into the trivial solution `f = 0` while continuing in `a`, and the
second chased a grid maximum artefact that turned out not to be one.

## Numerics

Pseudospectral in space, products formed on the grid with 2/3 dealiasing and
the Nyquist mode zeroed. Classical RK4 in time, with two constraints on the
step:

- **Accuracy.** `dt = cfl / max(|u_x|, |w|)`. Stretching drives the growth, so
  the step must shrink like `1 / ||w||` as the peak sharpens.
- **Stability.** The advective term has eigenvalues out to `i a u k_cut`, and
  RK4 is stable on the imaginary axis only to 2.828, so
  `dt < c_adv / (k_cut |a| u_max)`.

Getting that second bound wrong is expensive in a way that looks like nothing.
The finite difference instinct, `dt = cfl dx / u_max`, is about 67 times
smaller than necessary at `cfl = 0.02`, and it applies a penalty at `a = 0`
where the advection term is not present at all. Runs that should take ten
seconds take ten minutes and still produce correct answers, so nothing appears
to be wrong.

Three diagnostics deserve their own note, because each was wrong at first in a
way that produced plausible numbers:

| diagnostic | the trap |
| --- | --- |
| `sup_norm` | the grid maximum is wrong by `O(dx^2 f'')`, and near blowup the peak narrows like `1/||w||`, so the error grows exactly when it matters. Newton on `f'(x) = 0` against the interpolant fixes it for `O(N)` per point |
| `strip_width` | fitting the decay rate over the whole upper band measures the roundoff plateau, not the decay. Restricting to genuinely decaying decades fixes that, but see `regularity.py`: for `a > 0` there is no exponential to fit in the first place |
| `tail_energy_fraction` | calibrated on `a = 0`, where the spectrum is exponential. It is not a reliable guard for algebraically decaying spectra |

`w_max` defaults to 1e8 because that is where the clock runs out, not the
physics. Near a power law singularity `dt ~ cfl / ||w||` while `t` is of order
`T`, so `t + dt` stops meaning anything once `cfl / ||w|| < eps T`, around
`||w|| = 1e13`.

### Fitting a blowup

If `w ~ C (T - t)^-p` then `w / (dw/dt) = (T - t) / p`, a straight line in `t`
with slope `-1/p` and root `T`. One regression gives both numbers and the
residual says whether the power law fits at all.

The window matters more than the regression. A power law is only the leading
asymptotics, so any window of finite width in `t` carries an `O(width)` bias.
`fit_blowup` selects by amplitude rather than record fraction, and the honest
way to quote a result is two window sizes with the trend shown. On the exact
`a = 0` data, one decade gives `T = 1.99998, p = 0.9946` and half a decade
gives `T = 1.999997, p = 0.9981`.

## What is checked

| # | check | result |
| --- | --- | --- |
| 1 | `sin x` steady at `a = 1` over `t = 50` | 4e-13 |
| 2 | `a = 0` against the CLM closed form at `t = 1.8` | 2e-11 field, 6e-12 sup |
| 3 | mean vorticity conserved, `a` in `{0, 0.5, 1, -0.5}` | 1e-16 |
| 4 | spatial convergence follows `exp(-0.2887 k_cut)` | across four orders |
| 5 | time convergence is fourth order | ratios 16.65, 16.28, 16.16, 16.07 |
| 6 | resolution guard calibrated, not assumed | 1e-3 tighter buys 65x then 123x |
| 7 | strip width against `arccosh((4 + t^2)/4t)`, and `w * delta -> 1/2` | 3e-5, and 0.50254 |

## Caveats

`sin x` is not a generic datum. It is exactly the `a = 1` ground state, so this
describes how that ground state destabilises when advection is weakened, which
is narrower than "when does this family blow up". `sweep.py --ic tilted` and
`--ic onesigned` exist to test what survives a different datum, and neither has
been run past a smoke test.

Nothing here has been checked against the literature. Okamoto, Sakajo and
Wunsch studied this family numerically in 2008, and Chen, Hou and Huang proved
finite time blowup for De Gregorio on the line with low regularity data in
2021. Several of these findings may well be known.

Nothing here is a proof. A computer assisted proof in this area, in the style
of Chen and Hou, needs interval arithmetic producing certified enclosures, not
floating point plus convergence studies. This is the instrument you use to find
the scenario worth trying to prove, and the objects it has produced, a profile
equation with a stable solution, are the objects such a proof would need to
enclose.
