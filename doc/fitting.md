# Fitting parameters

`digimouth.optimization` fits model parameters to experimental curves by non-linear least
squares (`scipy.optimize.least_squares`).

| Function | Use it to |
|---|---|
| `optim_param_dynamic` | fit one experimental curve |
| `optim_param_dynamic_multi` | fit one parameter set to several curves at once |
| `optim_param_dynamic_dict` | same as `_multi`, with the curves given as dictionaries |
| `bootstrap_params` | estimate parameter uncertainty by residual bootstrap |
| `grid_search_initial_guess` | find a starting point by brute-force search |
| `calculate_residuals` | compute the residual vector for a given parameter vector |
| `optim_param_protocol` | fit a protocol with instantaneous events (swallow, chew…) and/or several stages |

## Fitting one curve

```python
import numpy as np
from digimouth import Model
from digimouth.optimization import optim_param_dynamic

model = Model({"C": {"k": 0.2}}, {"C": lambda y, t, k: -k * y["C"]})

t_exp = np.linspace(0, 10, 60)
y_exp = np.exp(-0.5 * t_exp)

fit = optim_param_dynamic(
    t_exp, y_exp,
    y0_dict={"C": 1.0},
    model=model,
    param_keys_to_optimize=["k"],
    initial_params_to_optimize=[0.3],
    bounds=([0.01], [5.0]),
    output_key="C",
    normalization=False,
)
```

Important points:

- **The model is simulated on the experimental time grid** `t_exp`. The initial condition
  `y0_dict` therefore applies at `t_exp[0]`.
- `output_key` is the state variable compared with the data (default `"CMS"`, the
  instrument signal).
- `initial_params_to_optimize` must be given, in the same order as
  `param_keys_to_optimize`.
- `bounds` is a tuple `(lower_values, upper_values)`, one value per fitted parameter.

### Result

`optim_param_dynamic` returns a dictionary:

| Key | Content |
|---|---|
| `opt_param` | fitted values (array, same order as `param_keys_to_optimize`) — `None` if the fit failed |
| `ssr`, `rmse` | sum of squared residuals, root mean squared error |
| `opt_params_dict` | the model's `params_dict` with the fitted values |
| `opt_model` | a new `Model` with the fitted values, ready to simulate |
| `covariance`, `stderr` | parameter covariance and standard errors |
| `conf_intervals` | confidence intervals, one `(lower, upper)` pair per parameter |

## Parameter names

Parameters are identified **by name only**. Fitting `"k"` updates `k` in every equation
whose parameters contain `k`. This is what you want for a physical constant used in
several equations (a partition coefficient, an exchange area…).

A consequence: two different quantities must have two different names. You cannot fit a
`k` of one equation independently of a `k` of another equation.

## Normalisation

- `normalization=True` (default): the simulated and the experimental curves are each
  divided by their own maximum before comparison. Only the **shape** of the curve is
  fitted; parameters that only change the amplitude cannot be identified.
- `normalization=False`: raw values are compared; use it when model and data share the
  same units.

## Initial conditions that depend on parameters

`y0_dependencies` maps a state variable to a function computing its initial value from the
current parameters, for example a gas phase initially at equilibrium with the liquid:

```python
y0_dependencies = {"Cg": lambda params, y0: params["Cg"]["Kaw"] * y0["Cl"]}
```

`params` is the current `params_dict` and `y0` the initial conditions.

## Confidence intervals

- `ci_method="t"` (default): intervals from the Jacobian at the optimum and Student's t
  distribution, at level `conf_level` (default 0.95).
- `ci_method="bootstrap"`: residual bootstrap with `n_boot` resamples;
  `boot_random_state` fixes the random seed for reproducibility.

`bootstrap_params(...)` can also be called directly; it returns the bootstrap samples, their
standard deviation and percentile intervals.

## Several experiments at once

`optim_param_dynamic_multi` accepts **lists** of time and intensity arrays, one per
experiment (replicates, subjects…). One parameter set is fitted to all of them: the
residuals of all experiments are concatenated. If `initial_params_to_optimize` is omitted,
the values stored in the model are used as the starting point.

```python
from digimouth.optimization import optim_param_dynamic_multi

fit = optim_param_dynamic_multi(
    [t_exp, t_exp], [y_exp, 0.9 * y_exp],
    y0_dict={"C": 1.0}, model=model,
    param_keys_to_optimize=["k"], bounds=([0.01], [5.0]),
    output_key="C", normalization=True,
)
```

`optim_param_dynamic_dict(time_exp_dict, intensity_exp_dict, **kwargs)` does the same with
dictionaries `{experiment_name: array}`.

## Finding a starting point

```python
from digimouth.optimization import grid_search_initial_guess

best_model, best_params, best_cost, all_costs = grid_search_initial_guess(
    [np.linspace(0.05, 2.0, 40)],                       # candidate values of each parameter
    dict(time_exp=t_exp, intensity_exp=y_exp, y0_dict={"C": 1.0}, model=model,
         param_keys=["k"], output_key="C", normalization=False),
)
```

The second argument holds the arguments of `calculate_residuals`. `all_costs` is a
DataFrame with the mean squared residual of every combination.

## Solver options

All fitting functions accept `ode_solver`, `ode_method` and `solve_ivp_options`, with the
same meaning as `solver`, `method` and `solve_ivp_options` in
[simulation.md](simulation.md#choosing-the-solver).

## Fitting a protocol: events and stages

`optim_param_dynamic` simulates the model continuously, **without events**. To fit a curve
shaped by swallows or chews, or a protocol made of several stages, use `optim_param_protocol`. It
simulates with `run_stages` (see [simulation.md](simulation.md)), events included, and
returns the same kind of result as `optim_param_dynamic`.

```python
import pandas as pd
from digimouth.optimization import optim_param_protocol

swallows = pd.DataFrame({"time": [30.0, 62.5], "type": ["swallow", "swallow"]})

fit = optim_param_protocol(
    t_exp, y_exp,
    stages=model,                          # one Model, or a list of stages (see below)
    y0_dict=y0,
    params={"kL": 0.1, "KAL": 1e-3},       # parameters to fit and their starting values
    bounds={"kL": (1e-4, 1.0)},            # optional, per parameter
    events=swallows,                       # columns "time" and "type"
    event_params={"VOA": 37.0, "VFA": 30.0, "VNA": 11.0, "VOL_m": 1.0},
    output_key="CMS",
)
fit["opt_params"], fit["conf_intervals"], fit["rmse"]
```

**Stages.** Instead of one model, `stages` can be a list of stages, each with its own model:

```python
stages = [
    {"name": "mouth", "t_start": 0,  "t_end": 60,  "model": model_mouth},
    {"name": "after", "t_start": 60, "t_end": 120, "model": model_after},
]
```

The experimental times must lie within the stages.

**Parameter names.** `"kL"` is shared by every stage where it appears. `"kL@after"` only
concerns the stage named `after`, so one parameter can take a different value in each stage:
`params={"kL@mouth": 0.1, "kL@after": 0.1}`.

**Several experiments.** Pass lists of curves, as for `optim_param_dynamic_multi`. `stages`,
`y0_dict`, `events` and `event_params` can be shared, or given as one value per experiment
(for instance the swallow times of each subject).

**Result.** `opt_param`, `opt_params` (`{name: value}`), `ssr`, `rmse`, `covariance`, `stderr`,
`conf_intervals` (`{name: (low, high)}`), `opt_stages` (the stages with the fitted models,
ready for `run_stages`), `success` and `message`.

Normalisation is the same as in `calculate_residuals`. Without events and with a single
stage, `optim_param_protocol` gives the same result as `optim_param_dynamic` (this is tested).

The older `optim_param_dynamic_multiphases` does not work with the current stage format and
is deprecated: use `optim_param_protocol`.
