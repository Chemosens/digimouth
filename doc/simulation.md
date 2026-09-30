# Simulation

All simulation functions live in `digimouth.dynamic_model` and return a **pandas
DataFrame** with a `time` column followed by one column per state variable, in the order of
the keys of `y0_dict`.

| Function | Use it to |
|---|---|
| `resolution_odeint_generic` | integrate a model on a given time grid, without events |
| `run_model` | simulate one time segment, with instantaneous events |
| `run_stages` | chain several segments (stages), possibly with different models |
| `resolution_dsolve_generic` | solve a small system **symbolically** (SymPy) |

The examples below use this model:

```python
import numpy as np
import pandas as pd
from digimouth import Model
from digimouth import dynamic_model as dm

decay = Model(
    params_dict={"C": {"k": 0.5}},
    funcs_dict={"C": lambda y, t, k: -k * y["C"]},
    event_funcs={"refill": lambda y, params_current=None: {"C": 1.0}},
)
```

## Integrating on a time grid

```python
t = np.linspace(0, 10, 101)
df = dm.resolution_odeint_generic(decay, {"C": 1.0}, t_sim=t)
```

The first time of `t_sim` is the time of the initial condition.

## Instantaneous events

```python
events = pd.DataFrame({"time": [2.0, 6.0], "type": ["refill", "refill"]})

df = dm.run_model(
    decay, {"C": 1.0}, t_start=0, t_end=10,
    df_event=events, event_func_dict=decay.event_funcs,
    n_points=100,
)
```

- `df_event` needs a `time` and a `type` column; `type` must be a key of
  `event_func_dict` (an unknown type raises `ValueError`). Other columns are allowed and
  ignored by the engine. Events are sorted by time.
- The model is integrated up to each event, the event function is applied, and
  integration restarts from the new state.
- The state **just after** each event is recorded at the event time, so the jump is
  visible in the curves. Each event time appears twice in the output (the recorded state,
  then the start of the next segment); use `df.drop_duplicates("time")` if you need
  strictly increasing times.
- `n_points` is the number of points of **each segment between two events**, not of the
  whole simulation.
- `event_params` (a dictionary) is passed to every event function as `params_current`.

## Multi-stage protocols

A protocol can be split into consecutive stages, each with its own model (for example
"product in mouth", then "after swallow" with other parameters):

```python
after = Model({"C": {"k": 0.1}}, {"C": lambda y, t, k: -k * y["C"]})

stages = [
    {"t_start": 0, "t_end": 5,  "model": decay},
    {"t_start": 5, "t_end": 20, "model": after},
]
df = dm.run_stages(stages, {"C": 1.0}, n_points=100)
```

- Each stage starts from the **last state of the previous stage**.
- Events are given once for the whole protocol (`df_event`); each stage receives the
  events with `t_start <= time < t_end`.
- All stages must share the same state variables.
- The time at a stage boundary appears twice (end of one stage, start of the next).
- The docstring of `run_stages` mentions an `"events"` key per stage; it is not used.
  Give the events through `df_event`.

## Choosing the solver

Every simulation function accepts the same solver options:

| Option | Default | Meaning |
|---|---|---|
| `solver` | `"odeint"` | `"odeint"` (SciPy LSODA wrapper, historical default) or `"solve_ivp"` |
| `method` | `"LSODA"` | Integration method for `solve_ivp` (`"RK45"`, `"BDF"`, `"Radau"`…) |
| `rtol`, `atol` | `1e-6`, `1e-12` | Relative / absolute tolerances |
| `mxstep` | `0` | Maximum internal steps, `odeint` only (`0` = SciPy default) |
| `solve_ivp_options` | `None` | Extra `solve_ivp` arguments, e.g. `{"max_step": 0.1}` |

`odeint` stays the default so that existing results are reproduced exactly. Both solvers
return the same DataFrame layout, so switching is a one-argument change:

```python
df = dm.run_model(decay, {"C": 1.0}, t_start=0, t_end=10, solver="solve_ivp", method="BDF")
```

## Symbolic solution

For small linear systems, `resolution_dsolve_generic(y0_dict, funcs_dict, params_dict)`
returns `{variable: sympy_expression}` of time `t`.

The equations must be written specifically for it: a state variable is written
`y["C"](t)` (a SymPy function of time) instead of `y["C"]`, and only operations SymPy
understands are allowed (no `numpy` calls, no `if` on the state). The equations used for
numerical simulation therefore cannot be reused as they are (see
[known_issues.md](known_issues.md)).

```python
sol = dm.resolution_dsolve_generic(
    {"C": 1.0},
    {"C": lambda y, t, k: -k * y["C"](t)},
    {"C": {"k": 0.5}},
)
# {'C': 1.0*exp(-0.5*t)}
```
