# Models

A model in `digimouth` is a system of ordinary differential equations written **one
equation per state variable**, plus optional **instantaneous events** (swallow, chew…)
that change the state at given times.

## The `Model` object

```python
from digimouth import Model

model = Model(params_dict, funcs_dict, event_funcs=None, name=None)
```

| Argument | Type | Meaning |
|---|---|---|
| `params_dict` | `{variable: {parameter: value}}` | Parameters passed to the equation of each variable |
| `funcs_dict` | `{variable: function}` | Derivative of each state variable |
| `event_funcs` | `{event_type: function}` | Optional instantaneous events |
| `name` | `str` | Optional label |

Initial conditions are **not** stored in the model: they are passed to the simulation
functions as `y0_dict`. This lets you reuse one model with many initial states.

The keys of `y0_dict`, `funcs_dict` and `params_dict` must be exactly the same set of
variable names; otherwise the engine raises a `ValueError` that lists the missing or
extra names.

For backward compatibility, `model["params_dict"]` works like `model.params_dict`.

## Writing an equation

Each equation is a Python function with the signature

```python
def dX_dt(y, t, param1, param2, ...):
    return ...          # value of dX/dt
```

- `y` is a dictionary with the **current value of every state variable**
  (`y["COA"]`, `y["CMS"]`…), so an equation can depend on any other variable.
- `t` is the current time (useful for time-dependent inputs such as a breathing flow).
- The remaining arguments are the parameters. They are passed **by name** from
  `params_dict[variable]`. Every name in `params_dict[variable]` must be an argument of the
  function, otherwise the engine raises a `ValueError` showing the expected signature.

Example — a gas/liquid exchange with a measurement compartment:

```python
from digimouth import Model

def dCg_dt(y, t, Kaw, k, A, Vr):
    return k * A / Vr * (Kaw * y["Cl"] - y["Cg"])

def dCl_dt(y, t, Kaw, k, A, Vl):
    return -k * A / Vl * (Kaw * y["Cl"] - y["Cg"])

def dCMS_dt(y, t, tMS):
    return (y["Cg"] - y["CMS"]) / tMS

exchange = Model(
    params_dict={
        "Cg":  {"Kaw": 1e-3, "k": 1e-4, "A": 1e-3, "Vr": 1e-3},
        "Cl":  {"Kaw": 1e-3, "k": 1e-4, "A": 1e-3, "Vl": 1e-4},
        "CMS": {"tMS": 1.0},
    },
    funcs_dict={"Cg": dCg_dt, "Cl": dCl_dt, "CMS": dCMS_dt},
    name="gas-liquid exchange",
)
```

A parameter that appears in several equations (here `Kaw` and `k`) is one physical
quantity. When fitting, it is updated in **every** equation where it appears — see
[fitting.md](fitting.md#parameter-names).

## Events

An event is an instantaneous change of the state, for example a swallow that empties the
mouth and mixes the air compartments. An event function receives the state just before the
event and returns the **complete** state just after it:

```python
def swallow(y, params_current=None):
    new = dict(y)
    new["Cl"] = 0.0                      # the liquid is swallowed
    return new

model = Model(params_dict, funcs_dict, event_funcs={"swallow": swallow})
```

- `y` is the state dictionary just before the event.
- `params_current` receives the `event_params` dictionary given to `run_model` /
  `run_stages` (the same dictionary for every event). The built-in models read volumes from
  it, for example `{"VOA": ..., "VFA": ..., "VNA": ..., "VOL_m": ...}`.
- The returned dictionary must contain every state variable.

When and which events happen is described by a table of events, see
[simulation.md](simulation.md#instantaneous-events).

## Built-in models (`existing_models`)

Each entry is a **factory**: a function that takes parameter values and returns a `Model`.

| Factory | State variables | Events |
|---|---|---|
| `generate_convection_model` | `Cg, Cl, CMS` | `open` |
| `generate_convection_model_without_measure` | `Cg, Cl` | `open` |
| `in_vivo_diffusion_solution_model` | `VOL, COA, COL, CFA, CFL, CNA, CMS` | `swallow, jaw_move` |
| `in_vivo_diffusion_solid_model` | `VOP, VOL, COL, COA, CFA, CFL, CNA, CMS` | `swallow, chew, total_swallow` |
| `in_vivo_diffusion_solid_model_with_temp` | … + `TP` | `swallow, chew, total_swallow, open_velum` |
| `in_vivo_diffusion_solid_model_with_temp_and_surface` | … + `TP, SP` | `swallow, chew, total_swallow, open_velum` |
| `in_vivo_diffusion_solid_model_with_temp_and_AOLP` | … + `T` | `swallow, chew, total_swallow, open_velum` |

Variable naming: `O` = oral cavity, `F` = pharynx, `N` = nasal cavity; `A` = air,
`L` = liquid (saliva), `P` = product; `C` = concentration, `V` = volume; `CMS` = signal seen
by the instrument (mass spectrometer), `T`/`TP` = temperature, `SP` = product surface.

Parameter naming: the solution model and the solid models `*_with_temp*` use `AOAL, AFAL,
KAL, kL`; the solid model without temperature keeps the historical names `AOAP, AFAP, KOAL,
kOL`. The test file `tests/test_model_catalog.py` pins these conventions.

The in vivo models take the **breathing flow** as a function of time (`QNA_func`), usually
built by interpolating a measured breathing signal.

Example:

```python
import numpy as np
from digimouth.existing_models import in_vivo_diffusion_solution_model

def breathing(t):                                   # m3/s, positive = expiration
    return 2e-4 * np.sin(2 * np.pi * t / 4.0)

model = in_vivo_diffusion_solution_model(
    QNA_func=breathing, QSaliva=5e-2, AOAL=1e-4, VOA=4e-5, AFAL=6e-5, VFA=3e-5,
    VFL=1e-6, VNA=1.1e-5, tMS=1.0, KAL=2e-3, kL=0.14,
)
print(model)
```

## Adding a new model

1. Write one derivative function per state variable, following the signature above.
2. Write the event functions, if any.
3. Wrap them in a factory function `my_model(param1, param2, ...) -> Model` in
   `existing_models.py` (or in your own module: nothing requires a model to live in the
   package).
4. Add a test in `tests/` that builds the model, integrates it and checks a
   property you know (finite values, mass conservation, a known analytic case…).
