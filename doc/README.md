# digimouth — documentation

`digimouth` is a Python library to **describe, simulate and fit dynamic (ODE) models** of
aroma release in the mouth, with instantaneous events such as swallowing or chewing.

It is an **independent, portable package**: it depends only on standard scientific
libraries (NumPy, SciPy, pandas, SymPy, Matplotlib).

## Contents

| Page | What you will find |
|---|---|
| [models.md](models.md) | How a model is written (`Model`, equations, events) and the built-in model catalogue |
| [simulation.md](simulation.md) | Running a model: one time segment, instantaneous events, multi-stage protocols, solvers |
| [fitting.md](fitting.md) | Fitting parameters to experimental curves: residuals, bounds, confidence intervals, several experiments |
| [utilities.md](utilities.md) | Signal pre-processing helpers (`utils`) and plotting helpers (`graphical_methods`) |
| [design_principles.md](design_principles.md) | Rules that keep the package portable and reproducible (read before contributing) |
| [known_issues.md](known_issues.md) | Current limitations and known bugs |
| [../examples/README.md](../examples/README.md) | Example scripts, including simulations of the published dataset |

## Installation

From the folder that contains the package sources and its `pyproject.toml`:

```bash
pip install -e .            # the library and its dependencies
pip install -e ".[dev]"     # + pytest, to run the tests
```

## Quick start

A model is a set of named state variables, one derivative function per variable, and the
parameters of each function. Here is a first-order decay `dC/dt = -k C`:

```python
import numpy as np
from digimouth import Model, run_model

model = Model(
    params_dict={"C": {"k": 0.5}},                       # parameters of each equation
    funcs_dict={"C": lambda y, t, k: -k * y["C"]},       # dC/dt
    name="decay",
)

df = run_model(model, y0_dict={"C": 1.0}, t_start=0, t_end=10, n_points=101)
print(df.head())         # pandas DataFrame with columns: time, C
```

Fitting `k` to data:

```python
from digimouth import optim_param_dynamic

t_exp = np.linspace(0, 10, 60)
y_exp = np.exp(-0.5 * t_exp)                             # synthetic "measurements"

fit = optim_param_dynamic(
    t_exp, y_exp, {"C": 1.0}, model,
    param_keys_to_optimize=["k"],
    initial_params_to_optimize=[0.2],
    bounds=([0.01], [5.0]),
    output_key="C",
    normalization=False,
)
print(fit["opt_param"], fit["conf_intervals"])           # ~[0.5]
```

## Data

The experimental data used in the examples (in vivo aroma release measured by PTR-MS,
sensory evaluations and subject metadata, for three products: solution, gel and gusto) are
published separately, and are not included in this package:

> Peltier, Caroline (2026). *Dataset of aroma release in vivo (Digimouth)*, version 1.0.
> Recherche Data Gouv. [https://doi.org/10.57745/OVC3RL](https://doi.org/10.57745/OVC3RL) —
> licence Etalab 2.0.

| Files | Content |
|---|---|
| `conc_aci_sol`, `conc_aci_gel`, `conc_aci_gus` | Aroma concentration over time (PTR-MS), ~100 MB each |
| `senso_sol`, `senso_gel`, `senso_gus` | Sensory evaluations (swallow and chew times) |
| `int_sol`, `int_gel`, `int_gus` | Aroma intensity reported by the subjects (not needed for the models) |
| `metadata_sol`, `metadata_gel`, `metadata_gus` | Subject and session metadata (`rep = 0` marks warm-up trials, left out of the mean curves) |

The examples download the files they need automatically, with `examples/dataset.py`: each
file is fetched the first time, checked against the checksum published with the dataset,
and kept in a local cache (`%LOCALAPPDATA%\digimouth\data` on Windows,
`~/.cache/digimouth/data` elsewhere, or the folder given by the `DIGIMOUTH_DATA_DIR`
environment variable). To download everything in advance: `python examples/dataset.py`.

If you download the files by hand, note that Recherche Data Gouv converts some tabular files
to `.tab`; choose the **original format** to get the `.csv` / `.xlsx` files.

## Package layout

| Module | Role |
|---|---|
| `dynamic_model` | The numerical engine: `Model`, ODE integration, events, multi-stage protocols |
| `existing_models` | The scientific models (in vitro convection, in vivo solution / solid models) |
| `optimization` | Parameter fitting (least squares), confidence intervals, bootstrap, grid search |
| `utils` | Signal processing helpers (breath / swallow alignment, peak detection, R²) |
| `graphical_methods` | Plotting helpers (Matplotlib) |
| `tests/` | Automated tests (`pytest`, from the project folder) |

The most used functions are re-exported at the top level:
`from digimouth import Model, run_model, run_stages, optim_param_dynamic, ...`
(see `digimouth/__init__.py` for the full list).

## Running the tests

```bash
pytest -q                # from the project folder (runs digimouth/tests)
```
