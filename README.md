# digimouth

**Simulate and fit dynamic (ODE) models of aroma release in the mouth.**

`digimouth` is a Python library to describe a model as a set of ordinary differential
equations, simulate it with instantaneous events (swallowing, chewing…) and multi-stage
protocols, and fit its parameters to experimental curves such as in vivo PTR-MS
measurements. It ships with the aroma release models of the Digimouth project (in vitro
convection, in vivo solution and solid/gel models).

It depends only on standard scientific libraries: NumPy, SciPy, pandas, SymPy and
Matplotlib.

## Features

- **Generic models**: a model is a dictionary of named state variables, one derivative
  function per variable and its parameters (`Model`).
- **Instantaneous events**: apply a jump to the state at given times (e.g. a swallow empties
  the mouth), then resume integration (`run_model`).
- **Multi-stage protocols**: chain time segments, each with its own model (`run_stages`).
- **Parameter fitting**: non-linear least squares with bounds, standard errors and confidence
  intervals, on one or several curves, and on full protocols with events
  (`optim_param_dynamic`, `optim_param_dynamic_multi`, `optim_param_protocol`).
- **Uncertainty and starting points**: residual bootstrap and grid search
  (`bootstrap_params`, `grid_search_initial_guess`).
- **Model catalogue**: in vitro convection and in vivo solution / solid models
  (`existing_models`).
- **Signal helpers**: alignment of swallows on breathing, peak detection, R², plotting
  (`utils`, `graphical_methods`).

## Installation

Python 3.9 or later is required.

```bash
git clone <repository-url>
cd digimouth
pip install -e .            # the library and its dependencies
pip install -e ".[dev]"     # + pytest, to run the tests
```

## Quick start

A first-order decay `dC/dt = -k C`, simulated then fitted:

```python
import numpy as np
from digimouth import Model, run_model, optim_param_dynamic

model = Model(
    params_dict={"C": {"k": 0.5}},                       # parameters of each equation
    funcs_dict={"C": lambda y, t, k: -k * y["C"]},       # dC/dt
    name="decay",
)

df = run_model(model, y0_dict={"C": 1.0}, t_start=0, t_end=10, n_points=101)
print(df.head())                                         # columns: time, C

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

## Documentation

| Page | Content |
|---|---|
| [doc/models.md](doc/models.md) | Writing a model (`Model`, equations, events) and the built-in model catalogue |
| [doc/simulation.md](doc/simulation.md) | Simulating: one segment, instantaneous events, multi-stage protocols, solvers |
| [doc/fitting.md](doc/fitting.md) | Fitting parameters: residuals, bounds, confidence intervals, several experiments, protocols |
| [doc/utilities.md](doc/utilities.md) | Signal pre-processing and plotting helpers |
| [doc/design_principles.md](doc/design_principles.md) | Rules that keep the package portable and reproducible (read before contributing) |
| [doc/known_issues.md](doc/known_issues.md) | Current limitations and known bugs |
| [examples/README.md](examples/README.md) | Example scripts, including the simulations of the published dataset |

## Examples and data

The [`examples/`](examples/README.md) folder contains two kinds of scripts:

- `examples_*.py` show how to use the functions on simple or synthetic cases (no data
  needed);
- `digimouth_*.py` reproduce the results of the Digimouth study, on the published dataset.

The experimental data are **not included** in this repository. They are published
separately:

> Peltier, Caroline (2026). *Dataset of aroma release in vivo (Digimouth)*, version 1.0.
> Recherche Data Gouv. <https://doi.org/10.57745/OVC3RL> — licence Etalab 2.0.

The example scripts download the files they need on first use, check them against the
published checksums and keep them in a local cache (see
[examples/README.md](examples/README.md#data)). The examples also read Excel files, which
needs `pip install openpyxl`.

## Package layout

```text
digimouth/
├── digimouth/
│   ├── dynamic_model.py      # numerical engine: Model, ODE integration, events, stages
│   ├── existing_models.py    # scientific models (in vitro convection, in vivo solution / solid)
│   ├── optimization.py       # parameter fitting, confidence intervals, bootstrap, grid search
│   ├── utils.py              # signal processing helpers
│   ├── graphical_methods.py  # plotting helpers
│   └── tests/                # automated tests (pytest)
├── doc/                      # documentation
├── examples/                 # example scripts and dataset downloader
├── pyproject.toml
└── LICENSE
```

The most used functions are re-exported at the top level
(`from digimouth import Model, run_model, run_stages, optim_param_dynamic, ...`); see
`digimouth/__init__.py` for the full list.

## Running the tests

From the repository folder:

```bash
pytest -q
```

## Citation

If you use `digimouth` in your research, please cite the software and, if you use the data,
the dataset:

```bibtex
@software{digimouth,
  author  = {Peltier, Caroline},
  title   = {digimouth: simulation and parameter fitting of dynamic models of aroma release in the mouth},
  version = {2.0.0},
  year    = {2026}
}

@dataset{digimouth_data,
  author    = {Peltier, Caroline},
  title     = {Dataset of aroma release in vivo (Digimouth)},
  version   = {1.0},
  publisher = {Recherche Data Gouv},
  year      = {2026},
  doi       = {10.57745/OVC3RL}
}
```

## License

Copyright (C) 2023–2026 INRAE. Author: Caroline Peltier.

This program is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version. It is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; see the [LICENSE](LICENSE) file for details.

The published dataset is distributed separately under the Etalab 2.0 licence.
