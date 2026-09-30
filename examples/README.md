# digimouth — examples

Two kinds of scripts:

- **`digimouth_*`** reproduce the results of the Digimouth paper. Those that use measurements
  run on the published Digimouth dataset, which they download automatically.
- **`examples_*`** show how to use the functions of the package and the effect of their
  parameters, on simple or synthetic cases (no data needed).

## Before you start

1. Install `digimouth` (from the project folder, the one that contains `pyproject.toml`):

   ```bash
   pip install -e .
   ```

2. The examples also read Excel files, which needs `openpyxl`:

   ```bash
   pip install openpyxl
   ```

3. An internet connection is needed the first time, to download the in vivo data (see
   below). The small in vitro data file is included in `data/`.

## Reproducing the paper results (`digimouth_*`)

| Script | What it does |
|---|---|
| `digimouth_protocol_simulation_clean.py` | **In vitro.** Aroma release in a closed vessel: effect of the model parameters, and protocols with swallow events and several stages. |
| `digimouth_optim_in_vitro_y0_dependency.py` | **In vitro, fit on measurements** (data included in `data/`). Fits the convection model (`k`, `tMS`, and optionally `Kaw`) to three repetitions of isoamyl acetate release, one by one then all together. Shows how to make an initial condition depend on a fitted parameter (`y0_dependencies`: the initial gas concentration `Cg = Kaw × Cl`). |
| `digimouth_protocol_simulation_clean_in_vivo.py` | **In vivo, generic cases.** Solution and gel (solid) models with an idealised breathing signal: effect of swallows, chewing, temperature and product surface. |
| `digimouth_protocol_simulation_from_all_evals_sol.py` | **In vivo, aroma solution, whole panel** (published data). For every subject and repetition, builds the solution model driven by the subject's own breathing and swallow times, simulates it, averages the simulations and compares the mean curve with the mean measured curve, for the two protocols (`fast` / `long`). Also shows, for one subject, the effect of multiplying a parameter (saliva flow, breathing flow, pharynx film volume, transfer coefficient) by a factor `dix`. |
| `digimouth_protocol_simulation_from_all_evals_gel.py` | **In vivo, gel, whole panel** (published data). Same approach with the gel model and the two protocols `chew` / `succ` (chewing / sucking). |

The two `from_all_evals` scripts take a few minutes each (all subjects are simulated).

## Exploring the functions (`examples_*`)

| Script | What it does |
|---|---|
| `examples_dynamic_model.py` | Writing a model by hand (equations, parameters, initial conditions), simulating it, the effect of each parameter, adding swallow events, chaining stages, and an in vivo solution model with and without swallows. |
| `examples_existing_models.py` | The models of the catalogue (`existing_models`): in vitro convection model, in vivo solution model, and the solid models with chewing, temperature and product surface. |
| `examples_optimization.py` | Fitting parameters on synthetic data: residuals, grid search for a starting point, least-squares fit, several curves at once, initial conditions that depend on the parameters. |

## Data

The measurements come from the published dataset:

> Peltier, Caroline (2026). *Dataset of aroma release in vivo (Digimouth)*, version 1.0.
> Recherche Data Gouv. https://doi.org/10.57745/OVC3RL — licence Etalab 2.0.

`dataset.py` handles them; you do not need to download anything by hand.

- `fetch("conc_aci_sol.csv")` returns the local path of a file of the dataset. The first time,
  the file is downloaded from Recherche Data Gouv and checked against the checksum published
  with the dataset; afterwards it is read from a local cache:
  `%LOCALAPPDATA%\digimouth\data` on Windows, `~/.cache/digimouth/data` elsewhere, or the folder
  given by the `DIGIMOUTH_DATA_DIR` environment variable.
- `experimental_mean_curve("sol")` (or `"gel"`) returns the mean measured curve across subjects,
  one row per second and protocol, computed from the published data (warm-up trials,
  `rep = 0`, are left out).
- `python dataset.py` downloads every file in advance (about 300 MB), for instance before
  working offline.

The example scripts import these functions with `from dataset import fetch, experimental_mean_curve`,
so `dataset.py` must stay in the same folder as the examples.

## Running an example

As a script, from this folder:

```bash
python digimouth_protocol_simulation_from_all_evals_sol.py
```

Or line by line in an interactive Python session (e.g. Shift+Enter in VS Code): always start
by running the top of the file (imports, then the `EXAMPLES_DIR` block and
`from dataset import ...`), then go on block by block, in order. The final comparison figure
reads the mean curves written by the simulation part, so run that part first.

## Results

Results are written to `output/` in this folder (ignored by git):

- by default, only the mean simulated curves (`results_std_*.csv`, a few tens of KB);
- set `SAVE_SUBJECT_RESULTS = True` at the top of a script to also save the simulation of
  every subject (several hundred MB).

`output/` can be emptied at any time; the scripts write it again.
