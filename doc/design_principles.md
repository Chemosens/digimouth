# Design principles

`digimouth` should stay an **independent, portable and reproducible** library: anyone should
be able to install it on another machine and re-run the simulations and fits of a paper.
These rules keep it that way. Please follow them when changing the package.

## 1. Only package imports

Inside the package, modules import each other with relative imports:

```python
from . import dynamic_model as dm
```

Outside the package, code imports it as a package:

```python
from digimouth import dynamic_model as dm
```

Do not add the `digimouth/` folder to `sys.path` and import `dynamic_model` directly, and do
not write absolute paths such as `C:/Users/...`: they only work on one machine.

## 2. Declared, light dependencies

The core depends on NumPy, SciPy, pandas, SymPy and Matplotlib, as declared in
`pyproject.toml`. A new dependency must be declared there. Heavy or specialised libraries
(scikit-learn, trimesh, gmsh…) do not belong in the package; put the code that needs them in
a separate module or package that uses `digimouth`.

## 3. Computing and plotting are separate

A function that **computes** something returns numbers (arrays, DataFrames, dictionaries)
and never opens a window. A separate function **draws** those numbers when asked.

Why:

- a fit or a simulation can run many times in a loop, or on a computing server without a
  screen, without being blocked by figures;
- the computation can be tested on its own;
- the same results can be plotted in different ways.

Preferred style:

```python
r2, rmse, residuals = compare(t1, x1, t2, x2)      # computes, returns numbers
plot_compare(t1, x1, t2, x2)                       # draws, only when you ask
```

Some existing functions still mix the two (`utils.compare_curves` always opens a plot;
`optimization.calculate_residuals` can draw a diagnostic figure with `save_path`). New code
should follow the preferred style, and these functions can be split progressively while
keeping the old behaviour available for existing scripts.

## 4. No output on the console by default

Library functions should not `print` progress or debug values. Report problems with
exceptions (or the `warnings` module); leave printing to the code that calls the library.

## 5. Reproducibility

- `odeint` stays the default solver so that published results are reproduced exactly;
  other solvers are opt-in (`solver="solve_ivp"`).
- Anything random takes a seed (`boot_random_state`, `random_state`).
- Every behaviour you rely on for a result should be covered by a test in
  `tests/`, ideally against a case whose answer is known (analytic solution,
  synthetic data with known parameters, conservation of mass).

## 6. Tests and scripts are kept apart

`tests/` contains only automated tests (functions `test_*` with assertions), run by
`pytest`. Exploratory scripts (plots, analyses of specific data files) are not tests and do
not belong in the package.
