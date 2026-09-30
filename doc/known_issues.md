# Known issues

Last reviewed: 2026-09-29.

## Bugs

### Multi-stage fitting does not update the parameters

`optimization.optim_param_dynamic_multiphases` and `calculate_residuals_multiphases` look for
the parameters in `stage["params"]`, while `dynamic_model.run_stages` reads the model from
`stage["model"]` (see [simulation.md](simulation.md#multi-stage-protocols)).

- With stages in the current format (`{"t_start", "t_end", "model"}`),
  `optim_param_dynamic_multiphases` raises `ValueError: param_key '...' not found in any stage params`.
- `calculate_residuals_multiphases` runs, but the parameter values it receives are never
  applied, so the residuals do not depend on them.

These two functions are deprecated (they emit a `DeprecationWarning`) and kept only for
existing scripts. Use `optim_param_protocol` / `calculate_residuals_protocol` instead, which work
with the current stage format and also take events into account
(see [fitting.md](fitting.md#fitting-a-protocol-events-and-stages)).

## Limitations

- **Symbolic and numerical equations are written differently.**
  `resolution_dsolve_generic` passes SymPy functions in `y`, so equations must use
  `y["C"](t)`; the usual numerical form `y["C"]` raises `TypeError`
  (see [simulation.md](simulation.md#symbolic-solution)).
- **Normalisation conventions differ.** `calculate_residuals` divides the simulation and the
  data by their own maximum, whereas `calculate_residuals_multiphases` rescales the
  simulation to the maximum of the data.
- **Parameters are matched by name** across equations (see
  [fitting.md](fitting.md#parameter-names)).
- **Computing and plotting are still mixed** in `utils.compare_curves` and in the diagnostic
  option of `calculate_residuals` (see [design_principles.md](design_principles.md#3-computing-and-plotting-are-separate)).
