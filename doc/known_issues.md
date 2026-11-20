# Known issues

Last reviewed: 2026-11-20.

## Limitations

- **Confidence intervals assume independent residuals.** The default intervals
  (`ci_method="t"`, and those of `optim_param_protocol`) come from the Jacobian at the optimum,
  and `bootstrap_params` resamples residuals one by one. Both ignore the autocorrelation of
  residuals along time, which is usual with kinetic curves (instrument smoothing, model
  mismatch), so the intervals are generally too narrow. Treat them as a lower bound on the
  uncertainty; fitting replicates separately gives a more realistic spread
  (see [fitting.md](fitting.md#confidence-intervals)).
- **Symbolic and numerical equations are written differently.**
  `resolution_dsolve_generic` passes SymPy functions in `y`, so equations must use
  `y["C"](t)`; the usual numerical form `y["C"]` raises `TypeError`
  (see [simulation.md](simulation.md#symbolic-solution)).
- **Parameters are matched by name** across equations (see
  [fitting.md](fitting.md#parameter-names)).
- **Computing and plotting are still mixed** in `utils.compare_curves` and in the diagnostic
  option of `calculate_residuals` (see [design_principles.md](design_principles.md#3-computing-and-plotting-are-separate)).
