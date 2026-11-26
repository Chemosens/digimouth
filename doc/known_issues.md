# Known issues

Last reviewed: 2026-11-20.

## Limitations

- **Default confidence intervals assume independent residuals.** The Jacobian intervals
  (`ci_method="t"`, and those of `optim_param_protocol` and `optim_param_dynamic_multi`) and
  `ci_method="bootstrap"` ignore the autocorrelation of residuals along time, which is usual
  with kinetic curves, so they are generally too narrow. For one curve, use
  `bootstrap_params(..., resampling="ar1")` or `resampling="block"`
  (see [fitting.md](fitting.md#confidence-intervals)). These are not yet available for
  several curves fitted together or for protocols.
- **Bootstrap intervals assume the model is right.** Part of the residual correlation can come
  from model mismatch, which no resampling of the residuals accounts for.
- **Symbolic and numerical equations are written differently.**
  `resolution_dsolve_generic` passes SymPy functions in `y`, so equations must use
  `y["C"](t)`; the usual numerical form `y["C"]` raises `TypeError`
  (see [simulation.md](simulation.md#symbolic-solution)).
- **Parameters are matched by name** across equations (see
  [fitting.md](fitting.md#parameter-names)).
- **Computing and plotting are still mixed** in `utils.compare_curves` and in the diagnostic
  option of `calculate_residuals` (see [design_principles.md](design_principles.md#3-computing-and-plotting-are-separate)).
