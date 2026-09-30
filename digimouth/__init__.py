"""
Digimouth Core: Simulation and Optimization Package
====================================================

A package for dynamic ODE-based simulations with parameter optimization,
designed for multi-phase and multi-experiment scenarios.

Main Components:
  - Model: generic container for simulation models
  - run_model / run_stages: run ODE simulations, with instantaneous events
  - optim_param_dynamic*: fit parameters to experimental data

Quick Start:
    from digimouth import Model, run_model, optim_param_dynamic

    model = Model(params_dict, funcs_dict)
    result = run_model(model, y0_dict, t_start=0, t_end=10)
    fit = optim_param_dynamic(time_exp, intensity_exp, y0_dict, model,
                              param_keys_to_optimize=["k"], initial_params_to_optimize=[0.1])

See the doc/ folder for the full documentation.
"""

__version__ = "2.0.0"

# ============================================================
# Core Classes
# ============================================================
from .dynamic_model import Model

# ============================================================
# Simulation API (main user-facing)
# ============================================================
from .dynamic_model import (
    # Low-level functions (advanced users)
    resolution_odeint_generic,
    run_model,
    run_stages,
    apply_event_func,
)

# ============================================================
# Optimization API (main user-facing)
# ============================================================
from .optimization import (
    optim_param_dynamic,
    optim_param_dynamic_multi,
    # Low-level functions
    calculate_residuals,
    calculate_residuals_multi,
    calculate_residuals_multiphases,
    bootstrap_params,
    grid_search_initial_guess,
    optim_param_protocol,
    calculate_residuals_protocol,
)

# ============================================================
# Model Generators
# ============================================================
from .existing_models import (
    generate_convection_model,
    in_vivo_diffusion_solution_model,
    in_vivo_diffusion_solid_model,
)

# ============================================================
# Visualization (utilities)
# ============================================================
from .graphical_methods import (
    plot_xy,
    plot_grid,
)

# ============================================================
# Public API (what users should import)
# ============================================================
__all__ = [
    # Classes
    "Model",
    # Simulation
    "resolution_odeint_generic",
    "run_model",
    "run_stages",
    "apply_event_func",
    # Optimization
    "optim_param_dynamic",
    "optim_param_dynamic_multi",
    "calculate_residuals",
    "calculate_residuals_multi",
    "calculate_residuals_multiphases",
    "bootstrap_params",
    "grid_search_initial_guess",
    "optim_param_protocol",
    "calculate_residuals_protocol",
    # Model Generators
    "generate_convection_model",
    "in_vivo_diffusion_solution_model",
    "in_vivo_diffusion_solid_model",
    # Visualization
    "plot_xy",
    "plot_grid",
]
