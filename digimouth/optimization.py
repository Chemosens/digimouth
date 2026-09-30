import numpy as np
import pandas as pd
import copy
import warnings
from scipy.optimize import least_squares
from scipy.stats import t
import itertools

from . import existing_models as exmod
from . import graphical_methods as gmet
from . import dynamic_model as dm


def calculate_residuals(params_vector, time_exp, intensity_exp, y0_dict, model, 
                        param_keys, param_sharing=None, output_key="CMS", normalization=True,
                        y0_dependencies=None, save_path=None, ode_solver="odeint",
                        ode_method="LSODA", solve_ivp_options=None):
    """
    Compute residuals between model simulation and experimental data.
    This function is generic: you can choose which parameters to optimize by passing
    `param_keys`, and it will update `params_dict` accordingly.
    Args:
        params_vector (array-like): Current values of parameters to optimize.
        time_exp (array-like): Experimental time points.
        intensity_exp (array-like): Experimental intensity data.
        y0_dict (dict): Initial conditions for the simulation.
        model (Model): Model object with funcs_dict and params_dict.
        param_keys (list): Names of the parameters being optimized.
        param_sharing (dict, optional): Mapping of param_name -> list of variables sharing it.
            Ex: {"Kaw": ["Cg", "Cl"]} means optimize Kaw once, update in both Cg and Cl.
        output_key (str): Which variable to extract from simulation.
        normalization (bool): Whether to normalize the simulation.

        y0_dependencies (dict, optional): Maps y0 variable names to callable functions that
            compute them from current_params and y0_local. Allows parameterizing initial conditions.
            Ex: {"Cg": lambda params, y0: params["Cg"]["Kaw"] * y0["Cl"]}
        save_path (str, optional): diagnostic plot of the simulated vs experimental curve,
            titled with the current parameter values. "show" displays it, any other value is
            the file where it is saved. None (default): no plot.
    Returns:
        residuals (np.array): Difference between normalized simulation and experimental data.
    """
  #  print("residuals")
    # Quick validation (should be caught by optim_param_dynamic)
    if output_key not in y0_dict:
        raise KeyError(f"output_key '{output_key}' not in y0_dict")
    # Deep copy to avoid mutating the original
    current_params = copy.deepcopy(model.params_dict)
    # copy initial conditions locally so we don't mutate the caller's y0_dict
    y0_local = copy.deepcopy(y0_dict)
    current_param_values = {
    k: float(v)
    for k, v in zip(param_keys, params_vector)
    }       
    # Update parameters using param_sharing
    if param_sharing is None:
        param_sharing = {}
    for param_name, param_value in current_param_values.items():
        if param_name in param_sharing:
            # Mettre à jour dans TOUTES les variables listées
            for var_name in param_sharing[param_name]:
                if var_name in current_params and param_name in current_params[var_name]:
                    current_params[var_name][param_name] = param_value
        else:
            # Fallback: chercher et mettre à jour dans chaque variable
            for var in current_params:
                if param_name in current_params[var]:
                    current_params[var][param_name] = param_value
    # Apply y0 dependencies (parameterized initial conditions)
    if y0_dependencies is not None:
        for var_name, func in y0_dependencies.items():
            if callable(func):
                y0_local[var_name] = func(current_params, y0_local)
    # Legacy support: initial_conditions_depending_on_params (Cg = Kaw * Cl)
    model_current = exmod.Model(
        params_dict=current_params, 
        funcs_dict=model.funcs_dict, 
        event_funcs=model.event_funcs if hasattr(model, 'event_funcs') else {},
        name=model.name if hasattr(model, 'name') else None
    )
    
    df_result = dm.resolution_odeint_generic(
        model_current,
        y0_local,
        time_exp,
        solver=ode_solver,
        method=ode_method,
        solve_ivp_options=solve_ivp_options,
    )
    # Normalize simulation to experimental max
    CMS_sim = df_result[output_key].to_numpy()
    if normalization:
        if np.max(intensity_exp) > 0:
            CMS_norm = CMS_sim / np.max(CMS_sim)
            residuals = CMS_norm - intensity_exp/np.max(intensity_exp) 
        else:
             CMS_norm = np.zeros_like(CMS_sim)
             residuals = CMS_norm - intensity_exp/ np.max(intensity_exp)
    else:
        CMS_norm = CMS_sim
        residuals = CMS_norm - intensity_exp
    if save_path is not None:
        title = ", ".join(f"{name}: {value:.2e}" for name, value in current_param_values.items())
        data = intensity_exp / np.max(intensity_exp) if normalization else intensity_exp
        if save_path == 'show':
            gmet.plot_xy(time_exp, CMS_norm, opt_y=data, save_path=None, show=True, title=title)
        else:
            gmet.plot_xy(time_exp, CMS_norm, opt_y=data, save_path=f"{save_path}", show=False, title=title)
    return residuals

def calculate_residuals_multi(params_vector, time_exp, intensity_exp, y0_dict, model, 
                        param_keys, param_sharing=None, output_key="CMS", normalization=True,
                        y0_dependencies=None, save_path=None, ode_solver="odeint",
                        ode_method="LSODA", solve_ivp_options=None):
    """
    Compute residuals between model simulation and experimental data.
    Accepts either:
      - time_exp: array-like, intensity_exp: array-like  (single experiment)
      - time_exp: list/tuple of array-like, intensity_exp: list/tuple of array-like (multiple experiments)
    Returns concatenated 1D residuals vector.
    
    Args:
        y0_dependencies (dict, optional): Maps y0 variable names to callable functions.
    """
    # validate param keys
    if param_keys is None or len(param_keys) == 0:
        raise ValueError("param_keys must be provided and non-empty")
    multi = isinstance(time_exp, (list, tuple)) or isinstance(intensity_exp, (list, tuple))
    if multi:
        # require both lists and same length
        if not (isinstance(time_exp, (list, tuple)) and isinstance(intensity_exp, (list, tuple))):
            raise ValueError("Both time_exp and intensity_exp must be lists/tuples for multi-experiment mode")
        if len(time_exp) != len(intensity_exp):
            raise ValueError("time_exp and intensity_exp lists must have same length")
        residuals_list = []
        for i, (t_exp_i, inten_exp_i) in enumerate(zip(time_exp, intensity_exp)):
            #t_exp_i = np.asarray(t_exp_i, dtype=float)
            #inten_exp_i = np.asarray(inten_exp_i, dtype=float)
            # une figure par expérience : "show" les affiche, sinon <save_path>_<i>
            save_path_i = save_path if save_path in (None, "show") else f"{save_path}_{i}"
            residuals_i=calculate_residuals(params_vector=params_vector, time_exp=t_exp_i, intensity_exp=inten_exp_i, y0_dict=y0_dict, model=model,
                            param_keys=param_keys, param_sharing=param_sharing, output_key=output_key, normalization=normalization,y0_dependencies=y0_dependencies,
                            save_path=save_path_i, ode_solver=ode_solver, ode_method=ode_method, solve_ivp_options=solve_ivp_options)
            residuals_list.append(residuals_i)

        residuals=np.concatenate([r for r in residuals_list])      
        return residuals
    else:
        # single experiment case
        residuals=calculate_residuals(params_vector=params_vector, time_exp=time_exp, intensity_exp=intensity_exp, y0_dict=y0_dict, model=model, 
                                param_keys=param_keys, param_sharing=param_sharing, output_key=output_key, normalization=normalization,y0_dependencies=y0_dependencies,
                                save_path=save_path, ode_solver=ode_solver, ode_method=ode_method, solve_ivp_options=solve_ivp_options)
        return residuals

def calculate_residuals_multiphases(
        params_vector,
        time_exp,
        intensity_exp,
        stages,
        y0_initial,
        param_keys,
        output_key="CMS",
        normalization=True,
        event_func_dict=None,
        n_points=200,
        event_params=None,
        save_path=None):
    # detect multi experiment
    multi = isinstance(time_exp, (list, tuple))
    if not multi:
        time_exp = [time_exp]
        intensity_exp = [intensity_exp]
        stages = [stages]
        y0_initial = [y0_initial]
    residuals_list = []
    for i, (t_exp, inten_exp, stages_i, y0_i) in enumerate(
            zip(time_exp, intensity_exp, stages, y0_initial)):
        t_exp = np.asarray(t_exp, dtype=float)
        inten_exp = np.asarray(inten_exp, dtype=float)
        stages_copy = copy.deepcopy(stages_i)
        # update parameters
        current_param_values = dict(zip(param_keys, params_vector))
        for st in stages_copy:
            for var in st.get("params", {}): 
                for p_name, p_val in current_param_values.items():
                    if p_name in st["params"][var]:
                        st["params"][var][p_name] = p_val
        # run simulation
        df_sim = dm.run_stages(
            stages_copy,
            y0_i,
            event_func_dict=event_func_dict,
            n_points=n_points,
            event_params=event_params
        )
        sim_time = df_sim["time"].to_numpy()
        sim_vals = df_sim[output_key].to_numpy()
        # interpolate on observation grid
        sim_interp = np.interp(t_exp, sim_time, sim_vals)
        # normalization
        if normalization:
            max_sim = np.max(sim_interp)
            max_exp = np.max(inten_exp)
            if max_sim > 0 and max_exp > 0:
                sim_interp = sim_interp / max_sim * max_exp
            else:
                sim_interp = np.zeros_like(sim_interp)
        # optional plot
        if save_path:
            title = f"exp{i}"
            gmet.plot_xy(
                t_exp,
                sim_interp,
                opt=inten_exp,
                save_path=f"{save_path}_{i}.png",
                title=title,
                show=False
            )
        residuals_list.append((sim_interp - inten_exp).ravel())
    return np.concatenate(residuals_list)


def bootstrap_params(time_exp, intensity_exp, y0_dict, model,
                     param_keys_to_optimize, initial_params_to_optimize=None,
                     bounds=None, method='trf', output_key="CMS",
                     normalization=True, initial_conditions_depending_on_params=False,
                     y0_dependencies=None, n_boot=1000, conf_level=0.95, random_state=None,
                     ode_solver="odeint", ode_method="LSODA", solve_ivp_options=None):
    """
    Estimate parameter uncertainty via residual bootstrap.

    Parameters
    ----------
    time_exp, intensity_exp, y0_dict, model :
        same arguments as for :func:`optim_param_dynamic` (model + data).
    param_keys_to_optimize : list of str
        names of parameters to fit.
    initial_params_to_optimize : array-like or None
        starting guess; if None the values from model.params_dict are used.
    bounds : tuple or None
        bounds passed to least_squares.
    method : str
        optimization method for least_squares.
    output_key : str
        variable key to compare with experiments (default "CMS").
    normalization : bool
        whether to normalize simulation to data max before computing residuals.
    initial_conditions_depending_on_params : bool
        if True, update y0 based on parameter values (legacy support).
    y0_dependencies : dict, optional
        Maps y0 variable names to callable functions for parameterized initial conditions.
    n_boot : int
        number of bootstrap samples.
    conf_level : float
        confidence level for returned intervals (e.g. 0.95).
    random_state : int or np.random.Generator or None
        seed or generator for reproducibility.

    Returns
    -------
    dict with keys:
        samples : np.ndarray, shape (n_successful, n_params)
            bootstrapped parameter vectors.
        stderr_boot : np.ndarray, shape (n_params,)
            standard deviation of the bootstrap samples.
        conf_intervals_boot : np.ndarray, shape (n_params,2)
            percentile-based confidence intervals.
        original_fit : dict
            dictionary returned by :func:`optim_param_dynamic` for the original data.
    """
    # run initial optimization to obtain residuals and starting point
    if initial_params_to_optimize is None:
        initial_params_to_optimize = []
    orig = optim_param_dynamic(time_exp, intensity_exp, y0_dict, model,
                               param_keys_to_optimize,
                               initial_params_to_optimize, bounds,
                               method=method, output_key=output_key,
                               normalization=normalization,
                               y0_dependencies=y0_dependencies,
                               conf_level=conf_level,
                               ode_solver=ode_solver, ode_method=ode_method,
                               solve_ivp_options=solve_ivp_options)
    if orig.get("opt_param") is None:
        raise RuntimeError("initial optimization failed, cannot bootstrap")
    opt_param = orig["opt_param"]
    # compute residual vector on original data
    resid = calculate_residuals(opt_param, time_exp, intensity_exp,
                                 y0_dict, model,
                                 param_keys_to_optimize,
                                 param_sharing=None,
                                 output_key=output_key,
                                 normalization=normalization,
                                 y0_dependencies=y0_dependencies,
                                 ode_solver=ode_solver, ode_method=ode_method,
                                 solve_ivp_options=solve_ivp_options)
    resid = np.asarray(resid)
    # prepare random generator
    if random_state is None:
        rng = np.random.default_rng()
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    samples = []
    lower, upper = None, None
    if bounds is not None:
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
    # resampling loop
    for i in range(n_boot):
        # draw residuals with replacement
        boot_resid = rng.choice(resid, size=resid.shape, replace=True)
        # new artificial data
        intensity_boot = intensity_exp + boot_resid
        try:
            result = least_squares(
                calculate_residuals,
                opt_param,  # start from original solution
                kwargs={
                    'time_exp': time_exp,
                    'intensity_exp': intensity_boot,
                    'y0_dict': y0_dict,
                    'model': model,
                    'param_keys': param_keys_to_optimize,
                    'param_sharing': None,
                    'output_key': output_key,
                    'normalization': normalization,
                    'y0_dependencies': y0_dependencies,
                    'ode_solver': ode_solver,
                    'ode_method': ode_method,
                    'solve_ivp_options': solve_ivp_options
                },
                bounds=[lower, upper] if bounds is not None else (-np.inf, np.inf),
                method=method
            )
            if result.success:
                samples.append(result.x.copy())
        except Exception:
            # ignore failure and continue
            continue
    samples = np.vstack(samples) if samples else np.zeros((0, len(opt_param)))
    stderr_boot = np.std(samples, axis=0, ddof=1) if samples.size else None
    if samples.size:
        alpha = (1.0 - conf_level) / 2.0
        lower_ci = np.percentile(samples, 100 * alpha, axis=0)
        upper_ci = np.percentile(samples, 100 * (1 - alpha), axis=0)
        #conf_intervals_boot = np.vstack((lower_ci, upper_ci)).T
        conf_intervals_boot = {
            key: (low, high)
            for key, low, high in zip(param_keys_to_optimize, lower_ci, upper_ci)
        }
    else:
        conf_intervals_boot = None
    return {
        'samples': samples,
        'stderr_boot': stderr_boot,
        'conf_intervals_boot': conf_intervals_boot,
        'original_fit': orig
    }

def grid_search_initial_guess(param_grids,  kwargs):
    """
    Brute-force grid search to find a good initial guess for least-squares problems.
    Parameters
    ----------
    param_grids : list of array-like
        Candidate values for each parameter.
    residual_func : callable
        Function returning residual vector (ex: calculate_residuals).
    kwargs : dict
        Arguments passed to residual_func.
    Returns
    -------
    best_params
    best_cost
    all_costs (DataFrame)
    """
    param_keys = kwargs.get("param_keys", [f"param_{i}" for i in range(len(param_grids))])
    best_params = None
    best_cost = np.inf
    records = []
    for params in itertools.product(*param_grids):
        params = np.array(params).copy()
        residuals = calculate_residuals(params, **kwargs)
        cost = np.mean(residuals**2)
        record = dict(zip(param_keys, params))
        record["cost"] = cost
        records.append(record)
        if cost < best_cost:
            best_cost = cost
            best_params = params
    all_costs = pd.DataFrame(records)  
    best_params_dict = copy.deepcopy(kwargs['model'].params_dict)
    current_param_values = dict(zip(param_keys, best_params))
    # Mettre à jour chaque paramètre trouvé dans toutes les variables
    for param_name, param_value in current_param_values.items():
        for var_name in best_params_dict:
            if param_name in best_params_dict[var_name]:
                best_params_dict[var_name][param_name] = param_value
    
    # Créer le model avec les meilleurs paramètres
    best_model = exmod.Model(
        funcs_dict=kwargs['model'].funcs_dict,
        params_dict=best_params_dict,
        event_funcs=kwargs['model'].event_funcs if hasattr(kwargs['model'], 'event_funcs') else {},
        name=f"{kwargs['model'].name}_optimized" if hasattr(kwargs['model'], 'name') else "optimized_model"
    )
    return best_model, best_params, best_cost, all_costs

def optim_param_dynamic(time_exp, intensity_exp, y0_dict, model, param_keys_to_optimize=None,
                         initial_params_to_optimize=[], bounds=None, 
                           method='trf', output_key="CMS",normalization=True,
                           y0_dependencies=None, conf_level=0.95, ci_method='t', n_boot=200,
                           boot_random_state=None, verbose=0, ode_solver="odeint",
                           ode_method="LSODA", solve_ivp_options=None):
    """
    Optimize parameters to fit experimental data.
    
    Args:
        time_exp, intensity_exp, y0_dict, model: Model and data
        param_keys_to_optimize (list): Parameters to optimize
        initial_params_to_optimize (array): Initial guess for parameters
        bounds (tuple): Lower and upper bounds for parameters
        method (str): Optimization method for least_squares
        ode_solver (str): ODE solver, ``"odeint"`` (default) or ``"solve_ivp"``.
        ode_method (str): Integration method used by ``solve_ivp``.
        solve_ivp_options (dict, optional): Additional ``solve_ivp`` options.
        output_key (str): Which variable to optimize against
        normalization (bool): Whether to normalize simulation
        initial_conditions_depending_on_params (bool): Legacy support; update y0 based on params (e.g. Kaw)
        y0_dependencies (dict, optional): Maps y0 variable names to callables for parameterized IC.
        conf_level (float, 0-1): desired confidence level for parameter intervals (default 0.95)
        ci_method (str): 't' for t‑based intervals or 'bootstrap' to run residual bootstrap.
        n_boot (int): number of bootstrap samples when ci_method='bootstrap'.
        boot_random_state: seed or Generator for bootstrap reproducibility.
    
    Returns:
        dict with keys:
            opt_param : optimized parameter vector
            ssr : sum of squared residuals
            rmse : root mean squared error
            opt_params_dict : params_dict updated with optimized values
            covariance : covariance matrix or None
            stderr : standard errors or None
            conf_intervals : array[n_params,2] lower/upper bounds or None
            ci_method : which method was used ('t' or 'bootstrap')
    """
    # Validation checks
    if param_keys_to_optimize is None or len(param_keys_to_optimize) == 0:
        raise ValueError("param_keys_to_optimize must be provided and non-empty")
    n = len(param_keys_to_optimize)
    if bounds is None:
        lower = np.full(n, -np.inf)
        upper = np.full(n, np.inf)
    else:
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise ValueError("bounds must be a tuple (lower_array, upper_array) or None")
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if lower.size != n:
            raise Exception("Lower bounds size does not match number of parameters to optimize")
        if upper.size != n:
            raise Exception("Upper bounds size does not match number of parameters to optimize")
    if len(y0_dict) != len(model.funcs_dict):
        raise ValueError(f"y0_dict has {len(y0_dict)} variables but funcs_dict has {len(model.funcs_dict)}")
    if output_key not in y0_dict:
        raise ValueError(f"output_key '{output_key}' not found in y0_dict keys: {list(y0_dict.keys())}") 
    if output_key not in model.funcs_dict:
        raise ValueError(f"output_key '{output_key}' not found in funcs_dict keys: {list(model.funcs_dict.keys())}")
    param_sharing = {}  # {"Kaw": ["Cg", "Cl"], "k": ["Cg", "Cl"], ...}
    for param_key in param_keys_to_optimize:
        found_in_vars = []
    # Chercher ce paramètre dans TOUTES les variables
        for var in model.params_dict:
            if param_key in model.params_dict[var]:
                found_in_vars.append(var)
        if not found_in_vars:
            raise ValueError(f"param_key '{param_key}' not found in any variable of fixed_params_dict")
    # Mapper ce paramètre à toutes les variables où il apparaît
        param_sharing[param_key] = found_in_vars
    try: 
        #print("optim param")
        #print(initial_params_to_optimize)
        result = least_squares(
            calculate_residuals,  
            initial_params_to_optimize,        
            kwargs={
                'time_exp': time_exp,
                'intensity_exp': intensity_exp,
                'y0_dict': y0_dict,
                'model':model,
                'param_keys': param_keys_to_optimize,
                'param_sharing': param_sharing, 
                'output_key': output_key,
                'normalization': normalization,
                'y0_dependencies': y0_dependencies,
                'ode_solver': ode_solver,
                'ode_method': ode_method,
                'solve_ivp_options': solve_ivp_options
            },
            bounds=[lower, upper],
            method=method  ,verbose=verbose  
        )
        if result.success:
            opt_param = result.x
            ssr = 2*result.cost  # result.cost = 0.5 * sum(res^2) donc SSR = 2 * cost
            n_pts = result.fun.size if result.fun is not None else 0
            rmse = np.sqrt(ssr / max(1, n_pts))
            # compute covariance and confidence intervals depending on method
            cov = None
            stderr = None
            conf_intervals = None
            if ci_method == 't':
                p = len(param_keys_to_optimize)
                dof = max(1, n_pts - p)
                if n_pts > p and result.jac is not None:
                    res_var = ssr / dof
                    try:
                        jt_j = result.jac.T.dot(result.jac)
                        cov = np.linalg.inv(jt_j) * res_var
                        stderr_array = np.sqrt(np.diag(cov))
                        alpha = (1.0 - conf_level) / 2.0
                        tval = t.ppf(1.0 - alpha, dof)
                        lower = opt_param - tval * stderr_array
                        upper = opt_param + tval * stderr_array
                        stderr = {
                            key: val
                            for key, val in zip(param_keys_to_optimize, stderr_array)
                        }
                        conf_intervals = {
                            key: (low, high)
                            for key, low, high in zip(param_keys_to_optimize, lower, upper)
                        }
                    except np.linalg.LinAlgError:
                        cov = None
                        stderr = None
                        conf_intervals = None
                # else leave None
            elif ci_method == 'bootstrap':
                # perform residual bootstrap to obtain stderr and conf_ints
                boot = bootstrap_params(time_exp, intensity_exp, y0_dict, model,
                                         param_keys_to_optimize,
                                         initial_params_to_optimize,
                                         bounds=bounds, method=method,
                                         output_key=output_key,
                                         normalization=normalization,
                                         y0_dependencies=y0_dependencies,
                                         n_boot=n_boot, conf_level=conf_level,
                                         random_state=boot_random_state,
                                         ode_solver=ode_solver, ode_method=ode_method,
                                         solve_ivp_options=solve_ivp_options)
                stderr = boot.get('stderr_boot')
                conf_intervals = boot.get('conf_intervals_boot')
                # covariance not available from bootstrap; keep cov=None
            else:
                raise ValueError(f"ci_method must be 't' or 'bootstrap', got '{ci_method}'")
            # Construire le dictionnaire des paramètres optimisés
            opt_params_dict = copy.deepcopy(model.params_dict)
            current_param_values = dict(zip(param_keys_to_optimize, opt_param))
            for param_name, param_value in current_param_values.items():
                if param_name in param_sharing:
                    # Mettre à jour dans TOUTES les variables listées
                    for var_name in param_sharing[param_name]:
                        if var_name in opt_params_dict and param_name in opt_params_dict[var_name]:
                            opt_params_dict[var_name][param_name] = param_value
                else:
                    # Fallback: chercher et mettre à jour dans chaque variable
                    for var in opt_params_dict:
                        if param_name in opt_params_dict[var]:
                            opt_params_dict[var][param_name] = param_value
            
            opt_model = exmod.Model(
                funcs_dict=model.funcs_dict,
                params_dict=opt_params_dict,
                event_funcs=model.event_funcs if hasattr(model, 'event_funcs') else {},
                name=f"{model.name}_optimized" if hasattr(model, 'name') else "optimized_model"
            )
            return {"opt_param":opt_param, "ssr":ssr, "rmse":rmse,
                    "opt_params_dict":opt_params_dict,
                    "opt_model":opt_model,
                    "covariance":cov, "stderr":stderr,
                    "conf_intervals":conf_intervals}
        else:
            warnings.warn(f"Optimisation échouée pour: {result.message}")
            return {"opt_param":None, "ssr":None, "rmse":None}
    except Exception as e:
        warnings.warn(f"Erreur lors de l'optimisation: {e}")
        return {"opt_param":None, "ssr":None, "rmse":None,}

def optim_param_dynamic_multi(time_exp, intensity_exp, y0_dict, model, param_keys_to_optimize=None,
                         initial_params_to_optimize=None, bounds=None,
                           method='trf', output_key="CMS",normalization=True,
                           y0_dependencies=None, save_path=None, conf_level=0.95, ci_method='t',
                           n_boot=200, boot_random_state=None, verbose=0,
                           ode_solver="odeint", ode_method="LSODA", solve_ivp_options=None):
    """
    Optimize parameters to fit single or multiple experimental series.
    time_exp and intensity_exp can be arrays (single) or lists of arrays (multiple).
    """
    if param_keys_to_optimize is None or len(param_keys_to_optimize) == 0:
        raise ValueError("param_keys_to_optimize must be provided and non-empty")
    # detect sharing
    param_sharing = {}
    for param_key in param_keys_to_optimize:
        found_in_vars = []
        for var in model.params_dict:
            if param_key in model.params_dict[var]:
                found_in_vars.append(var)
        if not found_in_vars:
            raise ValueError(f"param_key '{param_key}' not found in any variable of params_dict")
        param_sharing[param_key] = found_in_vars
    n = len(param_keys_to_optimize)
    # initial guess
    if initial_params_to_optimize is None:
        initial_params_to_optimize = []
        for pk in param_keys_to_optimize:
            var = param_sharing[pk][0]
            initial_params_to_optimize.append(model.params_dict[var][pk])
    initial_params_to_optimize = np.asarray(initial_params_to_optimize, dtype=float)
    # bounds
    if bounds is None:
        lower = np.full(n, -np.inf)
        upper = np.full(n, np.inf)
    else:
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise ValueError("bounds must be a tuple (lower_array, upper_array) or None")
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if lower.size != n:
            lower = np.resize(lower, n)
        if upper.size != n:
            upper = np.resize(upper, n)
    try:
        result = least_squares(
            calculate_residuals_multi,
            initial_params_to_optimize,
            kwargs={
                'time_exp': time_exp,
                'intensity_exp': intensity_exp,
                'y0_dict': y0_dict.copy(),
                'model': model,
                'param_keys': param_keys_to_optimize,
                'param_sharing': param_sharing,
                'output_key': output_key,
                'normalization': normalization,
                'y0_dependencies': y0_dependencies,
                'save_path': save_path,
                'ode_solver': ode_solver,
                'ode_method': ode_method,
                'solve_ivp_options': solve_ivp_options
            },
            bounds=(lower, upper),
            method=method,
            verbose=verbose
        )
        if result.success:
            opt_param = result.x
            ssr = 2.0 * result.cost
            n_pts = result.fun.size if result.fun is not None else 0
            rmse = np.sqrt(ssr / max(1, n_pts))
            # compute covariance and confidence intervals depending on method
            cov = None
            stderr = None
            conf_intervals = None
            if ci_method == 't':
                p = len(param_keys_to_optimize)
                dof = max(1, n_pts - p)
                if n_pts > p and result.jac is not None:
                    res_var = ssr / dof
                    try:
                        jt_j = result.jac.T.dot(result.jac)
                        cov = np.linalg.inv(jt_j) * res_var
                        stderr_array = np.sqrt(np.diag(cov))
                        alpha = (1.0 - conf_level) / 2.0
                        tval = t.ppf(1.0 - alpha, dof)
                        lower = opt_param - tval * stderr_array
                        upper = opt_param + tval * stderr_array
                        stderr = {
                            key: val
                            for key, val in zip(param_keys_to_optimize, stderr_array)
                        }
                        conf_intervals = {
                            key: (low, high)
                            for key, low, high in zip(param_keys_to_optimize, lower, upper)
                        }
                    except np.linalg.LinAlgError:
                        cov = None
                        stderr = None
                        conf_intervals = None
                # else leave None
            elif ci_method == 'bootstrap':
                # perform residual bootstrap to obtain stderr and conf_ints
                boot = bootstrap_params(time_exp, intensity_exp, y0_dict, model,
                                         param_keys_to_optimize,
                                         initial_params_to_optimize,
                                         bounds=bounds, method=method,
                                         output_key=output_key,
                                         normalization=normalization,
                                         y0_dependencies=y0_dependencies,
                                         n_boot=n_boot, conf_level=conf_level,
                                         random_state=boot_random_state,
                                         ode_solver=ode_solver, ode_method=ode_method,
                                         solve_ivp_options=solve_ivp_options)
                stderr = boot.get('stderr_boot')
                conf_intervals = boot.get('conf_intervals_boot')
            opt_params_dict = copy.deepcopy(model.params_dict)
            for param_name, param_value in zip(param_keys_to_optimize, opt_param):
                if param_name in param_sharing:
                    for var_name in param_sharing[param_name]:
                        if var_name in opt_params_dict and param_name in opt_params_dict[var_name]:
                            opt_params_dict[var_name][param_name] = param_value
                else:
                    for var in opt_params_dict:
                        if param_name in opt_params_dict[var]:
                            opt_params_dict[var][param_name] = param_value

            opt_model = exmod.Model(
                funcs_dict=model.funcs_dict,
                params_dict=opt_params_dict,
                event_funcs=model.event_funcs if hasattr(model, 'event_funcs') else {},
                name=f"{model.name}_optimized" if hasattr(model, 'name') else "optimized_model"
            )
            return {"opt_param":opt_param, "ssr":ssr, "rmse":rmse,
                    "opt_model": opt_model,
                    "opt_params_dict":opt_params_dict,
                    "covariance":cov, "stderr":stderr,
                    "conf_intervals":conf_intervals}
        else:
            warnings.warn(f"Optimisation échouée pour: {result.message}")
            return {"opt_param": None, "ssr": None, "rmse": None}
    except Exception as e:
        warnings.warn(f"Erreur lors de l'optimisation: {e}")
        return {"opt_param": None, "ssr": None, "rmse": None}


def optim_param_dynamic_dict(time_exp_dict, intensity_exp_dict, keys_order=None, **kwargs):
    """
    Wrapper pour optimiser à partir de dictionnaires time/intensity.
    - time_exp_dict, intensity_exp_dict : dict(key -> array-like)
    - keys_order : optionnel, liste pour forcer l'ordre des clés (sinon ordre d'insertion)
    - **kwargs : arguments passés à optim_param_dynamic_multi (y0_dict, funcs_dict, params_dict, ...)
    Retour : même dict de résultat que optim_param_dynamic_multi.
    """
    if not isinstance(time_exp_dict, dict) or not isinstance(intensity_exp_dict, dict):
        raise ValueError("time_exp_dict et intensity_exp_dict doivent être des dicts")
    keys_time = set(time_exp_dict.keys())
    keys_int = set(intensity_exp_dict.keys())
    if keys_time != keys_int:
        raise ValueError("Les deux dicts doivent contenir les mêmes clés")
    if keys_order is None:
        keys = list(time_exp_dict.keys())
    else:
        if not isinstance(keys_order, (list, tuple)):
            raise ValueError("keys_order doit être une liste/tuple de clés")
        if set(keys_order) != keys_time:
            raise ValueError("keys_order doit contenir exactement les mêmes clés que les dicts")
        keys = list(keys_order)
    time_list = [np.asarray(time_exp_dict[k], dtype=float) for k in keys]
    intensity_list = [np.asarray(intensity_exp_dict[k], dtype=float) for k in keys]
    return optim_param_dynamic_multi(time_list, intensity_list, **kwargs)

def optim_param_dynamic_multiphases(time_exp, intensity_exp, stages, y0_initial,
                          param_keys_to_optimize=None, initial_params_to_optimize=None,
                          bounds=None, method='trf', output_key="CMS", normalization=True,
                          event_func_dict=None):
    """
    Optimize parameters for multi-phase models with events.
    - `stages` can be a single list of stage dicts (reused for all experiments) OR
      a list-of-lists where each element is a list of stage dicts for that experiment.
    - `y0_initial` can be a single dict or a list of dicts (one per experiment).
    Stage dict structure: {'t_start': float, 't_end': float, 'funcs': {...}, 'params': {...}, 'events': DataFrame (opt)}
    See `dynamic_model.run_multiple_stages` for expected stage format.

    Deprecated: does not work with the current stage format; use `optim_param_protocol` instead.
    """
    warnings.warn("optim_param_dynamic_multiphases is deprecated and does not work with the current "
                  "stage format; use optim_param_protocol instead.", DeprecationWarning, stacklevel=2)
    if param_keys_to_optimize is None or len(param_keys_to_optimize) == 0:
        raise ValueError("param_keys_to_optimize must be provided and non-empty")
    # Determine number of experiments
    multi = isinstance(time_exp, (list, tuple)) or isinstance(intensity_exp, (list, tuple))
    if multi:
        if not (isinstance(time_exp, (list, tuple)) and isinstance(intensity_exp, (list, tuple))):
            raise ValueError("Both time_exp and intensity_exp must be lists/tuples for multi-experiment mode")
        if len(time_exp) != len(intensity_exp):
            raise ValueError("time_exp and intensity_exp lists must have same length")
        n_exp = len(time_exp)
    else:
        n_exp = 1
    # Build automatic param_sharing across stages if possible:
    # param_sharing param_name -> list of (exp_idx, stage_idx, var_name)
    param_sharing = {}
    # Prepare a temporary stages_per_exp to inspect where parameters are located
    if isinstance(stages, (list, tuple)) and len(stages) > 0 and isinstance(stages[0], (list, tuple)):
        stages_per_exp_tmp = list(stages)
    else:
        stages_per_exp_tmp = [stages] * max(1, n_exp)
    for p in param_keys_to_optimize:
        occ = []
        for iexp, stages_i in enumerate(stages_per_exp_tmp):
            for istage, st in enumerate(stages_i):
                params_dict = st.get("params", {})
                for var in params_dict:
                    if p in params_dict[var]:
                        occ.append((iexp, istage, var))
        if not occ:
            raise ValueError(f"param_key '{p}' not found in any stage params")
        param_sharing[p] = occ
    # initial guess
    if initial_params_to_optimize is None:
        initial_params_to_optimize = []
        for pk in param_keys_to_optimize:
            # take the first occurrence found
            occ0 = param_sharing[pk][0]
            iexp0, istage0, var0 = occ0
            st0 = (stages_per_exp_tmp[iexp0][istage0])
            initial_params_to_optimize.append(st0["params"][var0][pk])
    initial_params_to_optimize = np.asarray(initial_params_to_optimize, dtype=float)
    # bounds
    n = len(param_keys_to_optimize)
    if bounds is None:
        lower = np.full(n, -np.inf)
        upper = np.full(n, np.inf)
    else:
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            raise ValueError("bounds must be a tuple (lower_array, upper_array) or None")
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        if lower.size != n:
            lower = np.resize(lower, n)
        if upper.size != n:
            upper = np.resize(upper, n)
    try:
        result = least_squares(
            calculate_residuals_multiphases,
            initial_params_to_optimize,
            kwargs={
                'time_exp': time_exp,
                'intensity_exp': intensity_exp,
                'stages': stages,
                'y0_initial': y0_initial,
                'param_keys': param_keys_to_optimize,
                'param_sharing': param_sharing,
                'output_key': output_key,
                'normalization': normalization,
                'event_func_dict': event_func_dict
            },
            bounds=(lower, upper),
            method=method
        )
        if result.success:
            opt_param = result.x
            ssr = 2.0 * result.cost
            n_pts = result.fun.size if result.fun is not None else 0
            rmse = np.sqrt(ssr / max(1, n_pts))
            # Build opt_params structure per stage (deepcopy original stages and update)
            if isinstance(stages, (list, tuple)) and len(stages) > 0 and isinstance(stages[0], (list, tuple)):
                pass  # return opt_param (not structured as before)
            return {
                "opt_param": opt_param,
                "ssr": ssr,
                "rmse": rmse,
                "success": result.success,
                "result": result
            }
        else:
            return {
                "success": False,
                "result": result
            }
    except Exception as e:
        raise RuntimeError(f"optim_param_dynamic_multiphases failed: {e}") from e


# ============================================================
# Fitting a whole protocol: several stages and instantaneous events
# ============================================================

def _as_stage_list(stages, time_exp):
    """A single Model is turned into one stage covering the experimental time range."""
    if isinstance(stages, dm.Model):
        t = np.asarray(time_exp, dtype=float)
        return [{"name": "all", "t_start": float(t.min()), "t_end": float(t.max()), "model": stages}]
    return list(stages)


def _stage_name(stage, index):
    return str(stage.get("name", index))


def _apply_protocol_params(stages, values):
    """Return new stages whose models carry the given parameter values.

    values: {"k": v} updates k in every stage where it appears (shared parameter);
            {"k@name": v} updates k only in the stage called `name` (or with that index).
    """
    new_stages = []
    for i, stage in enumerate(stages):
        name = _stage_name(stage, i)
        params = copy.deepcopy(stage["model"].params_dict)
        for key, value in values.items():
            param, _, target = key.partition("@")
            if target and target != name:
                continue
            for var in params:
                if param in params[var]:
                    params[var][param] = float(value)
        model = stage["model"]
        new_stages.append({**stage, "model": dm.Model(params, model.funcs_dict,
                                                      event_funcs=model.event_funcs, name=model.name)})
    return new_stages


def _check_protocol_params(stages, keys):
    names = [_stage_name(st, i) for i, st in enumerate(stages)]
    for key in keys:
        param, _, target = key.partition("@")
        if target and target not in names:
            raise ValueError(f"Parameter '{key}': no stage named '{target}' (stages: {names})")
        found = any(param in st["model"].params_dict[var]
                    for i, st in enumerate(stages) if not target or _stage_name(st, i) == target
                    for var in st["model"].params_dict)
        if not found:
            raise ValueError(f"Parameter '{param}' is not a parameter of "
                             + (f"stage '{target}'" if target else "any stage"))


def _per_experiment(value, n, name):
    """Accept one value shared by all experiments or a list with one value per experiment."""
    if isinstance(value, (list, tuple)) and len(value) == n and n > 1 and name != "stages":
        return list(value)
    if name == "stages" and isinstance(value, (list, tuple)) and len(value) == n and n > 1 \
            and isinstance(value[0], (list, tuple, dm.Model)):
        return list(value)
    return [value] * n


def calculate_residuals_protocol(params_vector, param_keys, time_exp, intensity_exp, stages, y0_dict,
                                 events=None, event_funcs=None, event_params=None, output_key="CMS",
                                 normalization=True, n_points=200, rtol=1e-6, atol=1e-12, mxstep=0,
                                 ode_solver="odeint", ode_method="LSODA", solve_ivp_options=None):
    """Residuals of a protocol (stages and events) against one or several experimental curves.

    The model is simulated with `dynamic_model.run_stages` (events included), then interpolated
    on the experimental times. Normalisation is the same as in `calculate_residuals`: with
    `normalization=True`, simulation and data are each divided by their own maximum.

    For several experiments, `time_exp` and `intensity_exp` are lists; `stages`, `y0_dict`,
    `events` and `event_params` can be shared or given as one value per experiment.
    """
    values = dict(zip(param_keys, params_vector))
    multi = isinstance(time_exp, (list, tuple))
    times = list(time_exp) if multi else [time_exp]
    data = list(intensity_exp) if multi else [intensity_exp]
    n = len(times)
    residuals = []
    for t_exp, y_exp, stages_i, y0_i, events_i, ev_params_i in zip(
            times, data, _per_experiment(stages, n, "stages"), _per_experiment(y0_dict, n, "y0"),
            _per_experiment(events, n, "events"), _per_experiment(event_params, n, "event_params")):
        t_exp = np.asarray(t_exp, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)
        stage_list = _as_stage_list(stages_i, t_exp)
        funcs = event_funcs
        if funcs is None:
            funcs = {}
            for st in stage_list:
                funcs.update(st["model"].event_funcs or {})
        df = dm.run_stages(_apply_protocol_params(stage_list, values), y0_i, df_event=events_i,
                           event_func_dict=funcs, n_points=n_points, event_params=ev_params_i,
                           rtol=rtol, atol=atol, mxstep=mxstep, solver=ode_solver,
                           method=ode_method, solve_ivp_options=solve_ivp_options)
        # after an event or at a stage boundary a time appears twice: keep the state after it
        df = df.drop_duplicates("time", keep="last")
        sim = np.interp(t_exp, df["time"].to_numpy(), df[output_key].to_numpy())
        if normalization:
            sim = sim / np.max(sim) if np.max(sim) > 0 else np.zeros_like(sim)
            y_exp = y_exp / np.max(y_exp) if np.max(y_exp) > 0 else y_exp
        residuals.append(sim - y_exp)
    return np.concatenate(residuals)


def optim_param_protocol(time_exp, intensity_exp, stages, y0_dict, params, bounds=None,
                 events=None, event_funcs=None, event_params=None, output_key="CMS",
                 normalization=True, n_points=200, method="trf", conf_level=0.95,
                 rtol=1e-6, atol=1e-12, mxstep=0, ode_solver="odeint", ode_method="LSODA",
                 solve_ivp_options=None, verbose=0):
    """Fit parameters of a protocol made of stages and instantaneous events (swallow, chew...).

    Complements `optim_param_dynamic`, which simulates a single model without events.

    Args:
        time_exp, intensity_exp: experimental curve, or lists of curves (several experiments
            fitted with the same parameters).
        stages: a `Model` (one stage covering the experimental times) or a list of stages
            {"name": str (optional), "t_start": float, "t_end": float, "model": Model}.
        y0_dict: initial state at the start of the first stage.
        params: {name: initial value} of the parameters to fit, in order. "k" is shared by every
            stage where it appears; "k@name" only concerns the stage called `name`.
        bounds: {name: (low, high)} for some or all parameters (others are unbounded).
        events: DataFrame of events (columns "time" and "type"), as for `run_stages`.
        event_funcs: {type: function}; by default the event functions of the stage models.
        event_params: dictionary passed to the event functions.
        Other arguments: see `calculate_residuals_protocol` and `run_stages`.

    Returns:
        dict with opt_param (array), opt_params ({name: value}), ssr, rmse, covariance,
        stderr ({name: value}), conf_intervals ({name: (low, high)}), opt_stages (the stages
        with the fitted models), success and message. On failure opt_param is None.
    """
    keys = list(params)
    if not keys:
        raise ValueError("params must contain at least one parameter to fit")
    times = list(time_exp) if isinstance(time_exp, (list, tuple)) else [time_exp]
    n_exp = len(times)
    for stages_i, t_i in zip(_per_experiment(stages, n_exp, "stages"), times):
        stage_list = _as_stage_list(stages_i, t_i)
        _check_protocol_params(stage_list, keys)
        t_i = np.asarray(t_i, dtype=float)
        if t_i.min() < stage_list[0]["t_start"] or t_i.max() > stage_list[-1]["t_end"]:
            raise ValueError("The experimental times must lie within the stages "
                             f"[{stage_list[0]['t_start']}, {stage_list[-1]['t_end']}]")
    bounds = bounds or {}
    unknown = set(bounds) - set(keys)
    if unknown:
        raise ValueError(f"Bounds given for parameters that are not fitted: {sorted(unknown)}")
    lower = np.array([bounds.get(k, (-np.inf, np.inf))[0] for k in keys], dtype=float)
    upper = np.array([bounds.get(k, (-np.inf, np.inf))[1] for k in keys], dtype=float)
    x0 = np.array([params[k] for k in keys], dtype=float)

    result = least_squares(
        calculate_residuals_protocol, x0, bounds=(lower, upper), method=method, verbose=verbose,
        kwargs=dict(param_keys=keys, time_exp=time_exp, intensity_exp=intensity_exp, stages=stages,
                    y0_dict=y0_dict, events=events, event_funcs=event_funcs, event_params=event_params,
                    output_key=output_key, normalization=normalization, n_points=n_points,
                    rtol=rtol, atol=atol, mxstep=mxstep, ode_solver=ode_solver,
                    ode_method=ode_method, solve_ivp_options=solve_ivp_options),
    )
    if not result.success:
        warnings.warn(f"Optimisation échouée : {result.message}")
        return {"opt_param": None, "ssr": None, "rmse": None, "success": False, "message": result.message}

    opt_param = result.x
    ssr = 2 * result.cost          # result.cost = 0.5 * sum(res^2)
    n_pts = result.fun.size
    rmse = np.sqrt(ssr / max(1, n_pts))
    # confidence intervals: same computation as optim_param_dynamic (ci_method='t')
    cov = stderr = conf_intervals = None
    p = len(keys)
    dof = max(1, n_pts - p)
    if n_pts > p and result.jac is not None:
        try:
            cov = np.linalg.inv(result.jac.T.dot(result.jac)) * (ssr / dof)
            stderr_array = np.sqrt(np.diag(cov))
            tval = t.ppf(1.0 - (1.0 - conf_level) / 2.0, dof)
            stderr = dict(zip(keys, stderr_array))
            conf_intervals = {k: (v - tval * s, v + tval * s) for k, v, s in zip(keys, opt_param, stderr_array)}
        except np.linalg.LinAlgError:
            cov = stderr = conf_intervals = None
    opt_params = dict(zip(keys, opt_param))
    if n_exp > 1 and isinstance(stages, (list, tuple)) and len(stages) == n_exp \
            and isinstance(stages[0], (list, tuple, dm.Model)):
        opt_stages = [_apply_protocol_params(_as_stage_list(s, t_i), opt_params) for s, t_i in zip(stages, time_exp)]
    else:
        first_t = time_exp[0] if isinstance(time_exp, (list, tuple)) else time_exp
        opt_stages = _apply_protocol_params(_as_stage_list(stages, first_t), opt_params)
    return {"opt_param": opt_param, "opt_params": opt_params, "ssr": ssr, "rmse": rmse,
            "covariance": cov, "stderr": stderr, "conf_intervals": conf_intervals,
            "opt_stages": opt_stages, "success": True, "message": result.message}

