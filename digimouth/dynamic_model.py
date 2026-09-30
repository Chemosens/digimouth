import numpy as np
from scipy.integrate import odeint, solve_ivp
import pandas as pd
import sympy as sp

class Model:
    """
    Classe générique représentant un modèle dynamique avec toutes ses composantes.
    Encapsule params_dict, funcs_dict et event_funcs pour faciliter la réutilisation.
    y0_dict (conditions initiales) est maintenant géré en dehors de la classe et passé
    comme paramètre aux fonctions.
    
    Supporte interface pour rétro-compatibilité:
    - model.params_dict, model.funcs_dict, model.event_funcs
    - model["params_dict"], model["funcs_dict"], model["event_funcs"]
    """
    def __init__(self, params_dict, funcs_dict, event_funcs=None, name=None):
        """
        Args:
            params_dict (dict): Dictionnaire de paramètres {var_name: {param_name: param_value}}
            funcs_dict (dict): Dictionnaire de fonctions dérivées {var_name: func}
            event_funcs (dict, optional): Dictionnaire de fonctions d'événements {event_name: func}
            name (str, optional): Nom descriptif du modèle
        """
        # y0_dict n'est plus stocké comme attribut
        self.params_dict = params_dict
        self.funcs_dict = funcs_dict
        self.event_funcs = event_funcs or {}
        self.name = name
    
    def __getitem__(self, key):
        """Supporte la syntaxe ancien-style: model["params_dict"] pour rétro-compatibilité"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Supporte model["params_dict"] = ... pour rétro-compatibilité"""
        setattr(self, key, value)
    
    def __contains__(self, key):
        """Supporte 'params_dict' in model pour rétro-compatibilité"""
        return hasattr(self, key)

    
    def __repr__(self):
        name_str = f" ({self.name})" if self.name else ""
        return f"Model{name_str}: {len(self.funcs_dict)} variables, {len(self.event_funcs)} event types"




def generic_model(y_dict, t, funcs_dict, params_dict):
    """
    Modèle générique d'ODE basé sur des dictionnaires.
    Args:
        y_dict (dict): Dictionnaire des variables, ex: {"Cg":0.0, "Cl":0.0, "CMS":0.0}.
        t (float): Temps actuel.
        funcs_dict (dict): Dictionnaire des fonctions dérivées, clé = nom de variable.
            Chaque fonction doit accepter (y_dict, t, **params).
            Ex: {"Cg": dCg_dt, "Cl": dCl_dt, "CMS": dCMS_dt}
        params_dict (dict): Dictionnaire de dictionnaires de paramètres pour chaque fonction.
            Clé = nom de variable, valeur = dictionnaire de paramètres pour cette fonction.
            Ex: {"Cg":{"Kaw":1.0, "k":0.5}, "Cl": {...}, "CMS":{"tMS":2.0}}
    Returns:
        list: Liste des dérivées dans le même ordre que les clés de y_dict.   
    Raises:
        ValueError: Si une fonction dans funcs_dict n'a pas de clé correspondante dans y_dict.
                    Si un paramètre fourni n'existe pas dans la signature de la fonction.
                    Si les clés des trois dictionnaires ne correspondent pas.
    """
    # ========== VALIDATIONS ==========
    # 1. Vérifier que les clés sont cohérentes entre y_dict, funcs_dict, et params_dict
    y_keys = set(y_dict.keys())
    funcs_keys = set(funcs_dict.keys())
    params_keys = set(params_dict.keys())
    if y_keys != funcs_keys:
        missing_funcs = y_keys - funcs_keys
        extra_funcs = funcs_keys - y_keys
        msg = "Clés incohérentes entre y_dict et funcs_dict:\n"
        if missing_funcs:
            msg += f"  Fonctions manquantes pour: {missing_funcs}\n"
        if extra_funcs:
            msg += f"  Fonctions en trop: {extra_funcs}"
        raise ValueError(msg)
    if y_keys != params_keys:
        missing_params = y_keys - params_keys
        extra_params = params_keys - y_keys
        msg = "Clés incohérentes entre y_dict et params_dict:\n"
        if missing_params:
            msg += f"  Paramètres manquants pour: {missing_params}\n"
        if extra_params:
            msg += f"  Paramètres en trop: {extra_params}"
        raise ValueError(msg)
    # ========== CALCUL DES DÉRIVÉES ==========
    derivatives = []
    for var_name in y_dict.keys():
        func = funcs_dict[var_name]
        params = params_dict.get(var_name, {}) 
        # Vérifier que tous les paramètres fournis existent dans la signature de la fonction
        func_params = func.__code__.co_varnames[1:func.__code__.co_argcount]
        for param_name in params.keys():
            if param_name not in func_params:
                raise ValueError(
                    f"Paramètre '{param_name}' fourni à la fonction '{func.__name__}' "
                    f"(variable '{var_name}') mais n'existe pas dans sa signature.\n"
                    f"Signature attendue: {func_params}\n"
                    f"Paramètres fournis: {tuple(params.keys())}"
                )
        # Calculer la dérivée
        derivatives.append(func(y_dict, t, **params))
    return derivatives



def resolution_odeint_generic(
    model,
    y0_dict,
    t_sim=None,
    rtol=1e-6,
    atol=1e-12,
    mxstep=0,
    solver="odeint",
    method="LSODA",
    solve_ivp_options=None,
):
    """
    Simulation générique s'appuyant sur ``generic_model``.

    ``odeint`` reste le solveur par défaut afin de préserver la reproductibilité
    des simulations existantes. ``solve_ivp`` peut être sélectionné sans modifier
    la structure des entrées du modèle ni le DataFrame retourné.
    
    Args:
        model (Model): modèle contenant les fonctions et leurs paramètres.
        y0_dict (dict): conditions initiales, par exemple ``{"Cg": 0.0}``.
        t_sim (array-like): vecteur de temps
        rtol, atol: tolérances relatives et absolues du solveur.
        mxstep: nombre maximal de pas internes pour ``odeint`` uniquement.
        solver: ``"odeint"`` (défaut, rétro-compatible) ou ``"solve_ivp"``.
        method: méthode utilisée par ``solve_ivp`` (``"LSODA"`` par défaut).
        solve_ivp_options: options supplémentaires transmises à ``solve_ivp``,
            par exemple ``{"max_step": 0.1}`` ou ``{"jac": jacobian}``.
        
    Returns:
        pd.DataFrame: colonnes ["time", <variables...>]
        
    Exemples:
        # Comportement historique
        df = resolution_odeint_generic(model, y0_dict, t_sim)

        # Même interface et même sortie, avec solve_ivp
        df = resolution_odeint_generic(
            model, y0_dict, t_sim,
            solver="solve_ivp",
            method="LSODA",
            solve_ivp_options={"max_step": 0.1},
        )
    """
    
    funcs_dict = model.funcs_dict
    params_dict = model.params_dict
    
    # Normaliser t_sim en array 1D
    t_sim = np.asarray(t_sim, dtype=float)
    if t_sim.ndim == 0:
        t_sim = np.atleast_1d(t_sim)
    # Validations structuralles avant odeint (meilleures erreurs que lors du wrapper)
    y_keys = set(y0_dict.keys())
    funcs_keys = set(funcs_dict.keys())
    params_keys = set(params_dict.keys())
    if y_keys != funcs_keys:
        raise ValueError(f"[resolution_odeint_generic] Clés incohérentes entre y0_dict et funcs_dict: "
                         f"y0 keys={y_keys}, funcs keys={funcs_keys}")
    if y_keys != params_keys:
        raise ValueError(f"[resolution_odeint_generic] Clés incohérentes entre y0_dict et params_dict: "
                         f"y0 keys={y_keys}, params keys={params_keys}")
    # Vérifier que toutes les fonctions sont callables
    noncallable = [k for k, f in funcs_dict.items() if not callable(f)]
    if noncallable:
        raise TypeError(f"[resolution_odeint_generic] Fonctions non-callables trouvées pour: {noncallable}")
    # Ordre des variables pour odeint : alphabetic order
    keys = list(y0_dict.keys())
    y0_list = [y0_dict[k] for k in keys]
    # Convertit le vecteur du solveur en dictionnaire pour generic_model.
    def evaluate_derivatives(y_list, t):
        y_dict = dict(zip(keys, y_list))
        try:
            derivs = generic_model(y_dict, t, funcs_dict, params_dict)
        except Exception as e:
            # Fournir contexte utile pour le debug
            raise RuntimeError(f"[resolution_odeint_generic] erreur dans generic_model au temps {t}: {e}") from e
        # generic_model peut retourner liste ou dict
        if isinstance(derivs, dict):
            return [derivs[k] for k in keys]
        return derivs

    solver = solver.lower()
    if solver not in {"odeint", "solve_ivp"}:
        raise ValueError(
            "[resolution_odeint_generic] solver doit être 'odeint' ou 'solve_ivp'."
        )

    # Exécuter le solveur demandé. L'orientation finale reste toujours
    # (nombre de temps, nombre de variables) pour préserver la sortie historique.
    try:
        if solver == "odeint":
            def odeint_wrapper(y_list, t):
                return evaluate_derivatives(y_list, t)

            solution = odeint(
                odeint_wrapper,
                y0_list,
                t_sim,
                atol=atol,
                rtol=rtol,
                mxstep=mxstep,
            )
        else:
            if t_sim.size == 0:
                raise ValueError("t_sim ne peut pas être vide.")
            if t_sim.size == 1 or np.all(t_sim == t_sim[0]):
                solution = np.tile(
                    np.asarray(y0_list, dtype=float), (t_sim.size, 1)
                )
            else:
                options = dict(solve_ivp_options or {})
                reserved = {"fun", "t_span", "y0", "t_eval", "method", "rtol", "atol"}
                conflicting = reserved.intersection(options)
                if conflicting:
                    raise ValueError(
                        "solve_ivp_options ne doit pas redéfinir: "
                        + ", ".join(sorted(conflicting))
                    )

                result = solve_ivp(
                    fun=lambda t, y: evaluate_derivatives(y, t),
                    t_span=(t_sim[0], t_sim[-1]),
                    y0=y0_list,
                    t_eval=t_sim,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                    **options,
                )
                if not result.success:
                    raise RuntimeError(result.message)
                solution = result.y.T
    except Exception as e:
        raise RuntimeError(f"[resolution_odeint_generic] {solver} a échoué: {e}") from e
    # Construire DataFrame résultat
    df = pd.DataFrame(solution, columns=keys)
    df.insert(0, "time", t_sim)
    return df

def resolution_dsolve_generic(y0_dict, funcs_dict, params_dict, t_symbol=None, t0=0):
    """
    Résolution symbolique générique d'un système d'EDO couplées basé sur des dictionnaires.
    Cette version tente de résoudre le système simultanément.

    Args:
        y0_dict (dict): Conditions initiales, ex: {"Cg": 1.0, "Cl": 0.0}
        funcs_dict (dict): Fonctions dérivées, clé = nom de variable.
            Chaque fonction doit accepter (y_dict, t, **params)
        params_dict (dict): Dictionnaire des paramètres pour chaque fonction.
        t_symbol (sympy.Symbol, optionnel): symbole du temps (défaut = 't')
        t0 (float): temps initial pour les conditions initiales (défaut = 0)

    Returns:
        dict: Solutions symboliques {nom_variable: expression} (pas Eq, expressions simplifiées)
    """
    if t_symbol is None:
        t_symbol = sp.Symbol('t')
    y_keys = list(y0_dict.keys())
    if set(y_keys) != set(funcs_dict.keys()) or set(y_keys) != set(params_dict.keys()):
        raise ValueError("Les clés de y0_dict, funcs_dict et params_dict doivent correspondre.")
    # Créer des fonctions symboliques vectorielles
    y_funcs = {k: sp.Function(k) for k in y_keys}
    # Construire la liste d'équations couplées
    equations = []
    for var_name in y_keys:
        f_rhs = funcs_dict[var_name](y_funcs, t_symbol, **params_dict[var_name])
        eq = sp.Eq(sp.diff(y_funcs[var_name](t_symbol), t_symbol), f_rhs)
        equations.append(eq)
    # Préparer les conditions initiales sous forme de dictionnaire {f(t0): valeur}
    ics = {y_funcs[k](t0): v for k, v in y0_dict.items()}
    # Essayer de résoudre le système complet
    try:
        sols = sp.dsolve(equations, ics=ics)
    except Exception:
        # fallback : résoudre variable par variable si le système échoue
        sols = {}
        for var_name, eq in zip(y_keys, equations):
            try:
                sol = sp.dsolve(eq, ics={y_funcs[var_name](t0): y0_dict[var_name]})
            except Exception:
                sol = sp.dsolve(eq)
            sols[var_name] = sol
    # Normaliser la sortie : dictionnaire {variable: expression simplifiée}
    solutions = {}
    if isinstance(sols, list):  # Cas système résolu simultanément : liste de Eq
        for eq in sols:
            var = eq.lhs.func.__name__
            solutions[var] = sp.simplify(eq.rhs)
    elif isinstance(sols, dict):  # Cas fallback variable par variable
        for var_name, eq in sols.items():
            solutions[var_name] = sp.simplify(eq.rhs if isinstance(eq, sp.Equality) else eq)
    return solutions

def apply_event_func(y_dict, ev_row, event_func_dict=None, event_params=None):
    """
    Applique l'événement ev_row sur l'état y_dict en utilisant event_func_dict.
    """
    event_type = ev_row.get("type")
    if event_func_dict and event_type in event_func_dict:
        handler = event_func_dict[event_type]
        # Cas 1 : handler simple, signature = handler(y)
        if callable(handler):
            return handler(y_dict,params_current=event_params)
        # Cas 2 : handler = (func, kwargs)
        if isinstance(handler, (tuple, list)) and callable(handler[0]):
            func = handler[0]
            kwargs = handler[1] if len(handler) > 1 else {}
            return func(y_dict, params_current=event_params, **kwargs)
        raise TypeError(f"Handler mal défini pour l'événement '{event_type}'")
    raise ValueError(f"Événement inconnu : {event_type}")

def run_model(model, y0_dict,  t_start=None, t_end=None, df_event=None, event_func_dict=None, event_params=None,n_points=200,
                     rtol=1e-6, atol=1e-12, mxstep=0, solver="odeint",
                     method="LSODA", solve_ivp_options=None):
    """
    Simule un segment de temps pour un système d'équations différentielles avec 
    possibilité d'appliquer des événements instantanés à certains instants.
    Nouveau format: run_single_stage(model, t_start, t_end, ...)
    
    Cette fonction est conçue pour être utilisée dans un pipeline multi-stage,
    où chaque stage peut avoir ses propres équations, paramètres et événements.
    Parameters
    ----------
    y0_dict : dict or Model
        Dictionnaire contenant les valeurs initiales des variables du système,
        ou objet Model.
        Exemple : {"Cg": 5.0, "Cl": 2.0, "CMS": 1.2}    
    funcs_dict : dict or float
        Dictionnaire des fonctions représentant les équations différentielles
        du système, ou t_start si format Model.
        Exemple : {"Cg": f_Cg, "Cl": f_Cl, "CMS": f_CMS}
    params_dict : dict
        Dictionnaire contenant les paramètres nécessaires aux fonctions ODE.
        Les noms doivent correspondre à ceux utilisés dans `funcs_dict`.
    t_start : float
        Temps initial de la simulation pour ce stage.
    t_end : float
        Temps final de la simulation pour ce stage.
    df_event : pandas.DataFrame
        DataFrame contenant les événements à appliquer pendant ce stage.
        Colonnes attendues :
            - "time" : temps de l'événement
            - "type" : type d'événement (ex: "multiply", "add", "reset", "transfer")
            - "value" : valeur ou coefficient de l'événement
            - "details" : liste des variables concernées
        Si None ou vide, aucun événement n'est appliqué.
    apply_event_func : function
        Fonction qui applique un événement à l'état du système.
        Signature : apply_event_func(y_dict, ev_row) -> y_dict modifié
    Returns
    -------
    pandas.DataFrame
        DataFrame contenant l'évolution des variables du système pendant tout
        le stage, avec les événements appliqués aux instants définis.
        L'index est réinitialisé pour une continuité temporelle.
    Notes
    -----
    - Les événements sont appliqués **immédiatement** après la simulation jusqu'à 
      leur temps.
    - Les variables qui ne changent pas dans un événement restent constantes.
    - La simulation est segmentée entre chaque événement pour assurer la continuité.
    """
    if df_event is not None and not df_event.empty:
        df_event = df_event.sort_values("time")
    t_current = t_start
    y_current = y0_dict.copy()
    results = []
    if df_event is not None:
        for _, ev in df_event.iterrows(): #ev = df_event.iloc[0] 
            t_event = ev["time"]
            # simuler jusqu'à l'événement
            t_sim = np.linspace(t_current, t_event, n_points)
            res = resolution_odeint_generic(
                model, y0_dict=y_current, t_sim=t_sim[:-1], rtol=rtol,
                atol=atol, mxstep=mxstep, solver=solver, method=method,
                solve_ivp_options=solve_ivp_options,
            )
            results.append(res)
            # état avant événement
            y_last = {col: res[col].iloc[-1] for col in res.columns if col != "time"} # on selectionne la derniere variable 
            # reordering y_last in the same order than y0_dict to avoid issues with event functions that expect a certain order
            y_last = {k: y_last[k] for k in y0_dict.keys()}
            # appliquer l’événement
            y_ev = apply_event_func(y_last, ev, event_func_dict=event_func_dict, event_params=event_params)
            t_current = t_event
            y_current=y_ev.copy()
            add_line=pd.DataFrame({ "time": t_current,**y_current,}, index=[0])
            results.append(add_line)# on ajoute une ligne avec les nouvelles conditions initiales après l'événement
    # dernière simulation jusqu'à t_end
    # et cas si pas d'événement du tout : on simule de t_start à t_end directement
    t_sim = np.linspace(t_current, t_end, n_points)
    res = resolution_odeint_generic(
        model, y0_dict=y_current, t_sim=t_sim, rtol=rtol, atol=atol,
        mxstep=mxstep, solver=solver, method=method,
        solve_ivp_options=solve_ivp_options,
    )
    results.append(res)
    final = pd.concat(results, axis=0).reset_index(drop=True)
    return final

def run_stages(stages, y0_dict, df_event=None, event_func_dict=None,n_points=200,
               event_params=None, rtol=1e-6, atol=1e-12, mxstep=0,
               solver="odeint", method="LSODA", solve_ivp_options=None):
    """
    Simule un protocole complet composé de plusieurs stages consécutifs,
    chacun pouvant avoir ses propres équations, paramètres et événements instantanés.
    Cette fonction assemble les résultats de tous les stages pour produire
    une timeline continue et cohérente du système.

    Parameters
    ----------
    stages : list of dict
        Liste de dictionnaires décrivant chaque stage du protocole. 
        Chaque stage doit contenir :
            - "t_start" : float, temps initial du stage
            - "t_end" : float, temps final du stage
            - "model"
            - "events" (optionnel) : pandas.DataFrame d'événements à appliquer       
    y0_initial : dict
        Dictionnaire contenant les valeurs initiales du système pour le
        premier stage.
        Exemple : {"Cg": 5.0, "Cl": 2.0, "CMS": 1.2}
    
    apply_event_func : function, optional
        Fonction appliquant les événements instantanés à l'état du système.
        Signature : apply_event_func(y_dict, ev_row) -> y_dict modifié
        Par défaut : `apply_event_default`.
    Returns
    -------
    pandas.DataFrame
        DataFrame contenant l'évolution de toutes les variables du système
        pendant tous les stages, concaténée en une timeline continue.
        L'index est réinitialisé pour assurer la continuité.
    Notes
    -----
    - Les conditions initiales de chaque stage sont automatiquement définies
      par le **dernier état** du stage précédent.
    - Les événements sont appliqués **au sein de chaque stage** selon la DataFrame
      `events`.
    - Idéal pour simuler des protocoles multi-stage complexes (différents modèles,
      paramètres, événements instantanés).
    """
    y_current = y0_dict.copy()
    all_results = []
    for stage in stages:
        time1 = stage["t_start"]
        time2 = stage["t_end"]
        if df_event is not None:
            df_subset_event = df_event[(df_event['time'] >= time1) & (df_event['time'] < time2)]
            if df_subset_event.empty:
                df_subset_event = None
        else:
            df_subset_event = None
        res = run_model(
            model=stage['model'],
            y0_dict=y_current,
            t_start=stage["t_start"],
            t_end=stage["t_end"],
            df_event=   df_subset_event,
            event_func_dict=event_func_dict,
            n_points=n_points,
            event_params=event_params,
            rtol=rtol,
            atol=atol,
            mxstep=mxstep,
            solver=solver,
            method=method,
            solve_ivp_options=solve_ivp_options,
        )
        all_results.append(res)
        # conditions initiales du stage suivant = dernier état
        y_current = {col: res[col].iloc[-1] for col in res.columns if col != "time"}
        
    return pd.concat(all_results, axis=0).reset_index(drop=True)
