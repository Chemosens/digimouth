import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from . import graphical_methods as gmet


def r2_score(y_true, y_pred):
    """Coefficient de détermination R², mêmes résultats que sklearn.metrics.r2_score
    (sans dépendre de scikit-learn). Données constantes : 1 si prédiction parfaite, 0 sinon."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def adjust_swallow_with_breath(breath_eval, t_degs, adjust='next_zero',rising_factor=0.66, display_graph=False):
    """
    Adjust deglutition times based on breath signal zero crossings.
    Parameters:
        breath_eval (pd.DataFrame): DataFrame with 'time' and 'intensity' columns for breath signal. This signal should be periodic
        t_degs (pd.Series): Series of deglutition times to adjust   .
        rising_factor (float): Factor to adjust the zero crossing level. If zero, use exact zero crossings. If >0, shift upwards to find rising crossings.
    Returns:
        np.ndarray: Indices of adjusted deglutition times in the aroma_eval DataFrame
    """
    y=(breath_eval["intensity"]) # centering
    y_smooth = pd.Series(y).rolling(window=5, center=True, min_periods=1).median().to_numpy()
    y_smooth_centered = y_smooth - np.mean(y_smooth)
    # Finding deglutition times adjusted to zero crossing of iso signal
    t_iso=breath_eval["time"].values
    # décaler le signal pour viser les moments de remontée
    rf = rising_factor
    max_iter = 5
    crossings = []
    for _ in range(max_iter):
        y_iso = y_smooth_centered + rf * (y_smooth.max() - y_smooth.min()) / 2
        crossings = np.where(np.diff(np.sign(y_iso)))[0]
        if len(crossings) > 0:
            break
        rf *= 0.5  # réduire progressivement le décalage
    # ne garder que ceux où le signal monte
    up_crossings = []
    # parcourir les crossings
    for idx in crossings:
        if y_iso[idx+1] - y_iso[idx] > 0:  # pente positive
            # instant retenu : milieu de l'intervalle où le signal passe par zéro
            t1, t2 = t_iso[idx], t_iso[idx+1]
            t_zero = t1 + (t2-t1)/2
            up_crossings.append(t_zero)
    if display_graph:
        gmet.plot_xy(t_iso,y_iso,vertical_lines=up_crossings)
    up_crossings = np.array(up_crossings)
    adjusted_times = []
    for td in t_degs:
        next_zeros = up_crossings[up_crossings > td]
        prev_zeros = up_crossings[up_crossings < td]
        if len(next_zeros) > 0:
            adjusted_times.append(next_zeros[0])
        else:
            adjusted_times.append(prev_zeros[len(prev_zeros)-1])  # si pas de passage après
    t_degs2=pd.Series(adjusted_times)
   # iDeglut2 = np.array([ (np.abs(aroma_eval["time"].values - td)).argmin() for td in t_degs2])
    if display_graph:
        gmet.plot_xy(t_iso,y_iso,vertical_lines=t_degs2)
    return t_degs2

def compare_curves(t1, x1, t2, x2, time_window=(0,120),norm=True):
    # Normalisation
    if norm:
        x1_normalized = x1 / np.max(x1)
        x2_normalized = x2 / np.max(x2)
    else:
        x1_normalized = x1
        x2_normalized = x2
    # Interpolation de x2 sur t1
    interp_func = interp1d(t2, x2_normalized, bounds_error=False, fill_value=0)
    x2_interpolated = interp_func(t1) 
    # Calcul des métriques
    residuals=(x1_normalized - x2_interpolated)
    # Filtrage pour la fenêtre temporelle
    mask = (t1 >= time_window[0]) & (t1 <= time_window[1])
    r2_window = r2_score(x1_normalized[mask], x2_interpolated[mask])
    rmse_window = np.sqrt(np.mean((x1_normalized[mask] - x2_interpolated[mask])**2))
    # Plot des courbes
    plt.figure(figsize=(10,5))
    plt.plot(t1[mask], x1_normalized[mask], label='x1 (normalized)', color='blue')
    plt.plot(t1[mask], x2_interpolated[mask], label='x2 interpolated (normalized)', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Concentration')
    plt.title(f'Comparison of Curves\n RMSE : {rmse_window:.4f}, R² : {r2_window:.4f}')
    plt.legend()
    plt.show() 
    return r2_window,rmse_window,residuals


def detect_start(
    f_sel,
    ion="m61.065 (C3H9O+) (Corr)",
    time_col="time",
    detection_intensity=0.05,
    n_consecutive=5,
    method="max_intensity_ratio",
    noise_period=[0,20],
    signal_on_noise=3
):
    """
    Detecte le début du pic pour un ion donné dans un fichier donné.
    Retourne :
    - t_start : temps du début du pic (float)
    - None si aucun pic détecté
    """
    # Sélection du fichier
    # Si pas de données pour ce fichier
    # Signal et temps
    t = f_sel[time_col].values
    signal = f_sel[ion].values
    if method=="max_intensity_ratio":
        # Seuil de détection
        threshold = detection_intensity
        # Condition de dépassement
        above = signal > threshold
        # Recherche de n points consécutifs au-dessus du seuil
        conv = np.convolve(above, np.ones(n_consecutive, dtype=int), mode="valid")
        if np.any(conv == n_consecutive):
            idx_start = np.where(conv == n_consecutive)[0][0]
            return t[idx_start]
        else:
            return None
    elif method=="noise":
        noise_mask = (t >= noise_period[0]) & (t <= noise_period[1])
        noise_signal = signal[noise_mask]
        noise_level = np.mean(noise_signal)
        threshold = noise_level * signal_on_noise
        above = signal > threshold
        conv = np.convolve(above, np.ones(n_consecutive, dtype=int), mode="valid")
        if np.any(conv == n_consecutive):
            idx_start = np.where(conv == n_consecutive)[0][0]
            return t[idx_start]
        else:
            return None 
        

def escalier_func(t,chew_times,multiplier=1.05):
    """Retourne 2^(nombre de chews avant t)"""
    if isinstance(t, np.ndarray):
        result = np.ones_like(t, dtype=float)
        for i, chew_time in enumerate(chew_times):
            mask = t >= chew_time
            result[mask] = multiplier ** (i + 1)
        return result
    else:
        # Cas scalaire
        for i, chew_time in enumerate(chew_times):
            if t < chew_time:
                return multiplier ** i
        return multiplier ** len(chew_times)

def AOLP_func_param(t,chew_times,r0,v,A0,multiplier=1.05):
    r_t = r0 - v*t
    aolp_base = np.where(r_t >= 0, A0 * (r_t / r0)**2, 0)
    return aolp_base * escalier_func(t,chew_times,multiplier=multiplier)



def recale_time_on_peak(
    df,
    file_col="file",
    time_col="time",
    intensity_col="intensity",
    baseline_max_time=20,   # zone supposée sans pic
    threshold_std=5,        # sensibilité (4–6 recommandé)
    min_time_after=0        # garder t >= 0 après recalage
):
    """
    Recale le temps sur le début du pic pour chaque fichier
    et supprime les observations avant ce point.
    """  
    df_out = []
    for file, g in df.groupby(file_col):
        g = g.sort_values(time_col).copy()
        # --- baseline ---
        baseline = g[g[time_col] <= baseline_max_time][intensity_col]
        if baseline.empty:
            # pas de baseline exploitable
            continue
        mu = baseline.mean()
        sigma = baseline.std()
        threshold = mu + threshold_std * sigma
        # --- détection du début du pic ---
        idx_peak = g[g[intensity_col] > threshold].index
        if len(idx_peak) == 0:
            # aucun pic détecté
            continue
        t0 = g.loc[idx_peak[0], time_col]
        # --- recalage du temps ---
        g[time_col] = g[time_col] - t0
        # --- suppression des temps négatifs ---
        g = g[g[time_col] >= min_time_after]
        df_out.append(g)
    return pd.concat(df_out, ignore_index=True)
