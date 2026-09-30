import sys
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import sympy as sp

from digimouth import dynamic_model as dm
from digimouth import existing_models as exmod
from digimouth import graphical_methods as gmet
from digimouth import utils as utils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from pathlib import Path
import digimouth
# Dossier des exemples : celui de ce script, ou, si on lance le code ligne par ligne dans un
# terminal (__file__ n'existe pas alors), le dossier examples/ du projet digimouth installé.
try:
    EXAMPLES_DIR = Path(__file__).resolve().parent
except NameError:
    EXAMPLES_DIR = Path(digimouth.__file__).resolve().parents[1] / "examples"

sys.path.insert(0, str(EXAMPLES_DIR))
from dataset import fetch, experimental_mean_curve  # données publiques : doi:10.57745/OVC3RL
# Dossier des résultats : examples/output/protocol_sol (ignoré par git)
saveRepo = EXAMPLES_DIR / "output" / "protocol_sol"
os.makedirs(saveRepo, exist_ok=True)
# Résultats complets sujet par sujet : plusieurs centaines de Mo. Par défaut, seules les courbes
# moyennes (results_std_*.csv, quelques dizaines de Ko) sont enregistrées.
SAVE_SUBJECT_RESULTS = False

def run_simulation_for_subject(subject, rep, fop, metadata_sol, ptr_sol, senso_sol,
                               VOL_ini=2, COA_ini=0, COL_ini=2, CFA_ini=0, CFL_ini=0, CNA_ini=0, CMS_ini=0,
                               QSaliva=4e-2, AOAL=100.0, VOA=37.0, AFAL=60.0, VFA=30.0, VFL=None, VNA=11.0, tMS=1.03,
                               KAL=2.3E-02, kL=1.38e-02,VOL_m=1e-3,QBreath=300,eval_period=(0,120)):
    file_to_select = metadata_sol[
        (metadata_sol['subject'] == subject) &
        (metadata_sol['rep'] == rep) &
        (metadata_sol['fop'] == fop)
    ]['file'].values[0]
    h5_to_select = f"{file_to_select}.h5"
    eval_data = ptr_sol[ptr_sol['file'] == h5_to_select]
    aroma = eval_data[['time', 'conc_ACI']].copy()
    aroma.columns = ['time', 'intensity']
    breath = eval_data[['time', 'isoprene']].copy()
    breath.columns = ['time', 'intensity']
    senso_eval = senso_sol[
        (senso_sol['subject'] == subject) &
        (senso_sol['rep'] == rep) &
        (senso_sol['fop'] == fop)&
        (senso_sol['time'] <= eval_period[1])
    ].copy()
    # --- Breath processing ---
    centered = breath["intensity"] - breath["intensity"].mean()
    calc_breath = QBreath * centered * (-1) / (centered.abs().max())
    breath_time = breath["time"]
    iso_interp = interp1d(breath_time, calc_breath, bounds_error=False, fill_value=0)
    # --- Adjust swallows safely ---
    senso_eval_adjusted = senso_eval.copy()
    def safe_assign(sw_type, adjusted_values):
        mask = senso_eval_adjusted['sw'] == sw_type
        n_mask = mask.sum()
        if n_mask == 0:
            return  # rien à faire
        if adjusted_values is None:
            print(f"⚠️ {sw_type}: adjusted_values = None")
            return
        if len(adjusted_values) != n_mask:
            print(f"⚠️ {sw_type}: mismatch mask={n_mask} vs adjusted={len(adjusted_values)}")
            return
        if len(adjusted_values) == 0:
            print(f"⚠️ {sw_type}: liste vide")
            return
        senso_eval_adjusted.loc[mask, 'time'] = np.array(adjusted_values, dtype=float)
    # --- Compute adjustments ---
    adjusted_swallows = utils.adjust_swallow_with_breath(
        breath_eval=breath,
        t_degs=senso_eval[senso_eval['sw'] == "swallow"]['time'].values,
        adjust='next_zero',
        rising_factor=0.66
    )
    adjusted_total_swallows = utils.adjust_swallow_with_breath(
        breath_eval=breath,
        t_degs=senso_eval[senso_eval['sw'] == "TotalSwallow"]['time'].values,
        adjust='next_zero',
        rising_factor=0.66
    )
    adjusted_imposed_swallows = utils.adjust_swallow_with_breath(
        breath_eval=breath,
        t_degs=senso_eval[senso_eval['sw'] == "swallowImposed"]['time'].values,
        adjust='next_zero',
        rising_factor=0.66
    )
    # --- Safe assignment ---
    safe_assign("swallow", adjusted_swallows)
    safe_assign("TotalSwallow", adjusted_total_swallows)
    safe_assign("swallowImposed", adjusted_imposed_swallows)
    # --- Build events dataframe ---
    swallow_event_rare = senso_eval_adjusted[["time", "sw"]].copy()
    swallow_event_rare.columns = ["time", "type"]
    swallow_event_rare.loc[
        swallow_event_rare['type'].isin([ "swallowImposed","TotalSwallow"]),
        'type'
    ] = "swallow"
    swallow_event_rare.loc[
        swallow_event_rare['type'].isin([ "chew"]),
        'type'
    ] = "jaw_move"
    # --- Model ---
    res_in_vivo_sol_rare = exmod.in_vivo_diffusion_solution_model(
        QNA_func=iso_interp, QSaliva=QSaliva,
        AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,
         VFL=VFL, VNA=VNA, tMS=tMS,
        KAL=KAL, kL=kL
    )
    y0_dict=        {"VOL":VOL_ini, "COA":COA_ini, "COL":COL_ini,
        "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini, "CMS":CMS_ini}
    df_sim = dm.run_model(model=res_in_vivo_sol_rare,
                          y0_dict=y0_dict,
                        event_func_dict=res_in_vivo_sol_rare["event_funcs"],

        df_event=swallow_event_rare,
        event_params={"VFA": VFA, "VNA": VNA, "VOL_m": VOL_m, "VOA": VOA},
        t_start=0, t_end=120, n_points=1000,
        rtol=1e-4, atol=1e-9, mxstep=5000
    )
    # --- Safety check ---
    if not isinstance(df_sim, pd.DataFrame):
        raise ValueError(f"df_sim invalide: {type(df_sim)}")
    df_sim["subject"] = subject
    df_sim["rep"] = rep
    df_sim["fop"] = fop
    return {
        "df_sim": df_sim,
        "swallow_event_rare": swallow_event_rare,
        "breath_time": breath_time,
        "calc_breath": calc_breath,
        "aroma_eval": aroma,
        "senso_eval": senso_eval_adjusted
    }

def get_mean_std_curve(results_df, t_common):
    interp_curves = []
    for (subject, rep), df in results_df.groupby(["subject", "rep"]):
        f = interp1d(
            df["time"],
            df["CMS"],
            bounds_error=False,
            fill_value=np.nan
        )
        interp_curves.append(f(t_common))
    interp_curves = np.array(interp_curves)
    mean_curve = np.nanmean(interp_curves, axis=0)
    std_curve  = np.nanstd(interp_curves, axis=0)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(t_common, mean_curve, label="Moyenne")
    plt.fill_between(
        t_common,
        mean_curve - std_curve/np.sqrt(90),
        mean_curve + std_curve/np.sqrt(90),
        alpha=0.3,
        label="±1 SD"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("CMS")
    plt.legend()
    plt.title("Moyenne des simulations in vivo")
    plt.show()
    return mean_curve, std_curve

def normalize_mean_std(df, multiplier=1):
    df = df.copy()
    # Supposons que la colonne 'mean' contient la courbe moyenne et 'std' l'écart-type
    max_val = df['mean_CMS'].max()
    df['mean_norm'] = df['mean_CMS'] / max_val * multiplier
    df['std_norm'] = df['std_CMS'] / max_val * multiplier
    return df

#=================================
# In vivo protocols (solution)
#=================================
# Generate all data from experimental evaluation

# Protocol 1: only one swallow event
#=====================================
results_fast=[]
ptr_sol=pd.read_csv(fetch("conc_aci_sol.csv"))
senso_sol=pd.read_excel(fetch("senso_sol.xlsx"))
metadata_sol=pd.read_csv(fetch("metadata_sol.csv"),sep=";")
subjects=metadata_sol["subject"].unique()
#==============================
# In vivo (generics)
#==============================

t_sim = np.linspace(0, 100, 100)
scale_factor=1e+2
mass_factor=1e+3
VOL_ini=2e-6*scale_factor**3
COL_ini=0.4e+3*mass_factor/scale_factor**3 # Concentration du produit dans l'air oral
COA_ini=0
CFA_ini=0 # Concentration de l'air dans le pharynx
CFL_ini=0 # Concentration du produit dans le pharynx
CNA_ini=0 # Concentration dans le nez
CMS_ini=0

VOA = 37.0e-6*scale_factor**3 # Air oral volume
VFA = 30.0e-6*scale_factor**3
VNA = 11.0e-6*scale_factor**3
QSaliva = 0.025e-6*scale_factor**3# m3/s
QBreath=250e-6*scale_factor**3# m3/s

tMS_initial=0.88 #(s)

KAL=2.3E-02 
kL=1*(2.32e-04)*scale_factor # m/s

AOAL = 100.0e-4*scale_factor**2# aire
AFAL = 30.0e-4*scale_factor**2

#VFP=AFAL * eFP # Volume du produit dans le pharynx: dilution
VFL =1e-6*scale_factor**3
VOL_m=1e-6*scale_factor**3
# One evaluation for one subject, to visualize the effect of swallow adjustment
res_Q407_long_1=run_simulation_for_subject(subject="Q407", rep=1, fop="long", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL,QBreath=QBreath, VOL_m=VOL_m)
res_Q407_long_1.keys()
fig, axes = plt.subplots(3,1, figsize=(12,8 ))
gmet.plot_xy(res_Q407_long_1["aroma_eval"]['time'], y=res_Q407_long_1["aroma_eval"]['intensity'], vertical_lines=res_Q407_long_1["swallow_event_rare"]['time'].values, y_type="l", title="Experimenatl aroma signal with swallows", x_label="Time (s)", y_label="Aroma intensity",ax=axes[0],new=False,x_lim=(0,120))
gmet.plot_xy(res_Q407_long_1["df_sim"]['time'], y={"y": res_Q407_long_1["df_sim"]['CMS']},vertical_lines=res_Q407_long_1["senso_eval"]["time"],y_type="l",title="Simulated concentration",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1],x_lim=(0,120))
gmet.plot_xy(res_Q407_long_1["breath_time"], y=res_Q407_long_1["calc_breath"],vertical_lines=res_Q407_long_1["senso_eval"]["time"],y_type="l",title="Breathing signal",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[2],x_lim=(0,120))
plt.tight_layout()
plt.show()

# Illustrating the impact of different parameters
dix=10
res_Q407_long_1=run_simulation_for_subject(subject="Q407", rep=1, fop="long", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL,QBreath=QBreath,VOL_m=VOL_m)



res_Q407_long_10_QSaliva=run_simulation_for_subject(subject="Q407", rep=1, fop="long", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva*dix, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL,QBreath=QBreath,VOL_m=VOL_m)

res_Q407_long_10_VFL=run_simulation_for_subject(subject="Q407", rep=1, fop="long", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL*dix, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath)

res_Q407_long_10_QBreath=run_simulation_for_subject(subject="Q407", rep=1, fop="long", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath*dix,VOL_m=VOL_m)

res_Q407_long_10_k=run_simulation_for_subject(subject="Q407", rep=1, fop="long", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial  ,
                                              KAL=KAL, kL=kL*dix, QBreath=QBreath,VOL_m=VOL_m)

fig, axes= plt.subplots(3,2)
gmet.plot_xy(res_Q407_long_1["df_sim"]['time'], y={"y": res_Q407_long_1["df_sim"]['CMS']},vertical_lines=res_Q407_long_1["senso_eval"]["time"],y_type="l",title="Simulated concentration",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[0,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_long_10_QSaliva["df_sim"]['time'], y={"y": res_Q407_long_10_QSaliva["df_sim"]['CMS']},vertical_lines=res_Q407_long_10_QSaliva["senso_eval"]["time"],y_type="l",title=f"QSaliva * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[0,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_long_10_QBreath["df_sim"]['time'], y={"y": res_Q407_long_10_QBreath["df_sim"]['CMS']},vertical_lines=res_Q407_long_10_QBreath["senso_eval"]["time"],y_type="l",title=f"QBreath * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_long_10_VFL["df_sim"]['time'], y={"y": res_Q407_long_10_VFL["df_sim"]['CMS']},vertical_lines=res_Q407_long_10_VFL["senso_eval"]["time"],y_type="l",title=f"VFL * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_long_10_k["df_sim"]['time'], y={"y": res_Q407_long_10_k["df_sim"]['CMS']},vertical_lines=res_Q407_long_10_k["senso_eval"]["time"],y_type="l",title=f"k * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[2,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_long_1["aroma_eval"]['time'], y=res_Q407_long_1["aroma_eval"]['intensity'], vertical_lines=res_Q407_long_1["senso_eval"]["time"].values, y_type="l", title="Experimental aroma signal with swallows", x_label="Time (s)", y_label="Aroma intensity",ax=axes[2,1],new=False,x_lim=(0,120))

plt.tight_layout()
plt.show()

# Same for fast parameter
# Illustrating the impact of different parameters
res_Q407_fast_1=run_simulation_for_subject(subject="Q407", rep=1, fop="fast", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath,VOL_m=VOL_m)

res_Q407_fast_10_QSaliva=run_simulation_for_subject(subject="Q407", rep=1, fop="fast", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva*dix, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath,VOL_m=VOL_m)
res_Q407_fast_10_VFL=run_simulation_for_subject(subject="Q407", rep=1, fop="fast", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL*dix, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath,VOL_m=VOL_m)

res_Q407_fast_10_QBreath=run_simulation_for_subject(subject="Q407", rep=1, fop="fast", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath*dix)

res_Q407_fast_10_k=run_simulation_for_subject(subject="Q407", rep=1, fop="fast", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL*dix, QBreath=QBreath,VOL_m=VOL_m)

fig, axes= plt.subplots(3,2)
gmet.plot_xy(res_Q407_fast_1["df_sim"]['time'], y={"y": res_Q407_fast_1["df_sim"]['CMS']},vertical_lines=res_Q407_fast_1["senso_eval"]["time"],y_type="l",title="Simulated concentration",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[0,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_QSaliva["df_sim"]['time'], y={"y": res_Q407_fast_10_QSaliva["df_sim"]['CMS']},vertical_lines=res_Q407_fast_10_QSaliva["senso_eval"]["time"],y_type="l",title=f"QSaliva * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[0,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_QBreath["df_sim"]['time'], y={"y": res_Q407_fast_10_QBreath["df_sim"]['CMS']},vertical_lines=res_Q407_fast_10_QBreath["senso_eval"]["time"],y_type="l",title=f"QBreath * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_VFL["df_sim"]['time'], y={"y": res_Q407_fast_10_VFL["df_sim"]['CMS']},vertical_lines=res_Q407_fast_10_VFL["senso_eval"]["time"],y_type="l",title=f"VFL * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_k["df_sim"]['time'], y={"y": res_Q407_fast_10_k["df_sim"]['CMS']},vertical_lines=res_Q407_fast_10_k["senso_eval"]["time"],y_type="l",title=f"k * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[2,0],x_lim=(0,120))
plt.tight_layout()
plt.show()


fig, axes= plt.subplots(3,2)
gmet.plot_xy(res_Q407_fast_1["df_sim"]['time'], y={"y": res_Q407_fast_1["df_sim"]['CMS']},vertical_lines=res_Q407_fast_1["senso_eval"]["time"],y_type="l",title="Simulated concentration",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[0,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_QSaliva["df_sim"]['time'], y={"y": res_Q407_fast_10_QSaliva["df_sim"]['VOL']},vertical_lines=res_Q407_fast_10_QSaliva["senso_eval"]["time"],y_type="l",title=f"QSaliva * {dix} : liquid volume",x_label="Temps (s)",y_label="Volume (VOL)",opt_type="l",new=False,ax=axes[0,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_QBreath["df_sim"]['time'], y={"y": res_Q407_fast_10_QBreath["df_sim"]['CMS']},vertical_lines=res_Q407_fast_10_QBreath["senso_eval"]["time"],y_type="l",title=f"QBreath * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_VFL["df_sim"]['time'], y={"y": res_Q407_fast_10_VFL["df_sim"]['CMS']},vertical_lines=res_Q407_fast_10_VFL["senso_eval"]["time"],y_type="l",title=f"VFL * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_10_k["df_sim"]['time'], y={"y": res_Q407_fast_10_k["df_sim"]['CMS']},vertical_lines=res_Q407_fast_10_k["senso_eval"]["time"],y_type="l",title=f"k * {dix}",x_label="Temps (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[2,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_fast_1["aroma_eval"]['time'], y=res_Q407_fast_1["aroma_eval"]['intensity'], vertical_lines=res_Q407_fast_1["senso_eval"]["time"].values, y_type="l", title="Experimental aroma signal with swallows", x_label="Time (s)", y_label="Aroma intensity",ax=axes[2,1],new=False,x_lim=(0,120))

plt.tight_layout()
plt.show()

res_Q407_fast_VOLm=run_simulation_for_subject(subject="Q407", rep=1, fop="fast", metadata_sol=metadata_sol, ptr_sol=ptr_sol, senso_sol=senso_sol,
                                              VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL*dix, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath,VOL_m=VOL_m)
gmet.plot_xy(res_Q407_fast_VOLm["df_sim"]['time'], y=res_Q407_fast_VOLm["df_sim"]['CMS'], vertical_lines=res_Q407_fast_VOLm["senso_eval"]["time"].values, y_type="l", title="Experimental aroma signal with swallows", x_label="Time (s)", y_label="Aroma intensity",ax=axes[2,1],new=True,x_lim=(0,120))

results_fast=[]
# All evaluations (fast)
for rep in range(1,3):
    for subject in subjects:
        print(subject)
        print(rep)
        fop="fast"
        res_sim=run_simulation_for_subject(subject, rep, fop, metadata_sol, ptr_sol, senso_sol, 
                                           VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath,VOL_m=VOL_m
                                              )
        df = pd.DataFrame(res_sim["df_sim"])
        results_fast.append(df)

results_df_fast = pd.concat(results_fast, ignore_index=True)

# For long protocol
subjects=metadata_sol["subject"].unique()
results_long=[]
for rep in range(1,3):
    for subject in subjects:
        fop="long"
        res_sim=run_simulation_for_subject(subject, rep, fop, metadata_sol, ptr_sol, senso_sol, VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                                              QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial,
                                              KAL=KAL, kL=kL, QBreath=QBreath,VOL_m=VOL_m
                                              )
        df = pd.DataFrame(res_sim["df_sim"])
        results_long.append(df)

results_long_df = pd.concat(results_long, ignore_index=True)

t_common = np.linspace(0, 119, 1000)
mean_curve_long, std_curve_long = get_mean_std_curve(results_long_df, t_common)
mean_curve_fast, std_curve_fast = get_mean_std_curve(results_df_fast, t_common)

# standard deviations ans mean curves for simulations
mean_std_long_df = pd.DataFrame({
    "time": t_common,
    "mean_CMS": mean_curve_long,
    "std_CMS": std_curve_long
})

mean_std_fast_df = pd.DataFrame({
    "time": t_common,
    "mean_CMS": mean_curve_fast,
    "std_CMS": std_curve_fast
})

if SAVE_SUBJECT_RESULTS:
    results_long_df.to_csv(os.path.join(saveRepo, "results_long.csv"), index=False)
    results_df_fast.to_csv(os.path.join(saveRepo, "results_fast.csv"), index=False)
mean_std_long_df.to_csv(os.path.join(saveRepo, "results_std_long.csv"), index=False)
mean_std_fast_df.to_csv(os.path.join(saveRepo, "results_std_fast.csv"), index=False)

# Reading experimental data

# plotting experimental curves on a new subplot (axes [1,1])

# Reading modeled curves
results_long_df_std = pd.read_csv(os.path.join(saveRepo, "results_std_long.csv"))
results_df_fast_std = pd.read_csv(os.path.join(saveRepo, "results_std_fast.csv"))
# Reading experimental data
sol_expe=experimental_mean_curve("sol")  # courbe moyenne mesurée, recalculée à partir des données publiées
sol_expe = sol_expe.rename(columns={'time_bin': 'time'})
sol_expe = sol_expe.rename(columns={'mean_intensity': 'mean_CMS'})
sol_expe = sol_expe.rename(columns={'sd_intensity': 'std_CMS'})
sol_expe30=sol_expe[(sol_expe["time"]>=0) & (sol_expe["time"]<=120)]

sol_expe_freq=sol_expe30[sol_expe30["fop"]=="freq"].copy()
sol_expe_rare=sol_expe30[sol_expe30["fop"]=="rare"].copy()
sol_model_long=results_long_df_std[(results_long_df_std["time"]>=0) & (results_long_df_std["time"]<=120)].copy()   
sol_model_fast=results_df_fast_std[(results_df_fast_std["time"]>=0) & (results_df_fast_std["time"]<=120)].copy()   

max_expe=max((sol_expe_rare["mean_CMS"].max(),sol_expe_freq["mean_CMS"].max()))
max_model=max(sol_model_long["mean_CMS"].max(),sol_model_fast["mean_CMS"].max())

sol_model_long["CMS_norm"] = sol_model_long["mean_CMS"] / max_model
sol_model_fast["CMS_norm"] = sol_model_fast["mean_CMS"] / max_model
sol_expe_freq["CMS_norm"] = sol_expe_freq["mean_CMS"] / max_expe
sol_expe_rare["CMS_norm"] = sol_expe_rare["mean_CMS"] / max_expe

# Normalize std
sol_model_long["std_CMS_norm"] = sol_model_long["std_CMS"] / max_model
sol_model_fast["std_CMS_norm"] = sol_model_fast["std_CMS"] / max_model
sol_expe_freq["std_CMS_norm"] = sol_expe_freq["std_CMS"] / max_expe
sol_expe_rare["std_CMS_norm"] = sol_expe_rare["std_CMS"] / max_expe


# Plots
fig, axes = plt.subplots(3,2, figsize=(12,8 ))
gmet.plot_xy(res_Q407_long_1["aroma_eval"]['time'], y=res_Q407_long_1["aroma_eval"]['intensity'], vertical_lines=res_Q407_long_1["swallow_event_rare"]['time'].values, y_type="l", title="a. Experimental aroma signal with swallows", x_label="Time (s)", y_label="Aroma intensity",ax=axes[0,0],new=False,x_lim=(0,120),legend=False)
gmet.plot_xy(res_Q407_long_1["df_sim"]['time'], y={"y": res_Q407_long_1["df_sim"]['CMS']},vertical_lines=res_Q407_long_1["senso_eval"]["time"],y_type="l",title="b. Simulated concentration",x_label="Time (s)",y_label="$C_{MS}$",opt_type="l",new=False,ax=axes[1,0],x_lim=(0,120),legend=False)
gmet.plot_xy(res_Q407_long_1["breath_time"], y=res_Q407_long_1["calc_breath"],y_type="l",title="c. Breathing signal",x_label="Time (s)",y_label="Breathing flow",opt_type="l",new=False,ax=axes[2,0],x_lim=(0,120),legend=False)
# ajout sur un autre subplot   (axes [1,0  ]) 
axes[1,1].plot( t_common, mean_curve_long, label="rare",color="green")
axes[1,1].fill_between(t_common, mean_curve_long - std_curve_long / np.sqrt(90),mean_curve_long + std_curve_long / np.sqrt(90), alpha=0.3,    label="±1 SE ('rare')",color="green")
axes[1,1].plot( t_common, mean_curve_fast, label="freq",color="orange")
axes[1,1].fill_between(  t_common,  mean_curve_fast - std_curve_fast / np.sqrt(90),  mean_curve_fast + std_curve_fast / np.sqrt(90),   alpha=0.3,label="±1 SE ('freq')",color="orange")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].set_ylabel("$C_{MS}$")
axes[1,1].set_ylim(0, 6e-4) # (0,1.8e-4) for CMS, (0,1) for normalized
axes[1,1].set_title("e. Simulations in vivo (average of all data) ")
axes[0,1].plot(sol_expe_freq['time'], sol_expe_freq['mean_CMS'], label="Exp FOP freq", color='orange')
axes[0,1].fill_between(sol_expe_freq['time'], sol_expe_freq['mean_CMS'] - sol_expe_freq['std_CMS'] / np.sqrt(90), sol_expe_freq['mean_CMS'] + sol_expe_freq['std_CMS'] / np.sqrt(90), alpha=0.3, label="±1 SE ('freq)",color='orange')
axes[0,1].plot(sol_expe_rare['time'], sol_expe_rare['mean_CMS'], label="Exp FOP rare", color='green')
axes[0,1].fill_between(sol_expe_rare['time'], sol_expe_rare['mean_CMS'] - sol_expe_rare['std_CMS'] / np.sqrt(90), sol_expe_rare['mean_CMS'] + sol_expe_rare['std_CMS'] / np.sqrt(90), alpha=0.3, label="±1 SE ('rare')", color='green')
axes[0,1].set_xlabel("Time (s)")
axes[0,1].set_ylabel("$C_{MS}$")

axes[0,1].set_title("d. Experimental data (all subjects) ")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Événements (pointillés des panneaux a et b), colorés par type — mêmes couleurs que dans le script gel
color_map = {
    "chew": "dimgray",
    "swallow": "purple",
    "TotalSwallow": "tomato",
    "swallowImposed": "purple"
}

for _, row in res_Q407_long_1["senso_eval"].iterrows():
    for ax_event in (axes[0,0], axes[1,0]):
        ax_event.axvline(x=row["time"], color=color_map.get(row["sw"], "gray"), linestyle="--", alpha=0.8)

axes[2,1].axis('off')  # masque axes et cadre ; la légende reste affichée
legend_handles = [
    Line2D([0], [0], color='blue', lw=2),
    (Patch(facecolor='green', alpha=0.3), Line2D([0], [0], color='green', lw=2)),
    (Patch(facecolor='orange', alpha=0.3), Line2D([0], [0], color='orange', lw=2)),
]
legend_labels = [
    "Individual curves (one subject, one evaluation)",
    "Protocol 'rare': mean ± 1 SE",
    "Protocol 'freq': mean ± 1 SE",
]
present_events = set(res_Q407_long_1["senso_eval"]["sw"])
for event_keys, event_label in ((("swallow", "swallowImposed"), "Swallow"),
                                (("chew",), "Chew"),
                                (("TotalSwallow",), "Total swallow")):
    if present_events.intersection(event_keys):
        legend_handles.append(Line2D([0], [0], color=color_map[event_keys[0]], lw=1.5, ls="--"))
        legend_labels.append(f"{event_label} (event, dashed line)")

axes[2,1].legend(legend_handles, legend_labels, loc='center', frameon=False,
                 fontsize=11, title="Legend", title_fontsize=12)
plt.tight_layout()
plt.show()