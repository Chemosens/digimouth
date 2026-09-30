import os
import sys
from pathlib import Path
import digimouth
# Dossier des exemples : celui de ce script, ou, si on lance le code ligne par ligne dans un
# terminal (__file__ n'existe pas alors), le dossier examples/ du projet digimouth installé.
try:
    EXAMPLES_DIR = Path(__file__).resolve().parent
except NameError:
    EXAMPLES_DIR = Path(digimouth.__file__).resolve().parents[1] / "examples"

sys.path.insert(0, str(EXAMPLES_DIR))
# Dossier des résultats : examples/output/protocol_gel (ignoré par git)
resultsRepo = EXAMPLES_DIR / "output" / "protocol_gel"
os.makedirs(resultsRepo, exist_ok=True)
# Résultats complets sujet par sujet : plusieurs centaines de Mo. Par défaut, seules les courbes
# moyennes (results_std_*.csv, quelques dizaines de Ko) sont enregistrées.
SAVE_SUBJECT_RESULTS = False
import sys
import numpy as np
import pandas as pd


from digimouth import dynamic_model as dm
from digimouth import existing_models as exmod
from digimouth import graphical_methods as gmet
from digimouth import utils as utils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from dataset import fetch, experimental_mean_curve  # données publiques : doi:10.57745/OVC3RL

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


def run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel, COP=2,
                               VOP_ini=2, COA_ini=0, COL_ini=0, CFA_ini=0, CFL_ini=0, CNA_ini=0, CMS_ini=0, VOL_ini=1,
                               QSaliva=4e-2, AOAL=100.0, VOA=37.0, AFAL=60.0, VFA=30.0, VFL=0.1, VNA=11.0, tMS=1.03,
                               KAL=2.3E-02, kL=1.38e-02,VOL_m=0.1,QBreath=300,v=0.003,AOLP_coefficient=1,eval_period=(0,120),
                                   temp_in_model=False,shape_factor_in_model=True,kT=0.05,alpha=0.02,T_ini=4,v_mouth=1e-2,T_mouth=36,aolp_in_model=False,chew_factor=None,ray_product=None,SP_ini=None):
    file_to_select = metadata_gel[
        (metadata_gel['subject'] == subject) &
        (metadata_gel['rep'] == rep) &
        (metadata_gel['fop'] == fop)
    ]['file'].values[0]
    h5_to_select = f"{file_to_select}.h5"
    eval_data = ptr_gel[ptr_gel['file'] == h5_to_select]
    aroma = eval_data[['time', 'conc_ACI']].copy()
    aroma.columns = ['time', 'intensity']
    breath = eval_data[['time', 'isoprene']].copy()
    breath.columns = ['time', 'intensity']
    senso_eval = senso_gel[
        (senso_gel['subject'] == subject) &
        (senso_gel['rep'] == rep) &
        (senso_gel['fop'] == fop) &
        (senso_gel['time'] <= eval_period[1])
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
    swallow_event_chew = senso_eval_adjusted[["time", "sw"]].copy()
    swallow_event_chew.columns = ["time", "type"]
    swallow_event_chew.loc[
        swallow_event_chew['type'].isin([ "swallowImposed"]),
        'type'
    ] = "swallow"
    swallow_event_chew.loc[
         swallow_event_chew['type'].isin(["TotalSwallow"]),
         'type'
     ] = "total_swallow"
     #--- Model ---
    # AOLP_func modifiée
    chew_times = senso_eval[senso_eval['sw'] == "chew"]['time'].values
    if not temp_in_model:
        y0_dict={"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini,
                        "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini,"CMS":CMS_ini}
        res_in_vivo_gel_chew=exmod.in_vivo_diffusion_solid_model(
        tMS=tMS, kOL=kL, KOAL=KAL, v=v, QSaliva=QSaliva, AOAP=AOAL, VOA=VOA, AFAP=AFAL, 
        VFA=VFA, VNA=VNA, VFL=VFL, COP=COP, AOLP_coefficient=AOLP_coefficient,
        QNA_func=iso_interp)  
    if temp_in_model:  
        y0_dict={"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini,
                        "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini,"CMS":CMS_ini,"T":T_ini}
        res_in_vivo_gel_chew=exmod.in_vivo_diffusion_solid_model_with_temp(
        tMS=tMS, kL=kL, KAL=KAL, QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, 
        VFA=VFA, VNA=VNA, VFL=VFL, COP=COP, AOLP_coefficient=AOLP_coefficient,
        QNA_func=iso_interp, kT=kT,T_mouth=T_mouth,v_mouth=v_mouth,alpha=alpha )  
    if aolp_in_model:
        df_event=swallow_event_chew
        def make_chi_fun(df_event, delta_chi=3):
            # temps des mastications
            chew_times = np.sort(df_event.loc[df_event["type"]=="chew", "time"].values)
            def chi(t):
                # nombre de mastications déjà réalisées
                n_chews = np.sum(chew_times <= t)
                return 1 + delta_chi * n_chews
            return chi
        # Évaluation
        chi_fun=make_chi_fun(df_event)
        def A_melting(t,A0=np.pi*1,tf=60):
            return max(A0 * (1 - t/tf), 0)
        def aolp_fun(t):
            return A_melting(t) * chi_fun(t)
        y0_dict={"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini,
                        "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini,"CMS":CMS_ini,"T":T_ini}
        res_in_vivo_gel_chew=exmod.in_vivo_diffusion_solid_model_with_temp_and_AOLP(
        tMS=tMS, kL=kL, KAL=KAL, QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, 
        VFA=VFA, VNA=VNA, VFL=VFL, COP=COP, AOLP_coefficient=AOLP_coefficient,
        QNA_func=iso_interp, kT=kT,T_mouth=T_mouth,v_mouth=v_mouth,alpha=alpha,AOLP_fun=aolp_fun )  
    df_sim=dm.run_model(model=res_in_vivo_gel_chew,y0_dict=y0_dict, 
                                        df_event=swallow_event_chew, 
                                        event_func_dict=res_in_vivo_gel_chew["event_funcs"],
                                        event_params={"VFA": VFA,"VNA":VNA,"VOA":VOA,"VOL_m":VOL_m},
                                        t_start=0,t_end=120,n_points=1000)
    if shape_factor_in_model:
        res_in_vivo_gel_chew_surface=exmod.in_vivo_diffusion_solid_model_with_temp_and_surface(
                #static parameters
                tMS=tMS,kL=kL, KAL=KAL,QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VNA=VNA, VFL=VFL, COP=COP,
                # Fonctions dynamiques
                QNA_func=iso_interp, kT=kT,T_mouth=T_mouth,v_mouth=v_mouth,alpha= alpha)
        y0_init_gel_surface = {"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini, "CFA":CFA_ini, "CNA":CNA_ini,"CMS":CMS_ini, "CFL":CFL_ini,"TP":T_ini,"SP":SP_ini}
        df_sim=dm.run_model(model=res_in_vivo_gel_chew_surface, y0_dict=y0_init_gel_surface,
                                            df_event=swallow_event_chew, 
                                            event_func_dict=res_in_vivo_gel_chew_surface.event_funcs,
                                            event_params={"VFA": VFA,"VNA":VNA,"VOA":VOA,"VOL_m":VOL_m,"chew_factor":chew_factor},
                                            t_start=0,t_end=120,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)
    # --- Safety check ---
    if not isinstance(df_sim, pd.DataFrame):
        raise ValueError(f"df_sim invalide: {type(df_sim)}")
    df_sim["subject"] = subject
    df_sim["rep"] = rep
    df_sim["fop"] = fop
    return {
        "df_sim": df_sim,
        "swallow_event": swallow_event_chew,
        "breath_time": breath_time,
        "calc_breath": calc_breath,
        "aroma_eval": aroma,
        "senso_eval": senso_eval_adjusted
    }


ptr_gel=pd.read_csv(fetch("conc_aci_gel.csv"))
senso_gel=pd.read_excel(fetch("senso_gel.xlsx"))
metadata_gel=pd.read_csv(fetch("metadata_gel.csv"),sep=";")


t_sim = np.linspace(0, 100, 100)
scale_factor=1e+2
mass_factor=1e+3
ray_product_ini=8e-3*scale_factor# dans ce cas h=0.63

VOL_m=1e-6*scale_factor**3
T_ini=4
VOP_ini=2e-6*scale_factor**3
VOL_ini=VOL_m
COA_ini=0
COL_ini=0
CFA_ini=0 # Concentration de l'air dans le pharynx
CFL_ini=0
CFP_ini=0 # Concentration du produit dans le pharynx
CNA_ini=0 # Concentration dans le nez
CMS_ini=0
SP_ini=np.pi*(ray_product_ini)**2/(VOP_ini**(2/3)) # shape factor
COP = 0.4*mass_factor/(scale_factor**3)

VOA = 37.0e-6*scale_factor**3 # Air oral volume
VFA = 30.0e-6*scale_factor**3
VNA = 11.0e-6*scale_factor**3
# Instrumental characteristics
tMS_initial=0.88 #(s)

# Product characteristics
KAL=2.3E-02 
kL=1*(2.32e-04)*scale_factor # m/s
COP_ini=0.4*mass_factor/(scale_factor**3) # Concentration du produit dans l'air oral

# eFP = 0.1e-3*scale_factor
VFL=1e-6*scale_factor**3
# Dissolution parameters
kT=0.1 #thermal equilibration constant 1/20
alpha=0.5 # sensitivity of the dissolution velocity to temperature
v36= 0.15e-3*scale_factor# speed of dissolution at 36 degrees (m/s)
QSaliva = 0.025e-6*scale_factor**3# m3/s
QBreath=250e-6*scale_factor**3# m3/s

# Interfacial product/subject characteristics
AFAL = 60.0e-4*scale_factor**2
AOAL = 100.0e-4*scale_factor**2# aire

AOLP_coefficient=1.0e-3*scale_factor**2# m2/s
aolp_in_model=False
temp_in_model=False
shape_factor_in_model=True
fop="chew"
subject="Q407"
rep=1
chew_factor=1.25
v_chew=v_succ=v36

res_Q407_chew_1=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, VOL_ini=VOL_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v_chew,AOLP_coefficient=AOLP_coefficient,
                               v_mouth=v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,shape_factor_in_model=shape_factor_in_model,ray_product=ray_product_ini,chew_factor=chew_factor,SP_ini=SP_ini)
res_Q407_chew_1["senso_eval"]
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CMS": res_Q407_chew_1["df_sim"]['CMS']}, y_type="l", title="CMS", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=True)
gmet.plot_xy(res_Q407_chew_1["aroma_eval"]['time'], y={"CMS": res_Q407_chew_1["aroma_eval"]['intensity']}, y_type="l", title="CMS", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=True)

fop="succ"
res_Q407_succ_1=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,shape_factor_in_model=shape_factor_in_model,ray_product=ray_product_ini,chew_factor=chew_factor,SP_ini=SP_ini)
#kT: combien de temps le bonbon met il à arriver à 63% de 37-4 degrés ? 

res_Q407_succ_1["senso_eval"]

fig,axes=plt.subplots(3,3, figsize=(12,8 ))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CMS": res_Q407_succ_1["df_sim"]['CMS']}, y_type="l", title="Simulated succion", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[0,0])
#gmet.plot_xy(res_Q407_succ_1["aroma_eval"]['time'], y={"CMS": res_Q407_succ_1["aroma_eval"]['intensity']}, y_type="l", title="Experimental succion", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[2,2],x_lim=(0,120))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"COL": res_Q407_succ_1["df_sim"]['COL']}, y_type="l", title="COL", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[0,1])
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"VOL": res_Q407_succ_1["df_sim"]['VOL']}, y_type="l", title="VOL", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[0,2],x_lim=(0,120))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"VOP": res_Q407_succ_1["df_sim"]['VOP']}, y_type="l", title="VOP", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[1,0])
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CFA": res_Q407_succ_1["df_sim"]['CFA']}, y_type="l", title="CFA", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[1,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CFA": res_Q407_succ_1["df_sim"]['COA']}, y_type="l", title="COA", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[1,2],x_lim=(0,120))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CFA": res_Q407_succ_1["df_sim"]['TP']}, y_type="l", title="T", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[2,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CFA": res_Q407_succ_1["df_sim"]['CNA']}, y_type="l", title="CNA", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[2,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CFA": res_Q407_succ_1["df_sim"]['SP']}, y_type="l", title="Shape Factor", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[2,2],x_lim=(0,120))

fig.show()

fig,axes=plt.subplots(3,3, figsize=(12,8 ))
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CMS": res_Q407_chew_1["df_sim"]['CMS']}, y_type="l", title="Simulated chewing", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[0,0])
#gmet.plot_xy(res_Q407_chew_1["aroma_eval"]['time'], y={"CMS": res_Q407_chew_1["aroma_eval"]['intensity']}, y_type="l", title="Experimental chewing", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[2,2],x_lim=(0,120))
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"COL": res_Q407_chew_1["df_sim"]['COL']}, y_type="l", title="COL", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[0,1])
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"VOL": res_Q407_chew_1["df_sim"]['VOL']}, y_type="l", title="VOL", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[0,2],x_lim=(0,120))
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"VOP": res_Q407_chew_1["df_sim"]['VOP']}, y_type="l", title="VOP", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[1,0])
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CFA": res_Q407_chew_1["df_sim"]['CFA']}, y_type="l", title="CFA", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[1,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CFA": res_Q407_chew_1["df_sim"]['COA']}, y_type="l", title="COA", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[1,2],x_lim=(0,120))
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CFA": res_Q407_chew_1["df_sim"]['TP']}, y_type="l", title="T", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[2,0],x_lim=(0,120))

gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CFA": res_Q407_chew_1["df_sim"]['CNA']}, y_type="l", title="CNA", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[2,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CFA": res_Q407_chew_1["df_sim"]['SP']}, y_type="l", title="Shape Factor", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[2,2],x_lim=(0,120))

fig.show()


fig,axes=plt.subplots(2,2, figsize=(12,8 ))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CMS": res_Q407_succ_1["df_sim"]['CMS']}, y_type="l", title="Simulated succion", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[0,0])
gmet.plot_xy(res_Q407_succ_1["aroma_eval"]['time'], y={"CMS": res_Q407_succ_1["aroma_eval"]['intensity']}, y_type="l", title="Experimental succion", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[1,0],x_lim=(0,120))
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"CMS": res_Q407_chew_1["df_sim"]['CMS']}, y_type="l", title="Simulated chewing", x_label="Time (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[0,0])
gmet.plot_xy(res_Q407_chew_1["aroma_eval"]['time'], y={"CMS": res_Q407_chew_1["aroma_eval"]['intensity']}, y_type="l", title="Experimental chewing", x_label="Temps (s)", y_label="VOP", opt_type="l", y_col=['red'], new=False,ax=axes[1,0],x_lim=(0,120))
fig.show()


#================================
# Illustration of changing parameters
#==============================
fop="chew"

res_Q407_succ_1=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,chew_factor=chew_factor,SP_ini=SP_ini)
#kT: combien de temps le bonbon met il à arriver à 63% de 37-4 de
res_Q407_succ_1_5VOLm=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=5*VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,chew_factor=chew_factor,SP_ini=SP_ini)
res_Q407_succ_1_5v=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=5*v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,chew_factor=chew_factor,SP_ini=SP_ini)
#kT: combien de temps le bonbon met il à arriver à 63% de 37-4 de
res_Q407_succ_1_5alpha=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=v36,kT=kT,alpha=5*alpha,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,chew_factor=chew_factor,SP_ini=SP_ini)
#kT: combien de temps le bonbon met il à arriver à 63% de 37-4 de
res_Q407_succ_1_5kT=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=v36,kT=5*kT,alpha=alpha ,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,chew_factor=chew_factor,SP_ini=SP_ini)
                             
res_Q407_succ_1_5VFL=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=5*VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=v36,kT=kT,alpha=alpha ,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,chew_factor=chew_factor,SP_ini=SP_ini)
         
res_Q407_succ_1_5kOL=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=COP_ini,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=5*kL,VOL_m=VOL_m,QBreath=QBreath,v=v_succ,AOLP_coefficient=AOLP_coefficient, 
                               v_mouth=v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,aolp_in_model=aolp_in_model,chew_factor=chew_factor,SP_ini=SP_ini  )




var_to_study='VOP'
var_to_study='CMS'
fig,axes=plt.subplots(3,3, figsize=(12,8 ))
gmet.plot_xy(res_Q407_succ_1["df_sim"]['time'], y={"CMS": res_Q407_succ_1["df_sim"][var_to_study]}, y_type="l", title="Simulated succion", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[0,0])
gmet.plot_xy(res_Q407_succ_1["aroma_eval"]['time'], y={"CMS": res_Q407_succ_1["aroma_eval"]['intensity']}, y_type="l", title="Experimental succion", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[0,1],x_lim=(0,120))
gmet.plot_xy(res_Q407_succ_1_5VOLm["df_sim"]['time'], y={"CMS": res_Q407_succ_1_5VOLm["df_sim"][var_to_study]}, y_type="l", title="5 VOL_m", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[0,2])
gmet.plot_xy(res_Q407_succ_1_5VFL["df_sim"]['time'], y={"CMS": res_Q407_succ_1_5VFL["df_sim"][var_to_study]}, y_type="l", title="5 VFL", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[1,0])
gmet.plot_xy(res_Q407_succ_1_5kOL["df_sim"]['time'], y={"CMS": res_Q407_succ_1_5kOL["df_sim"][var_to_study]}, y_type="l", title="5 kL", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[1,1])
gmet.plot_xy(res_Q407_succ_1_5kT["df_sim"]['time'], y={"CMS": res_Q407_succ_1_5kT["df_sim"][var_to_study]}, y_type="l", title="5 kT", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[1,2])
gmet.plot_xy(res_Q407_succ_1_5alpha["df_sim"]['time'], y={"CMS": res_Q407_succ_1_5alpha["df_sim"][var_to_study]}, y_type="l", title="5 alpha", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[2,0])
gmet.plot_xy(res_Q407_succ_1_5v["df_sim"]['time'], y={"CMS": res_Q407_succ_1_5v["df_sim"][var_to_study]}, y_type="l", title="5 v", x_label="", y_label="VOP", opt_type="l", y_col=['blue'], new=False,ax=axes[2,1])

fig.show()


#kT: combien de temps le bonbon met il à arriver à 63% de 37-4 de
#=================================
# In vivo protocols (gusto)
#=================================
# Generate all data from experimental evaluation

# Protocol 1: chew
#=====================================
subjects=metadata_gel["subject"].unique()
results_chew=[]
for rep in range(1,3):
    for subject in subjects:
        print(subject)
        if subject =='E809':
            pass
        else:
            print(rep)
            fop="chew"
            res_sim=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=2,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v36,AOLP_coefficient=AOLP_coefficient,
                               v_mouth=v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,shape_factor_in_model=shape_factor_in_model,ray_product=ray_product_ini,chew_factor=chew_factor,SP_ini=SP_ini   )
            print("ok")
            df = pd.DataFrame(res_sim["df_sim"])
            results_chew.append(df)

 
results_df_chew = pd.concat(results_chew, ignore_index=True)


# For succion protocol
results_succ=[]
for rep in range(1,3):
    for subject in subjects:
        print(subject)
        if subject =='E809':
            pass
        else:
            fop="succ"
            res_sim=run_gel_simulation_for_subject(subject, rep, fop, metadata_gel, ptr_gel, senso_gel,COP=2,
                               VOP_ini=VOP_ini, COA_ini=COA_ini, COL_ini=COL_ini, CFA_ini=CFA_ini, CFL_ini=CFL_ini, CNA_ini=CNA_ini, CMS_ini=CMS_ini,
                               QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial,
                               KAL=KAL, kL=kL,VOL_m=VOL_m,QBreath=QBreath,v=v36,AOLP_coefficient=AOLP_coefficient,
                               v_mouth=v36,kT=kT,alpha=alpha,temp_in_model=temp_in_model,shape_factor_in_model=shape_factor_in_model,ray_product=ray_product_ini,chew_factor=chew_factor,SP_ini=SP_ini     )
            df = pd.DataFrame(res_sim["df_sim"])
            results_succ.append(df)

results_df_succ = pd.concat(results_succ, ignore_index=True)

t_common = np.linspace(0, 119, 1000)
mean_curve_chew, std_curve_chew = get_mean_std_curve(results_df_chew, t_common)
mean_curve_succ, std_curve_succ = get_mean_std_curve(results_df_succ, t_common)

# standard deviations ans mean curves for simulations
mean_std_chew_df = pd.DataFrame({
    "time": t_common,
    "mean_CMS": mean_curve_chew,
    "std_CMS": std_curve_chew
})

mean_std_succ_df = pd.DataFrame({
    "time": t_common,
    "mean_CMS": mean_curve_succ,
    "std_CMS": std_curve_succ
})





if SAVE_SUBJECT_RESULTS:
    results_df_chew.to_csv(os.path.join(resultsRepo, "results_chew.csv"), index=False)
    results_df_succ.to_csv(os.path.join(resultsRepo, "results_succ.csv"), index=False)

mean_std_chew_df.to_csv(os.path.join(resultsRepo, "results_std_chew.csv"), index=False)
mean_std_succ_df.to_csv(os.path.join(resultsRepo, "results_std_succ.csv"), index=False)

gmet.plot_xy(t_common,mean_curve_succ)
gmet.plot_xy(t_common,mean_curve_chew)

gel_expe=experimental_mean_curve("gel")  # courbe moyenne mesurée, recalculée à partir des données publiées
gel_expe = gel_expe.rename(columns={'time_bin': 'time'})
gel_expe = gel_expe.rename(columns={'mean_intensity': 'mean_CMS'})
gel_expe = gel_expe.rename(columns={'sd_intensity': 'std_CMS'})
gel_expe_chew=gel_expe[gel_expe["fop"]=="chew"].copy()
gel_expe_succ=gel_expe[gel_expe["fop"]=="succ"].copy()



fig, axes = plt.subplots(3,2, figsize=(12,8 ))
gmet.plot_xy(res_Q407_chew_1["aroma_eval"]['time'], y=res_Q407_chew_1["aroma_eval"]['intensity'],
            vertical_lines=res_Q407_chew_1["swallow_event"]['time'].values, y_type="l",
            title="a. Experimental aroma signal with swallows", x_label="Time (s)", 
            y_label="Aroma intensity",ax=axes[0,0],new=False,x_lim=(0,120),legend=False)
gmet.plot_xy(res_Q407_chew_1["df_sim"]['time'], y={"y": res_Q407_chew_1["df_sim"]['CMS']},vertical_lines=res_Q407_chew_1["senso_eval"]["time"],y_type="l",
             title="b. Simulated aroma release",x_label="Time (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[1,0],x_lim=(0,120),legend=False)
gmet.plot_xy(res_Q407_chew_1["breath_time"], y=res_Q407_chew_1["calc_breath"],y_type="l",title="c. Breathing signal",x_label="Time (s)",y_label="Concentration",opt_type="l",new=False,ax=axes[2,0],x_lim=(0,120),legend=False)

# ajout sur un autre subplot   (axes [1,0  ]) 
axes[1,1].plot( t_common, mean_curve_chew, label="FOP chew",color="orange")
axes[1,1].fill_between(t_common, mean_curve_chew - std_curve_chew / np.sqrt(90),mean_curve_chew + std_curve_chew / np.sqrt(90), alpha=0.3,    label="±1 SE (chew)",color="orange")
axes[1,1].plot( t_common, mean_curve_succ, label="FOP succ",color="green")
axes[1,1].fill_between(  t_common,  mean_curve_succ - std_curve_succ / np.sqrt(90),  mean_curve_succ + std_curve_succ / np.sqrt(90),   alpha=0.3,label="±1 SE (succ)",color="green")
axes[1,1].set_xlabel("Time (s)")
axes[1,1].set_ylabel("CMS")
axes[1,1].set_title("e. Simulations in vivo (average of all data)")

color_map = {
    "chew": "pink",
    "swallow": "cyan",
    "TotalSwallow": "cyan",
    "swallowImposed": "cyan"
}

event_lines = [
    (row["time"], color_map[row["sw"]])
    for _, row in res_Q407_chew_1["senso_eval"].iterrows()
]
for t, c in event_lines:
        axes[0,0].axvline(x=t, color=c, linestyle="--", alpha=0.8)
        axes[1,0].axvline(x=t, color=c, linestyle="--", alpha=0.8)

# plotting experimental curves on a new subplot (axes [1,1])
axes[0,1].plot(gel_expe_chew['time'], gel_expe_chew['mean_CMS'], label="Exp 'chew'", color='orange')
axes[0,1].fill_between(gel_expe_chew['time'], gel_expe_chew['mean_CMS'] - gel_expe_chew['std_CMS'] / np.sqrt(90), gel_expe_chew['mean_CMS'] +gel_expe_chew['std_CMS'] / np.sqrt(90), alpha=0.3, label="±1 SE (chew)",color='orange')
axes[0,1].plot(gel_expe_succ['time'], gel_expe_succ['mean_CMS'], label="Exp 'succ'", color='green')
axes[0,1].fill_between(gel_expe_succ['time'], gel_expe_succ['mean_CMS'] - gel_expe_succ['std_CMS'] / np.sqrt(90), gel_expe_succ['mean_CMS'] + gel_expe_succ['std_CMS'] / np.sqrt(90), alpha=0.3, label="±1 SE (succ)", color='green')
axes[0,1].set_xlabel("Time (s)")
axes[0,1].set_ylabel("CMS")
axes[0,1].set_title("d. Experimental aroma release (Panel, n=90) ")
axes[2,1].axis('off')  # masque axes et cadre ; la légende reste affichée
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_handles = [
    Line2D([0], [0], color='blue', lw=2),
    (Patch(facecolor='orange', alpha=0.3), Line2D([0], [0], color='orange', lw=2)),
    (Patch(facecolor='green', alpha=0.3), Line2D([0], [0], color='green', lw=2)),
]
legend_labels = [
    "Individual curves (one subject, one evaluation)",
    "Protocol chew: mean ±  SE",
    "Protocol succ: mean ±  SE",
]
present_events = set(res_Q407_chew_1["senso_eval"]["sw"])
for event_keys, event_label in ((("swallow", "swallowImposed", "TotalSwallow"), "Swallow"),
                                (("chew",), "Chew")):
    if present_events.intersection(event_keys):
        legend_handles.append(Line2D([0], [0], color=color_map[event_keys[0]], lw=1.5, ls="--"))
        legend_labels.append(f"{event_label} (event, dashed line)")

axes[2,1].legend(legend_handles, legend_labels, loc='center', frameon=False,
                 fontsize=11, title="Legend", title_fontsize=12)
plt.tight_layout()
plt.show()






