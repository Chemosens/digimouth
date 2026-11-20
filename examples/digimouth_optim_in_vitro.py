from pathlib import Path
import numpy as np
import pandas as pd
import digimouth
from digimouth import optimization as opt
from digimouth import graphical_methods as gmet
from digimouth import dynamic_model as dm
from digimouth import existing_models as exmod
from digimouth import utils as utils
import matplotlib.pyplot as plt
# Dossier des exemples : celui de ce script, ou, si on lance le code ligne par ligne dans un
# terminal (__file__ n'existe pas alors), le dossier examples/ du projet digimouth installé.
try:
    EXAMPLES_DIR = Path(__file__).resolve().parent
except NameError:
    EXAMPLES_DIR = Path(digimouth.__file__).resolve().parents[1] / "examples"

# Get from data matrix
#=====================================
# Release of isoamyl acetate in vitro (PTR-MS), three repetitions after 5 min of equilibrium:
# see data/README.md for the experimental conditions.
df = pd.read_csv(EXAMPLES_DIR / "data" / "in_vitro_isoamyl_acetate_5min.csv")
selected_files_expe = ["expe_1", "expe_2", "expe_3"]
total_isoamylAcetate = df[df["file"].isin(selected_files_expe)].copy()
# filling the dictionary
x_dict = {}
y_dict = {}
z_dict = {}
for file, g in total_isoamylAcetate.groupby("file"):
    g = g.sort_values("time")   # IMPORTANT pour des courbes propres
    x_dict[file] = g["time"].values
    y_dict[file] = g["intensity"].values
    z_dict[file] = g["intensity"].values/g["intensity"].max()

gmet.plot_xy(
    x=x_dict,
    y=y_dict,
    y_type="l",
    y_col=None,            # couleurs automatiques
    title="Cinétique par fichier",
    x_label="Time (s)",
    y_label="Intensity"
)
selected_files=list(x_dict.keys())
selected_file = selected_files[0]

total_isoamylAcetate_recaled=utils.recale_time_on_peak(
     df=total_isoamylAcetate,
     file_col="file",
     time_col="time",
     intensity_col="intensity",
     baseline_max_time=30,   # zone supposée sans pic
     threshold_std=4,        # sensibilité (4–6 recommandé)
     min_time_after=0        # garder t >= 0 après recalage
 )

total_isoamylAcetate_recaled=total_isoamylAcetate_recaled[total_isoamylAcetate_recaled["time"]<120]
x_dict = {}
y_dict = {}
z_dict = {}

for file, g in total_isoamylAcetate_recaled.groupby("file"):
     g = g.sort_values("time")   # IMPORTANT pour des courbes propres
     x_dict[file] = g["time"].values
     y_dict[file] = g["intensity"].values
     z_dict[file] = g["intensity"].values/g["intensity"].max()


gmet.plot_xy(
    x=x_dict,
    y=y_dict,
    y_type="l",
    y_col=None,            # couleurs automatiques
    title="Aroma release kinetics",
    x_label="Time (s)",
    y_label="Intensity",
    show=True
)


# --- Release
Kaw_initial=2.3E-02  # 7.10-3 mL/L
k_initial=(1.38e-04) # m/s
Cl_ini=2#(g/m3) car  2.0e-3 g/L =2 ppm
d = 3.7e-2  # m
A_ini = np.pi * (d / 2) ** 2  # (m^2)
Dg_ini= 2.633e-6    # (m^3/s)
#158 mL/min = 0.158e-6 m3/s
Vr=48e-6  # (m^3)
Vl=2e-6 # (m^3)
tMS_initial=1 #(s)


# modif param

res=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
initial_param_keys=[Kaw_initial,k_initial, tMS_initial]
initial_param_keys_2=[k_initial, tMS_initial]
# For one file (1)
y0_dict_release={"Cl": Cl_ini, "Cg": Kaw_initial*Cl_ini, "CMS": 0}


number_param_to_optimize=2
bounds_3=([1e-6, 1e-5, 1e-3], [1e-1, 1e-3, 6])
bounds_2=([ 1e-5, 1e-3], [ 1e-3, 2])
y0_dep_bool=True

if y0_dep_bool:
    y0_dep = {
        "Cg": lambda params, y0: params["Cg"]["Kaw"] * y0["Cl"]
    }
else:
    y0_dep=None



name_file="expe_1"
time_exp_selected=x_dict[name_file]
intensity_exp_selected=y_dict[name_file]
initial_model_release=dm.resolution_odeint_generic(y0_dict=y0_dict_release, model=res,
                                                    t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0,
                                                    )

y0_dict_release_opt= y0_dict_release.copy()

if number_param_to_optimize==3:
    res_opt_release_1=opt.optim_param_dynamic_multi(time_exp=time_exp_selected, intensity_exp=intensity_exp_selected,
                                        y0_dict=y0_dict_release, 
                                        model=res,
                                        param_keys_to_optimize=["Kaw","k","tMS"],
                                        initial_params_to_optimize=initial_param_keys, 
                                        bounds=bounds_3,
                                        method='trf', output_key="CMS",
                                        y0_dependencies=y0_dep
                                        )


if number_param_to_optimize==2:
    res_opt_release_1=opt.optim_param_dynamic_multi(time_exp=time_exp_selected, intensity_exp=intensity_exp_selected,
                                            y0_dict=y0_dict_release, 
                                            model=res,
                                            param_keys_to_optimize=["k","tMS"],
                                            initial_params_to_optimize=initial_param_keys_2, 
                                            bounds=bounds_2,
                                            method='trf', output_key="CMS",
                                            y0_dependencies=None
                                            )

residuals_1= opt.calculate_residuals(res_opt_release_1["opt_param"], time_exp_selected,
                            intensity_exp_selected, y0_dict_release, res, ["k", "tMS"])
np.sqrt((residuals_1**2).mean())

rmse1=res_opt_release_1["rmse"]
res_opt_release_1['opt_params_dict']
opt_Kaw_1=res_opt_release_1['opt_params_dict']["Cg"]["Kaw"]
opt_k_1=res_opt_release_1['opt_params_dict']["Cg"]["k"]
opt_tMS_1=res_opt_release_1['opt_params_dict']["CMS"]["tMS"]
idx_max = np.argmax(intensity_exp_selected)
time_exp_selected[idx_max]
#fig, axes = plt.subplots(2, 2, figsize=(12,8 ))
if(y0_dep is not None):
    y0_dict_release_opt_1= y0_dict_release.copy()
    y0_dict_release_opt_1["Cg"]=y0_dep["Cg"](res_opt_release_1['opt_params_dict'], y0_dict_release_opt_1)
    optimal_model_release_1=dm.resolution_odeint_generic(y0_dict=y0_dict_release_opt_1, model=res_opt_release_1["opt_model"],
                                                        t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)
else:
    optimal_model_release_1=dm.resolution_odeint_generic(y0_dict=y0_dict_release, model=res_opt_release_1["opt_model"],
                                                    t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)


idx_max = np.argmax(intensity_exp_selected)
time_exp_selected[idx_max]
fig, axes = plt.subplots(2, 2, figsize=(12,8 ))
gmet.plot_xy(optimal_model_release_1["time"],optimal_model_release_1["CMS"]/(optimal_model_release_1["CMS"].max()),
             opt_y=intensity_exp_selected/intensity_exp_selected.max(),
             title=f"{name_file} (Kaw: {opt_Kaw_1:.2e}, k: {opt_k_1:.2e}, tMS: {opt_tMS_1:.2f}s, rmse: {rmse1:.2e})",opt_type="l",
             new=False,
             ax=axes[0,0])


# For file 2
name_file="expe_2"
time_exp_selected=x_dict[name_file]
intensity_exp_selected=y_dict[name_file]

if number_param_to_optimize==3:
    res_opt_release_2=opt.optim_param_dynamic_multi(time_exp=time_exp_selected, intensity_exp=intensity_exp_selected,
                                        y0_dict=y0_dict_release, 
                                        model=res,
                                        param_keys_to_optimize=["Kaw","k","tMS"],
                                        initial_params_to_optimize=initial_param_keys, 
                                        bounds=bounds_3,
                                        method='trf', output_key="CMS",
                                        y0_dependencies=y0_dep
                                        )

if number_param_to_optimize==2:
    res_opt_release_2=opt.optim_param_dynamic_multi(time_exp=time_exp_selected, intensity_exp=intensity_exp_selected,
                                        y0_dict=y0_dict_release, 
                                        model=res,
                                        param_keys_to_optimize=["k","tMS"],
                                        initial_params_to_optimize=initial_param_keys_2, 
                                        bounds=bounds_2,
                                        method='trf', output_key="CMS"
                                        )

rmse2=res_opt_release_2["rmse"]
res_opt_release_2['opt_params_dict']
opt_Kaw_2=res_opt_release_2['opt_params_dict']["Cg"]["Kaw"]
opt_k_2=res_opt_release_2['opt_params_dict']["Cg"]["k"]
opt_tMS_2=res_opt_release_2['opt_params_dict']["CMS"]["tMS"]

if y0_dep is not None:
    y0_dict_release_opt_2= y0_dict_release.copy()
    y0_dict_release_opt_2["Cg"]=y0_dep["Cg"](res_opt_release_2['opt_params_dict'], y0_dict_release_opt_2)
    optimal_model_release_2=dm.resolution_odeint_generic(y0_dict=y0_dict_release_opt_2, model=res_opt_release_2["opt_model"],
                                                        t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)
else:
    optimal_model_release_2=dm.resolution_odeint_generic(y0_dict=y0_dict_release, model=res_opt_release_2["opt_model"],
                                                    t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)

idx_max = np.argmax(intensity_exp_selected)
time_exp_selected[idx_max]
gmet.plot_xy(optimal_model_release_2["time"],optimal_model_release_2["CMS"]/(optimal_model_release_2["CMS"].max()),
             opt_y=intensity_exp_selected/intensity_exp_selected.max(),
             title=f"{name_file} (Kaw: {opt_Kaw_2:.2e}, k: {opt_k_2:.2e}, tMS: {opt_tMS_2:.2f}s, rmse: {rmse2:.2e})",opt_type="l",
             new=False,ax=axes[0,1])
# For one file
name_file="expe_3"
time_exp_selected=x_dict[name_file]
intensity_exp_selected=y_dict[name_file]
initial_model_release_3=dm.resolution_odeint_generic(y0_dict=y0_dict_release, model=res,
                                                    t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)

y0_dict_release_opt_3= y0_dict_release.copy()
if number_param_to_optimize==3:
    res_opt_release_3=opt.optim_param_dynamic_multi(time_exp=time_exp_selected, intensity_exp=intensity_exp_selected,
                                        y0_dict=y0_dict_release_opt_3, 
                                        model=res,
                                        param_keys_to_optimize=["Kaw","k","tMS"],
                                        initial_params_to_optimize=initial_param_keys, 
                                        bounds=bounds_3,
                                        method='trf', output_key="CMS",
                                        y0_dependencies=y0_dep
                                        )

if number_param_to_optimize==2:
    res_opt_release_3=opt.optim_param_dynamic_multi(time_exp=time_exp_selected, intensity_exp=intensity_exp_selected,
                                        y0_dict=y0_dict_release_opt_3, 
                                        model=res,
                                        param_keys_to_optimize=["k","tMS"],
                                        initial_params_to_optimize=initial_param_keys_2, 
                                        bounds=bounds_2,
                                        method='trf', output_key="CMS"
                                        )

rmse3=res_opt_release_3["rmse"]
res_opt_release_3['opt_params_dict']
opt_Kaw_3=res_opt_release_3['opt_params_dict']["Cg"]["Kaw"]
opt_k_3=res_opt_release_3['opt_params_dict']["Cg"]["k"]
opt_tMS_3=res_opt_release_3['opt_params_dict']["CMS"]["tMS"]
y0_dict_release_opt_3= y0_dict_release.copy()

if(y0_dep is not None):
    y0_dict_release_opt_3= y0_dict_release.copy()
    y0_dict_release_opt_3["Cg"]=y0_dep["Cg"](res_opt_release_3['opt_params_dict'], y0_dict_release_opt_3)
    optimal_model_release=dm.resolution_odeint_generic(y0_dict=y0_dict_release_opt_3, model=res_opt_release_3["opt_model"],
                                                        t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)
else:
    optimal_model_release=dm.resolution_odeint_generic(y0_dict=y0_dict_release, model=res_opt_release_3["opt_model"],
                                                    t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)


idx_max = np.argmax(intensity_exp_selected)
time_exp_selected[idx_max]
gmet.plot_xy(optimal_model_release["time"],optimal_model_release["CMS"]/(optimal_model_release["CMS"].max()),
             opt_y=intensity_exp_selected/intensity_exp_selected.max(),
             title=f"{name_file} (Kaw: {opt_Kaw_3:.2e}, k: {opt_k_3:.2e}, tMS: {opt_tMS_3:.2f}s, rmse: {rmse3:.2e})",opt_type="l",new=False,ax=axes[1,0])


# For several files
if number_param_to_optimize==3:
    res_opt_release_dict_multi=opt.optim_param_dynamic_dict(time_exp_dict=x_dict, intensity_exp_dict=y_dict,
                                       y0_dict=y0_dict_release, 
                                       model=res,
                                       param_keys_to_optimize=["Kaw","k","tMS"],
                                       initial_params_to_optimize=initial_param_keys, 
                                        bounds=bounds_3,
                                        method='trf', output_key="CMS",
                                        y0_dependencies=y0_dep
                                        )

if number_param_to_optimize==2:
    res_opt_release_dict_multi=opt.optim_param_dynamic_dict(time_exp_dict=x_dict, intensity_exp_dict=y_dict,
                                       y0_dict=y0_dict_release, 
                                       model=res,
                                       param_keys_to_optimize=["k","tMS"],
                                       initial_params_to_optimize=initial_param_keys_2, 
                                       bounds=bounds_2,
                                       method='trf', output_key="CMS",
                                       y0_dependencies=None
                                       )

rmse4=res_opt_release_dict_multi["rmse"]
res_opt_release_dict_multi["opt_param"]
opt_Kaw=res_opt_release_dict_multi['opt_params_dict']["Cg"]["Kaw"]
opt_k=res_opt_release_dict_multi['opt_params_dict']["Cg"]["k"]
opt_tMS=res_opt_release_dict_multi['opt_params_dict']["CMS"]["tMS"]

if(y0_dep is not None):
    y0_dict_release_opt_multi= y0_dict_release.copy()
    y0_dict_release_opt_multi["Cg"]=y0_dep["Cg"](res_opt_release_dict_multi['opt_params_dict'], y0_dict_release_opt_multi)
    optimal_model_release_multi=dm.resolution_odeint_generic(y0_dict=y0_dict_release_opt_multi, model=res_opt_release_dict_multi["opt_model"],
                                                        t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)
else:
    optimal_model_release_multi=dm.resolution_odeint_generic(y0_dict=y0_dict_release, model=res_opt_release_dict_multi["opt_model"],
                                                    t_sim=time_exp_selected, rtol=1e-6, atol=1e-12, mxstep=0)


gmet.plot_xy(
    x=x_dict,
    y=z_dict,
    y_type="l",
    opt_y=optimal_model_release_multi["CMS"]/optimal_model_release_multi["CMS"].max(),
    y_col=None,            # couleurs automatiques
    title=f"Aroma release kinetics (Kaw: {opt_Kaw:.2e}, k: {opt_k:.2e}, tMS: {opt_tMS:.2f}s, rmse: {rmse4:.2e})",
    x_label="Time (s)",
    y_label="Intensity",
    opt_type="l",
    opt_col="black",
    new=False,
    ax=axes[1,1]
)

plt.tight_layout()
plt.show()


# --- Statistiques sur les 3 répétitions (réponse au reviewer)
fits = {"expe_1": res_opt_release_1, "expe_2": res_opt_release_2, "expe_3": res_opt_release_3}
if number_param_to_optimize == 3:
    keys, dep = ["Kaw", "k", "tMS"], y0_dep
else:
    keys, dep = ["k", "tMS"], None

rows = []
for name, fit in fits.items():
    r = opt.calculate_residuals(fit["opt_param"], x_dict[name], y_dict[name],
                                y0_dict_release, res, keys, y0_dependencies=dep)
    rows.append({"run": name, "n_points": r.size,
                 "rmse": np.sqrt(np.mean(r**2)),   # identique à fit["rmse"]
                 "bias": r.mean(),                  # moyenne des résidus
                 "max_abs": np.abs(r).max()})

stats_runs = pd.DataFrame(rows)
print(stats_runs)

rmse_reps = stats_runs["rmse"].to_numpy()
rmse_reps.var(ddof=1)


