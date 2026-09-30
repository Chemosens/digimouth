from scipy.integrate import odeint
import numpy as np
import pandas as pd
from digimouth import optimization as opt
from digimouth import graphical_methods as gmet
from digimouth import existing_models as exmod
from digimouth import dynamic_model as dm


#==============================================
# Release model test - NEW MODEL INTERFACE
#==============================================

# Simulation parameters
Kaw_initial = 7e-6   # 7.10-3 mL/L
k_initial = 1.38e-04 # m/s
Cl_ini = 20          # (g/m3)
d = 3.7e-2           # m
A_ini = np.pi * (d / 2) ** 2  # (m^2)
Dg_ini = 2.63e-6     # (m^3/s)
Vr = 28e-6           # (m^3)
Vl = 2e-6            # (m^3)
tMS_initial = 1      # (s)
time_exp = np.linspace(0, 200, 100)

# Create model using exmod
model_release = exmod.generate_convection_model(
    Kaw_initial=Kaw_initial,
    k_initial=k_initial,
    A_ini=A_ini,
    Dg_ini=Dg_ini,
    Vr=Vr,
    Vl=Vl,
    tMS=tMS_initial,
)

print(f"Model: {model_release}")
#==============================================
# Generate synthetic experimental data
#==========================================
print("\nGenerating synthetic experimental data...")
y0_dict={"Cl":Cl_ini,"Cg":Cl_ini*Kaw_initial,"CMS":0}
model_release_sim = dm.resolution_odeint_generic(model=model_release, y0_dict=y0_dict,t_sim=time_exp)

intensity_exp = model_release_sim["CMS"] * 1.2 + np.random.normal(
    0, 0.05 * max(model_release_sim["CMS"]), len(time_exp)
)
sd_noise = 0.05 * max(model_release_sim["CMS"])

initial_param_keys=[Kaw_initial,k_initial]
param_keys_to_optimize_release=["Kaw","k"]
model_diff=exmod.generate_convection_model(Kaw_initial=initial_param_keys[0],k_initial=initial_param_keys[1],A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)

#==============================================
# Testing residuals function
#==============================================
residuals = opt.calculate_residuals(params_vector=initial_param_keys, 
                                    time_exp=time_exp, 
                                    intensity_exp=intensity_exp, 
                                    y0_dict=y0_dict, 
                                    model=model_diff, 
                                    param_keys=param_keys_to_optimize_release, 
                                    output_key="CMS", normalization=True)
np.mean(residuals)
np.std(residuals)
residuals = opt.calculate_residuals(params_vector=initial_param_keys, time_exp=time_exp, intensity_exp=intensity_exp, 
                                    y0_dict=y0_dict, 
                                    model=model_diff, 
                                    param_keys=param_keys_to_optimize_release, 
                                    output_key="CMS",
                                    normalization=True,
                                    save_path="show")

initial_param_keys2=[Kaw_initial+1e-5,k_initial+1e-3]

residuals = opt.calculate_residuals(params_vector=initial_param_keys2, 
                                    time_exp=time_exp, 
                                    intensity_exp=intensity_exp, 
                                    y0_dict=y0_dict, 
                                    model=model_diff, 
                                    param_keys=param_keys_to_optimize_release, 
                                    output_key="CMS",
                                    normalization=True,
                                    save_path="show")

#==============================================
# Testing multi residual function
#==============================================
# For one curve (should give the same result as the single residuals function)
residuals1 = opt.calculate_residuals_multi(params_vector=initial_param_keys, 
                                           time_exp=time_exp, 
                                           intensity_exp=intensity_exp, 
                                           y0_dict=y0_dict, model=model_diff, param_keys=param_keys_to_optimize_release, output_key="CMS", normalization=True,
                                           save_path="show")
np.mean(residuals1)

# For two curves
residuals2 = opt.calculate_residuals_multi(params_vector=initial_param_keys, 
                                           time_exp=[time_exp, time_exp], 
                                           intensity_exp=[intensity_exp, intensity_exp], 
                                           y0_dict=y0_dict, 
                                           model=model_diff,
                                           param_keys=param_keys_to_optimize_release, output_key="CMS", normalization=True,
                                           save_path="show")
np.mean(residuals2)

assert np.allclose(residuals1, residuals2[0:100]) 

#==============================================
# Grid optimization
#====================
grid_Kaw = np.logspace(-9, -1, 5) 
grid_k = np.logspace(-9, -1, 20)   # k entre 1e-9 et 1e-1
grid_tMS = np.linspace(0.1, 5, 20) # tMS entre 0.1 et 10 secondes
param_grids = [grid_Kaw,grid_k, grid_tMS]
param_keys_to_optimize_release_3param=["Kaw","k","tMS"]
kwargs = {
    'time_exp': time_exp,
    'intensity_exp': intensity_exp,
    'y0_dict': y0_dict,
    'model':model_diff,
    'param_keys': param_keys_to_optimize_release_3param,
    'output_key': "CMS",
    'normalization': True,
}

best_model,best_initial, best_cost, all_costs = opt.grid_search_initial_guess(
    param_grids,
    kwargs
)

all_costs_sorted = all_costs.sort_values(by="cost")
print(all_costs_sorted )
print("Best initial guess from grid:", best_initial)
y0_dict_release_grid=y0_dict.copy()
#y0_dict_release_grid["Cg"] = y0_dict_release0["Cl"]*best_initial[0]

res_model_release_grid=dm.resolution_odeint_generic(y0_dict=y0_dict, model=best_model, t_sim=time_exp, rtol=1e-6, atol=1e-12, mxstep=0)
gmet.plot_xy(time_exp,res_model_release_grid["CMS"]*intensity_exp.max()/res_model_release_grid["CMS"].max(),opt_y=intensity_exp,title=f"optim_with_grid,Kaw={best_initial[0]:.2e} k={best_initial[1]:.2e}")


#==============================================
# Optimisation functions (3 params to be optimized)
#===============================================
# from original with initial parameters (theoretically optimal reached)
res_opt_from_original=opt.optim_param_dynamic_multi(time_exp, intensity_exp,
                                       y0_dict=y0_dict, 
                                       model=model_diff,
                                       param_keys_to_optimize=param_keys_to_optimize_release_3param, 
                                       initial_params_to_optimize=[7e-06,0.000138,0.1], 
                                        bounds=([1e-9, 1e-9, 1e-3], [np.inf, np.inf, np.inf]),
                                        method='trf', output_key="CMS",verbose=2
                                        )


y0_dict_opt_from_original=y0_dict.copy()
#y0_dict_opt_from_original["Cg"]=y0_dict_release0["Cl"]*res_opt_from_original['opt_param'][0]
optimal_model_release_3param=dm.resolution_odeint_generic(y0_dict=y0_dict_opt_from_original, 
model=res_opt_from_original['opt_model'], t_sim=time_exp, rtol=1e-6, atol=1e-12, mxstep=0)

gmet.plot_xy(optimal_model_release_3param["time"],optimal_model_release_3param["CMS"]*intensity_exp.max()/optimal_model_release_3param["CMS"].max(),opt_y=intensity_exp,title=f"optim from original Kaw{res_opt_from_original['opt_params_dict']['Cg']['Kaw']:.2e}k:{res_opt_from_original['opt_params_dict']['Cg']['k']:.2e}")

# From original with noisy parameters
initial_param_keys_noise=[7e-6*1.5, 0.000138*0.5, 0.1*2]
res_opt_from_noise=opt.optim_param_dynamic_multi(time_exp, intensity_exp,
                                       y0_dict=y0_dict, 
                                       model=model_diff,
                                       param_keys_to_optimize=param_keys_to_optimize_release_3param, 
                                       initial_params_to_optimize=initial_param_keys_noise, 
                                        bounds=([1e-9, 1e-9, 1e-3], [np.inf, np.inf, np.inf]),
                                        method='trf', output_key="CMS",verbose=2
                                        )
y0_dict_opt_from_noise=y0_dict.copy()
#y0_dict_opt_from_noise["Cg"]=y0_dict_release0["Cl"]*res_opt_from_noise['opt_param'][0]
optimal_model_release_3param_noise=dm.resolution_odeint_generic(y0_dict=y0_dict_opt_from_noise, model=res_opt_from_noise['opt_model'], t_sim=time_exp, rtol=1e-6, atol=1e-12, mxstep=0)
gmet.plot_xy(optimal_model_release_3param_noise["time"],optimal_model_release_3param_noise["CMS"]*intensity_exp.max()/optimal_model_release_3param_noise["CMS"].max(),opt_y=intensity_exp,title=f"optim from noise Kaw{res_opt_from_noise['opt_params_dict']['Cg']['Kaw']:.2e}k:{res_opt_from_noise['opt_params_dict']['Cg']['k']:.2e}")

# Not so good results: wrong initialisation 


#=========================
# 2 parameters only
#=========================
param_keys_to_optimize_release_2param=["k","tMS"]
initial_param_keys_2param=[k_initial, tMS_initial]
res2=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
y0_dict_release2=y0_dict.copy()

res_opt_release2=opt.optim_param_dynamic_multi(time_exp, intensity_exp,
                                       y0_dict=y0_dict_release2, 
                                        model=res2,
                                       param_keys_to_optimize=param_keys_to_optimize_release_2param, 
                                       initial_params_to_optimize=initial_param_keys_2param, 
                                        bounds=([ 1e-9, 1e-3], [ np.inf, np.inf]),
                                        method='trf', output_key="CMS"
                                        )
res_opt_release2['opt_params_dict']
optimal_model_release2=dm.resolution_odeint_generic(y0_dict=y0_dict_release2, model=res_opt_release2['opt_model'], t_sim=time_exp, rtol=1e-6, atol=1e-12, mxstep=0)
gmet.plot_xy(optimal_model_release2["time"],optimal_model_release2["CMS"]*intensity_exp.max()/optimal_model_release2["CMS"].max(),opt_y=intensity_exp)

# Analyse de sensibilité ( grid search)
#=========================
initial_param_keys2=[k_initial+0.1, tMS_initial+0.1]
res2=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
y0_dict_release2=y0_dict.copy()
grid_k = np.logspace(-9, -1, 50)   # k entre 1e-9 et 1e-1
grid_tMS = np.linspace(0.1, 5, 25) # tMS entre 0.1 et 10 secondes
param_grids_2 = [grid_k, grid_tMS]
kwargs_2 = {
    'time_exp': time_exp,
    'intensity_exp': intensity_exp,
    'y0_dict': y0_dict_release2,
    'model':res2,
    'param_keys': param_keys_to_optimize_release_2param,
    'output_key': "CMS",
    'normalization': True
}


best_model_2,best_initial_2, best_cost,all_costs = opt.grid_search_initial_guess(
    param_grids_2,
    kwargs_2
)

print("Best initial guess from grid:", best_initial)
res_opt_release_2param_after_grid=opt.optim_param_dynamic_multi(time_exp, intensity_exp,
                                       y0_dict=y0_dict_release2, 
                                       model=best_model_2,
                                       param_keys_to_optimize=["k","tMS"], 
                                       initial_params_to_optimize=best_initial_2, 
                                        bounds=([ 1e-9, 1e-9], [ np.inf, np.inf]),
                                        method='trf', output_key="CMS",normalization=True
                                        )

res_opt_release_2param_after_grid['opt_params_dict']

optimal_model_release_2param_after_grid=dm.resolution_odeint_generic(y0_dict=y0_dict_release2, 
                                                                     model=res_opt_release_2param_after_grid['opt_model'],
                                                                     t_sim=time_exp, rtol=1e-6, atol=1e-12, mxstep=0)
gmet.plot_xy(optimal_model_release_2param_after_grid["time"],optimal_model_release_2param_after_grid["CMS"]*intensity_exp.max()/optimal_model_release_2param_after_grid["CMS"].max(),opt_y=intensity_exp)

# Test res_opt_release_multi with several files
res_opt_from_grid=opt.optim_param_dynamic_multi([time_exp,time_exp], [intensity_exp,intensity_exp],
                                       y0_dict=y0_dict_release_grid, 
                                       model=res_opt_release_2param_after_grid['opt_model'],
                                       param_keys_to_optimize=param_keys_to_optimize_release_2param, 
                                       initial_params_to_optimize=best_initial_2, 
                                        bounds=([0.1*best_initial_2[0],0.1*best_initial_2[1]],
                                                 [10*best_initial_2[0],10*best_initial_2[1]]),
                                        method='trf', output_key="CMS",verbose=2
                                        )

res_opt_from_grid['opt_param']
y0_dict_opt_from_grid=y0_dict

optimal_model_release_3param_grid=dm.resolution_odeint_generic(y0_dict=y0_dict_opt_from_grid, model=res_opt_from_grid['opt_model'], t_sim=time_exp, rtol=1e-6, atol=1e-12, mxstep=0)
gmet.plot_xy(time_exp,optimal_model_release_3param_grid["CMS"]*intensity_exp.max()/optimal_model_release_3param_grid["CMS"].max(),opt_y=intensity_exp)
res_opt_from_grid["rmse"]


#==============================================
# Testing multiphase residual function # TODO
#==============================================
residuals = opt.calculate_residuals_multiphases(
    params_vector=initial_param_keys,
    time_exp=time_exp,
    intensity_exp=intensity_exp,
    stages=[{
        "t_start": 0,
        "t_end": 100,
        "model":model_diff
    }],
    y0_initial=y0_dict,
    param_keys=param_keys_to_optimize_release,
    output_key="CMS",
    normalization=True
)

#==============================================
# Testing with dipending initial conditions on parameters
#==============================================
Kaw_initial_ic=5e-7
# Create a new model with fresh parameters
model_with_ic_dep = exmod.generate_convection_model(
    Kaw_initial=Kaw_initial_ic,
    k_initial=k_initial,
    A_ini=A_ini,
    Dg_ini=Dg_ini,
    Vr=Vr,
    Vl=Vl,
    tMS=tMS_initial,
)

# y0_dict: here Cg is set to zero initially (will be parameterized)
y0_dict = {"Cl": Cl_ini, "Cg": 0.0, "CMS": 0}

# Generate synthetic data using the constraint Cg(0) = Kaw * Cl(0)
y0_true = {"Cl": Cl_ini, "Cg": Cl_ini * Kaw_initial, "CMS": 0}


# Define the dependency: Cg should equal Kaw * Cl
y0_dependencies = {
    "Cg": lambda params, y0: params["Cg"]["Kaw"] * y0["Cl"]
}
# Test without y0_dependencies
result_without_ic_dep = opt.optim_param_dynamic(
    time_exp, intensity_exp, y0_dict, model_with_ic_dep,
    param_keys_to_optimize=["Kaw", "k"],
    initial_params_to_optimize=[Kaw_initial, k_initial],
    y0_dependencies=None,  # ← No parameterized IC
    bounds=([1e-6, 1e-5], [1e-5, 2e-4]),  # ← Tight bounds around true values!
    method='trf', output_key="CMS", normalization=True, verbose=2
)

# Test 1: Optimize Kaw and k with y0_dependencies
print("\n[Test 1] Optimizing Kaw and k WITH y0_dependencies...")
result_with_ic_dep = opt.optim_param_dynamic(
    time_exp, intensity_exp, y0_dict, model_with_ic_dep,
    param_keys_to_optimize=["Kaw", "k"],
    initial_params_to_optimize=[Kaw_initial, k_initial],
    y0_dependencies=y0_dependencies,  # ← Using parameterized IC
    bounds=([1e-6, 1e-5], [1e-5, 2e-4]),  # ← Tight bounds around true values!
    method='trf', output_key="CMS", normalization=True, verbose=2
)

print(f"\nOptimized Kaw: {result_without_ic_dep['opt_params_dict']['Cg']['Kaw']:.6e} (true: {Kaw_initial:.6e})")
print(f"Optimized k:   {result_without_ic_dep['opt_params_dict']['Cg']['k']:.6e} (true: {k_initial:.6e})")
print(f"RMSE: {result_without_ic_dep['rmse']:.6e}")

print(f"\nOptimized Kaw: {result_with_ic_dep['opt_params_dict']['Cg']['Kaw']:.6e} (true: {Kaw_initial:.6e})")
print(f"Optimized k:   {result_with_ic_dep['opt_params_dict']['Cg']['k']:.6e} (true: {k_initial:.6e})")
print(f"RMSE: {result_with_ic_dep['rmse']:.6e}")

# Plot result - use opt_model directly from result!
# Apply y0_dependencies manually before simulation
y0_for_plot = y0_dict.copy()
y0_for_plot["Cg"] = y0_dependencies["Cg"](result_with_ic_dep['opt_params_dict'], y0_for_plot)

optimal_model_ic_dep = dm.resolution_odeint_generic(
    y0_dict=y0_for_plot,  # ← Now with Cg properly applied!
    model=result_with_ic_dep['opt_model'],
    t_sim=time_exp
)

gmet.plot_xy(
    optimal_model_ic_dep["time"],
    optimal_model_ic_dep["CMS"] * intensity_exp.max() / optimal_model_ic_dep["CMS"].max(),
    opt_y=intensity_exp,
    title=f"Optimization WITH y0_dependencies | Kaw={result_with_ic_dep['opt_params_dict']['Cg']['Kaw']:.2e}, k={result_with_ic_dep['opt_params_dict']['Cg']['k']:.2e}",
    show=True
)


#==============================================
# Fitting a protocol with swallow events (and stages): optim_param_protocol
#==============================================
# optim_param_dynamic simulates the model continuously, without events. optim_param_protocol
# simulates with run_stages, so the swallows (and stages) are taken into account during the fit.
# Synthetic in vivo data: solution model, sinusoidal breathing, three swallows, known parameters.
import matplotlib.pyplot as plt

breathing = lambda t: 300 * np.sin(2 * np.pi * t / 4.0)       # breathing flow (4 s period)
VOA_s, VFA_s, VNA_s, VFL_s = 37.0, 30.0, 11.0, 30.0 * 1e-3

def solution_model(KAL=2.3e-2, tMS=1.03):
    return exmod.in_vivo_diffusion_solution_model(
        QNA_func=breathing, QSaliva=4e-2, AOAL=100.0, VOA=VOA_s, AFAL=60.0, VFA=VFA_s,
        VFL=VFL_s, VNA=VNA_s, tMS=tMS, KAL=KAL, kL=1.38e-2)

y0_solution = {"VOL": 2.0, "COA": 0.0, "COL": 2.0, "CFA": 0.0, "CFL": 0.0, "CNA": 0.0, "CMS": 0.0}
swallows = pd.DataFrame({"time": [10.0, 25.0, 40.0], "type": ["swallow"] * 3})
swallow_params = {"VOA": VOA_s, "VFA": VFA_s, "VNA": VNA_s, "VOL_m": 1e-3}
t_obs = np.linspace(0, 60, 241)
rng = np.random.default_rng(0)

# "Measured" curve: simulation with the true values KAL = 2.3e-2, tMS = 1.03, plus 3 % noise
true_model = solution_model()
truth = dm.run_model(true_model, y0_solution, t_start=0, t_end=60, df_event=swallows,
                     event_func_dict=true_model.event_funcs, event_params=swallow_params, n_points=200)
truth = truth.drop_duplicates("time", keep="last")
cms_obs = np.interp(t_obs, truth["time"], truth["CMS"]) * (1 + 0.03 * rng.standard_normal(t_obs.size))

# Fit KAL (level of the curve) and tMS (instrument time constant, shape of the peaks)
fit_protocol_res = opt.optim_param_protocol(
    t_obs, cms_obs, stages=solution_model(KAL=1e-2, tMS=0.5), y0_dict=y0_solution,
    params={"KAL": 1e-2, "tMS": 0.5}, bounds={"KAL": (1e-4, 1.0), "tMS": (0.05, 10.0)},
    events=swallows, event_params=swallow_params, normalization=False)
print("optim_param_protocol (true: KAL=0.023, tMS=1.03):", fit_protocol_res["opt_params"])

# Same fit ignoring the swallows: the model cannot reproduce the curve
fit_no_events = opt.optim_param_dynamic(
    t_obs, cms_obs, y0_solution, solution_model(KAL=1e-2, tMS=0.5), param_keys_to_optimize=["KAL", "tMS"],
    initial_params_to_optimize=[1e-2, 0.5], bounds=([1e-4, 0.05], [1.0, 10.0]), normalization=False)
print("optim_param_dynamic, without events:", fit_no_events["opt_param"])

fitted = dm.run_stages(fit_protocol_res["opt_stages"], y0_solution, df_event=swallows,
                       event_func_dict=true_model.event_funcs, event_params=swallow_params, n_points=200)
plt.figure(figsize=(8, 4))
plt.plot(t_obs, cms_obs, ".", color="grey", label="synthetic data")
plt.plot(fitted["time"], fitted["CMS"], "b", label="optim_param_protocol (with swallows)")
for t_sw in swallows["time"]:
    plt.axvline(t_sw, color="red", linestyle=":")

plt.xlabel("Time (s)"); plt.ylabel("CMS"); plt.legend(); plt.title("Fit with swallow events")
plt.show()

# A parameter can also take a different value in each stage: "KAL@before" / "KAL@after"
stages_true = [{"name": "before", "t_start": 0, "t_end": 30, "model": solution_model(KAL=2.3e-2)},
               {"name": "after", "t_start": 30, "t_end": 60, "model": solution_model(KAL=0.8e-2)}]
truth_2 = dm.run_stages(stages_true, y0_solution, df_event=swallows, event_func_dict=true_model.event_funcs,
                        event_params=swallow_params, n_points=200).drop_duplicates("time", keep="last")
cms_obs_2 = np.interp(t_obs, truth_2["time"], truth_2["CMS"]) * (1 + 0.03 * rng.standard_normal(t_obs.size))

stages_start = [{**stage, "model": solution_model(KAL=1e-2)} for stage in stages_true]

fit_stages = opt.optim_param_protocol(
    t_obs, cms_obs_2, stages=stages_start, y0_dict=y0_solution,
    params={"KAL@before": 1e-2, "KAL@after": 1e-2},
    bounds={"KAL@before": (1e-4, 1.0), "KAL@after": (1e-4, 1.0)},
    events=swallows, event_params=swallow_params, normalization=False)

print("Two stages (true: before=0.023, after=0.008):", fit_stages["opt_params"])
