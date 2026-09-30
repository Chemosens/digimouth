
import numpy as np
import pandas as pd
from digimouth import dynamic_model as dm
from digimouth import existing_models as exmod
from digimouth import graphical_methods as gmet
import matplotlib.pyplot as plt

# Manual entering of a diffusion model
#====================================
# --- Initial parameters ---
Kaw_initial=7e-6  # 7.10-3 mL/L
k_initial=1.38e-04  # m/s
Cl_ini=2 #(g/m3) car  2.0e-2 g/L (1ppm = 1mg/L, donc 2ppm = 2mg/L = 2e-3 g/L = 2 g/m3)
d = 3.7e-2  # m
A_ini = np.pi * (d / 2) ** 2  # (m^2)
Dg_ini= 2.63e-6   # (m^3/s)
Vr=28e-6  # (m^3)
Vl=2e-6 # (m^3)

params_dict = {
    "Cg": {"Kaw": Kaw_initial, "k": k_initial , "Vr": Vr, "A": A_ini, "Dg":Dg_ini},
    "Cl": {"Kaw": Kaw_initial, "k":k_initial , "Vl": Vl, "A": A_ini},
    "CMS": {"tMS": 0.9}
}
# Defining the ODEs
def dCg_dt(y, t, Kaw, k, Vr, A, Dg):
    return ((k * A) / Vr) * (Kaw * y["Cl"] - y["Cg"]) - (Dg / Vr) * y["Cg"]

def dCl_dt(y, t, Kaw, k, Vl, A):
    return - ((k * A) / Vl) * (Kaw * y["Cl"] - y["Cg"])

def dCMS_dt(y, t, tMS):
    return (y["Cg"] - y["CMS"]) / tMS

#  creating dictionnaries
funcs_dict = {
    "Cg": dCg_dt,
    "Cl": dCl_dt,
    "CMS": dCMS_dt
}

mod_exple=exmod.Model( params_dict, funcs_dict, name="diff_model")

# Creating parameters and initial conditions dictionnaries


y0_dict = {"Cg": Kaw_initial * Cl_ini, "Cl": Cl_ini, "CMS": 0.0}

# Using resolution_odeint_generic for simulating the model
#=====================================================================
#Simulation with old format (dict-based)
t_sim = np.linspace(0, 50, 100)
df_result = dm.resolution_odeint_generic(model=mod_exple,y0_dict=y0_dict,  t_sim=t_sim)

# Simulation with new format (Model-based) - RECOMMENDED
# ====================================================
model_diffusion = exmod.generate_convection_model(
    Kaw_initial=7e-6, k_initial=1.38e-04, A_ini=A_ini, Dg_ini=Dg_ini,
     Vr=Vr, Vl=Vl, tMS=1)
df_result_model = dm.resolution_odeint_generic(model_diffusion, y0_dict=y0_dict,t_sim=t_sim)

# Verify both methods give same result
assert np.allclose(df_result["Cg"].values, df_result_model["Cg"].values), \
    "Old format and new Model format should give same results"
t_sim = np.linspace(0, 50, 100)
df_result = dm.resolution_odeint_generic(model_diffusion ,y0_dict, t_sim)
df_result2 = dm.resolution_odeint_generic(model_diffusion ,y0_dict={"Cl":2,"Cg":1.4e-5,"CMS":0}, t_sim=t_sim)
df_result3 = dm.resolution_odeint_generic(model_diffusion ,y0_dict={"Cl":2,"Cg":1.4e-5,"CMS":0}, t_sim=t_sim)

#   Plotting results
plt.figure(figsize=(8,5))
plt.plot(df_result["time"], df_result["Cg"], label="Cg (gazeuse)")
#plt.plot(df_result["time"], df_result["Cl"], label="Cl (liquide)")
plt.plot(df_result["time"], df_result["CMS"], label="CMS (stockage)")
plt.xlabel("Temps (s)")
plt.ylabel("Concentration")
plt.title("Simulation de diffusion avec compartiment CMS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Impact of parameters variation
#=====================================================================
model_diffusion_invitro_1=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_2=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=7.38e-03 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_3=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=7.38e-02 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=1)

model_diffusion_invitro_K1=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_K2=exmod.generate_convection_model(Kaw_initial=3e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_K3=exmod.generate_convection_model(Kaw_initial=3e-5, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=1)

model_diffusion_invitro_t1=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_t2=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=2)
model_diffusion_invitro_t3=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=3)

model_diffusion_invitro_D1=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=1e-6 ,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_D2=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=2.63e-6 ,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_D3=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=5e-6 ,Vr=Vr,Vl=Vl,tMS=1)

model_diffusion_invitro_A1=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini/2,Dg_ini=2.63e-6 ,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_A2=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=2.63e-6 ,Vr=Vr,Vl=Vl,tMS=1)
model_diffusion_invitro_A3=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini*2,Dg_ini=2.63e-6 ,Vr=Vr,Vl=Vl,tMS=1)

df_model_k1=dm.resolution_odeint_generic(model=model_diffusion_invitro_1,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_k2=dm.resolution_odeint_generic(model=model_diffusion_invitro_2,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_k3=dm.resolution_odeint_generic(model=model_diffusion_invitro_3,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_K1=dm.resolution_odeint_generic(model=model_diffusion_invitro_K1,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_K2=dm.resolution_odeint_generic(model=model_diffusion_invitro_K2,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_K3=dm.resolution_odeint_generic(model=model_diffusion_invitro_K3,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_t1=dm.resolution_odeint_generic(model=model_diffusion_invitro_t1,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_t2=dm.resolution_odeint_generic(model=model_diffusion_invitro_t2,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_t3=dm.resolution_odeint_generic(model=model_diffusion_invitro_t3,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_D1=dm.resolution_odeint_generic(model=model_diffusion_invitro_D1,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_D2=dm.resolution_odeint_generic(model=model_diffusion_invitro_D2,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_D3=dm.resolution_odeint_generic(model=model_diffusion_invitro_D3,y0_dict=y0_dict,  t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_A1=dm.resolution_odeint_generic(model=model_diffusion_invitro_A1,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_A2=dm.resolution_odeint_generic(model=model_diffusion_invitro_A2,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_A3=dm.resolution_odeint_generic(model=model_diffusion_invitro_A3,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

p1=gmet.plot_xy(df_model_k1['time'], y={"y_k_1.38e-04":df_model_k1['CMS'],"y_k_7.38e-03": df_model_k2['CMS'],"y_k_7.38e-02": df_model_k3['CMS']},y_type="l",title="Impact de k sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False)
p2=gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS'],"y_k_3e-6": df_model_K2['CMS'],"y_k_3e-5": df_model_K3['CMS']},y_type="l",title="Impact de K sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
# gmet.plot_xy(df_model_t1['time'], y={"y_t_1": df_model_t1['CMS'],"y_t_2": df_model_t2['CMS'],"y_t_3": df_model_t3['CMS']},y_type="l",title="Impact de t sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
# gmet.plot_xy(df_model_t1['time'], y={"y_D_1": df_model_D1['CMS'],"y_D_2": df_model_D2['CMS'],"y_D_3": df_model_D3['CMS']},y_type="l",title="Impact de D sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
# gmet.plot_xy(df_model_A1['time'], y={"y_A_1": df_model_A1['CMS'],"y_A_2": df_model_A2['CMS'],"y_A_3": df_model_A3['CMS']},y_type="l",title="Impact de A sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
# gmet.plot_xy(df_model_k1['time'], y={"y_k_1.38e-04":df_model_k1['CMS'],"y_k_7.38e-03": df_model_k2['CMS'],"y_k_7.38e-02": df_model_k3['CMS']},y_type="l",title="Impact de k sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1])
# gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'])

fig, axes = plt.subplots(2, 2, figsize=(12,8 ))
#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS'],"y_k_3e-6": df_model_K2['CMS'],"y_k_3e-5": df_model_K3['CMS']},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_model_k1['time'], y={"y_k_1.38e-04":df_model_k1['CMS'],"y_k_7.38e-03": df_model_k2['CMS'],"y_k_7.38e-02": df_model_k3['CMS']},y_type="l",title="b. k impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0])
gmet.plot_xy(df_model_t1['time'], y={"y_D_1": df_model_D1['CMS'],"y_D_2": df_model_D2['CMS'],"y_D_3": df_model_D3['CMS']},y_type="l",title="c. D impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,1])
gmet.plot_xy(df_model_t1['time'], y={"y_t_1": df_model_t1['CMS'],"y_t_2": df_model_t2['CMS'],"y_t_3": df_model_t3['CMS']},y_type="l",title="d. t_MS impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(7,7 ))
#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['Cl'],"y_k_3e-6": df_model_K2['Cl'],"y_k_3e-5": df_model_K3['Cl']},y_type="l",title="a. Concentration in liquid product",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['Cg'],"y_k_3e-6": df_model_K2['Cg'],"y_k_3e-5": df_model_K3['Cg']},y_type="l",title="b. Concentration in gazeous product",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,1])
gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS'],"y_k_3e-6": df_model_K2['CMS'],"y_k_3e-5": df_model_K3['CMS']},y_type="l",title="c. Concentration in PTRMS",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0])

plt.show()

#====================================================================
#  Testing resolution_odeint_generic with event steps  : run_single_stage
#=====================================================================
t_sim_1 = np.linspace(0, 50, 100)

# Defining the ODEs
def dCg_dt(y, t, Kaw, k, Vr, A, Dg):
    return ((k * A) / Vr) * (Kaw * y["Cl"] - y["Cg"]) - (Dg / Vr) * y["Cg"]

def dCl_dt(y, t, Kaw, k, Vl, A):
    return - ((k * A) / Vl) * (Kaw * y["Cl"] - y["Cg"])

def dCMS_dt(y, t, tMS):
    return (y["Cg"] - y["CMS"]) / tMS

#   creating dictionnaries


Kaw_initial=7e-6  # 7.10-3 mL/L
k_initial=1.38e-04  # m/s
Cl_ini=20 #(g/m3) car  2.0e-2 g/L
d = 3.7e-2  # m
A_ini = np.pi * (d / 2) ** 2  # (m^2)
Dg_ini= 2.63e-6   # (m^3/s)
Vr=28e-6  # (m^3)
Vl=2e-6 # (m^3)
params_dict1= {
    "Cg": {"Kaw": Kaw_initial, "k": k_initial , "Vr": Vr, "A": A_ini, "Dg":Dg_ini},
    "Cl": {"Kaw": Kaw_initial, "k":k_initial , "Vl": Vl, "A": A_ini},
    "CMS": {"tMS": 0.9}
}
funcs_dict1 = {
    "Cg": dCg_dt,
    "Cl": dCl_dt,
    "CMS": dCMS_dt
}
model_1=exmod.Model( params_dict, funcs_dict, name="diff_model")
y0_dict_1 = {"Cg": Kaw_initial * Cl_ini, "Cl": Cl_ini, "CMS": 0.0}


# Test without event
result=dm.run_model(model_1,y0_dict_1, t_start=0, t_end=200,df_event=None)
gmet.plot_xy(result["time"],result["Cg"])

# # Test with event
df_event = pd.DataFrame({
    "time": [25, 50, 75],
    "type": ["swallow", "swallow", "swallow"],
    "value": [ 0.1, 0.1,0.1],
    "details": [["Cg", "Cl"], ["Cg", "Cl"], ["Cg", "Cl"]]
})
# Defining event function (here swallowing event)
def swallow_step(y0_current,params_current=None):
    """
    Renvoie les dictionnaires pour la résolution d'une étape
    y_current : dict des conditions initiales
    """
    return {
        "Cg": 0 ,   # 50% absorbé → 50% restant
        "Cl": y0_current["Cl"] ,
        "CMS": y0_current["CMS"]         # inchangé
    }

def swallow_step_2(y0_current,params_current=None):
    """
    Renvoie les dictionnaires pour la résolution d'une étape
    y_current : dict des conditions initiales
    """
    return {
        "Cl": y0_current["Cl"] ,
        "Cg": 0 ,   # 50% absorbé → 50% restant
        "CMS": y0_current["CMS"]         # inchangé
    }

result=dm.run_model(model_1,y0_dict_1, t_start=0, t_end=200, df_event=df_event,  event_func_dict={"swallow": swallow_step})
#result=run_single_stage(y0_dict_1, funcs_dict1, params_dict1, t_start=0, t_end=200, df_event=df_event,  event_func_dict={"swallow": swallow_step})
gmet.plot_xy(result["time"],result["CMS"])
gmet.plot_xy(result["time"],result["Cg"])



result2=dm.run_model(model_1,y0_dict_1, t_start=0, t_end=200, df_event=df_event,  event_func_dict={"swallow": swallow_step_2})
#result=run_single_stage(y0_dict_1, funcs_dict1, params_dict1, t_start=0, t_end=200, df_event=df_event,  event_func_dict={"swallow": swallow_step})
gmet.plot_xy(result2["time"],result2["Cg"])

#===========run_multiple_stage ==========
# Defining the ODEs
def dCg_dt(y, t, Kaw, k, Vr, A, Dg):
    return ((k * A) / Vr) * (Kaw * y["Cl"] - y["Cg"]) - (Dg / Vr) * y["Cg"]

def dCl_dt(y, t, Kaw, k, Vl, A):
    return - ((k * A) / Vl) * (Kaw * y["Cl"] - y["Cg"])

def dCMS_dt(y, t, tMS):
    return (y["Cg"] - y["CMS"]) / tMS

#   creating dictionnaries
params_dict2= {
    "Cg": {"Kaw": Kaw_initial, "k": k_initial*200 , "Vr": Vr, "A": A_ini, "Dg":Dg_ini},
    "Cl": {"Kaw": Kaw_initial, "k":k_initial*200 , "Vl": Vl, "A": A_ini},
    "CMS": {"tMS": 0.9}
}
model_2=exmod.Model( params_dict=params_dict2, funcs_dict=funcs_dict1, name="diff_model_2")
stage1 = {
    "t_start": 0,
    "t_end": 10,
    "model": model_1,
}
stage2 = {
    "t_start": 10,
    "t_end": 200,
    "model":model_2,
}
stage3 = {
    "t_start": 210,
    "t_end": 300,
    "model":model_1,
}

# Test without event
stages = [stage1, stage2]
y0_initial = {"Cg": 10.0, "Cl": 5.0,"CMS": 0.0}
res=dm.run_stages(stages=stages, y0_dict= y0_initial, event_func_dict={"swallow": swallow_step})
gmet.plot_xy(res["time"],res["Cg"])
gmet.plot_xy(res["time"],res["CMS"])

# Tests with event
res=dm.run_stages(stages=stages, y0_dict=y0_initial, df_event=df_event,event_func_dict={"swallow": swallow_step},event_params=None)
gmet.plot_xy(res["time"],res["CMS"])

# Tests with only one stage
stages = [stage2]
y0_initial = {"Cg": 10.0, "Cl": 5.0,"CMS": 0.0}
res=dm.run_stages(stages=stages, y0_dict=y0_initial, event_func_dict={"swallow": swallow_step})
gmet.plot_xy(res["time"],res["Cg"])

res=dm.run_stages(stages, y0_dict=y0_initial, df_event=df_event,event_func_dict={"swallow": swallow_step})
gmet.plot_xy(res["time"],res["CMS"])

# test 3 stages not continuous
stages = [stage2,stage3]
res=dm.run_stages(stages, y0_dict=y0_initial, df_event=df_event,event_func_dict={"swallow": swallow_step})
gmet.plot_xy(res["time"],res["CMS"])

# Run with multiple stages and events works all the time ! 

#==========================
# Test with real models (existing_model)
# IN VITRO
#=============================
model_diffusion_invitro=exmod.generate_convection_model(Kaw_initial=7e-6, k_initial=1.38e-04 ,A_ini=A_ini,Dg_ini=2.63e-6 ,Vr=Vr,Vl=Vl,tMS=1)
# In vitro: testing the impact of swallowig events during diffusion
stages = [{
    "t_start": 0,
    "t_end": 10,
    "model": model_diffusion_invitro
    }]
res=dm.run_stages(stages, y0_dict=y0_initial, df_event=df_event,event_func_dict={"swallow": swallow_step})
gmet.plot_xy(res["time"],res["CMS"])


#====================================
# IN VIVO
# Solutions
#====================================
Kaw_initial=7e-6  # 7.10-3 mL/L
k_initial=1.38e-04  # m/s
QSaliva = 4e-2 # L/s
QBreath=200
AOAL = 100.0 # aire d'échange air/liquide en bouche
VOA = 37.0 # Air oral volume
VOL_m = 1.0 # volume de liquide résiduel après déglutition
AFAL = 60.0
VFA = 30.0
eFL = 1e-3 # épaisseur du film liquide dans le pharynx
VNA = 11.0
tMS = 1.03
VFL = VFA * eFL # Volume du liquide dans le pharynx: dilution
QNA_func = lambda t: QBreath*np.sin(t) # débit respiratoire (sinusoïde)
y0_dict = {
        "VOL": 2,   # volume de liquide en bouche
        "COA": 0,   # concentration dans l'air en bouche
        "COL": 2,   # concentration dans le liquide en bouche
        "CFA": 0,   # concentration dans l'air du pharynx
        "CFL": 0,   # concentration dans le liquide du pharynx
        "CNA": 0,   # concentration dans le nez
        "CMS": 0    # concentration mesurée (PTR-MS)
    }

res_in_vivo = exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func, QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA,
    tMS=tMS, KAL=Kaw_initial, kL=k_initial
)

def plot_in_vivo(df, title):
    fig, axes = plt.subplots(4, 2, figsize=(12, 8))
    panels = [("COL", "a. Liquid concentration in oral cavity (COL)"), ("CFA", "c. Concentration in pharynx (air, CFA)"),
              ("COA", "b. Air concentration in oral cavity (COA)"), ("CFL", "d. Concentration in pharynx (liquid, CFL)"),
              ("CNA", "e. Concentration in nasal cavity (CNA)"), ("CMS", "f. Concentration in PTR-MS (CMS)"),
              ("VOL", "h. Liquid volume in oral cavity (VOL)")]
    for ax, (var, panel_title) in zip(axes.flat, panels):
        gmet.plot_xy(df['time'], y={"y": df[var]}, y_type="l", title=panel_title, x_label="Temps (s)",
                     y_label=var, opt_type="l", new=False, ax=ax)
    gmet.plot_xy(df['time'], y={"y": QNA_func(df['time'])}, y_type="l", title="g. Breath", x_label="Temps (s)",
                 y_label="QNA", opt_type="l", new=False, ax=axes[3, 1])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# Without event (solution)
#=========================
t_sim = np.linspace(0, 50, 200)
df_in_vivo = dm.resolution_odeint_generic(model=res_in_vivo, y0_dict=y0_dict,
                                          t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
plot_in_vivo(df_in_vivo, "In vivo solution, without event")

# With deglutition events (solution)
#===================================
swallow_event = pd.DataFrame({
    "time": [10, 20, 30],
    "type": ["swallow", "swallow", "swallow"],
})
df_in_vivo = dm.run_model(model=res_in_vivo, y0_dict=y0_dict,
                          df_event=swallow_event,
                          event_func_dict=res_in_vivo.event_funcs,
                          event_params={"VFA": VFA, "VNA": VNA, "VOA": VOA, "VOL_m": VOL_m},
                          t_start=0, t_end=120, n_points=1000)
plot_in_vivo(df_in_vivo, "In vivo solution, with 3 swallows (10, 20, 30 s)")
