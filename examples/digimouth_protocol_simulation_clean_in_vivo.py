import sys
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import sympy as sp
from digimouth import dynamic_model as dm
from digimouth import existing_models as exmod
from digimouth import graphical_methods as gmet
import matplotlib.pyplot as plt

#======================================
# In vitro protocols

#==============================
# In vivo (generics)
#==============================
# Initial conditions

t_sim = np.linspace(0, 100, 100)
scale_factor=1e+2
mass_factor=1e+3

VOL_ini=2e-6*scale_factor**3
COL_ini=0.4*1e+3*mass_factor/scale_factor**3 # Concentration du produit dans l'air oral
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

Kaw_initial=2.3E-02 
k_initial=(2.32e-04)*scale_factor # m/s

AOAP = 100.0e-4*scale_factor**2# aire
AFAP = 30.0e-4*scale_factor**2

#VFP=AFAP * eFP # Volume du produit dans le pharynx: dilution
VFL = 1e-6*scale_factor**3
VOL_m=1e-6*scale_factor**3

QNA_func= lambda t: QBreath*np.sin(2*np.pi*((t+5)/5))

#=================================
# In vivo protocols (solution)
#=================================
# Protocol 0: Easy protocol
swallow_event_rare = pd.DataFrame({
    "time": [60,80, 20,40],
    "type": [
             "swallow","swallow","jaw_move","jaw_move"]
})

res_in_vivo_sol_rare= exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, VFA=VFA,  VFL=VFL, VNA=VNA,
     tMS=tMS_initial, KAL=Kaw_initial, kL=k_initial
)
y0_dict_sol={"VOL":VOL_ini, "COA":COA_ini, "COL":COL_ini, "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini, "CMS":CMS_ini}
df_in_vivo_sol_rare=dm.run_model(model=res_in_vivo_sol_rare,
                                  y0_dict=y0_dict_sol,
                                  df_event=swallow_event_rare, 
                                  event_func_dict=res_in_vivo_sol_rare["event_funcs"],
                                  event_params={"VFA": VFA,"VNA":VNA,"VOA":VOA,"VOL_m":VOL_m},
                                  t_start=0,t_end=100,n_points=1000,
                                  rtol=1e-4, atol=1e-9, mxstep=5000)

fig, axes = plt.subplots(4, 2, figsize=(12,8 ))

#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y": df_in_vivo_sol_rare['VOL']},y_type="l",title="a. Liquid volume in the oral cavity ($V_{OL}$)",x_label="Time (s)",y_label="Volume (mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0],y_lim=(0,5),legend=False)
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y":df_in_vivo_sol_rare['COL']},y_type="l",title="b. Aroma concentration in the liquid in oral cavity ($C_{OL}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,1],legend=False)
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y":df_in_vivo_sol_rare['COA']},y_type="l",title="c. Aroma concentration in the air in oral cavity ($C_{OA}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0],legend=False)
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y": df_in_vivo_sol_rare['CFA']},y_type="l",title="d. Aroma concentration in the air in pharynx cavity ($ C_{FA}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1],legend=False)
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y": df_in_vivo_sol_rare['CFL']},y_type="l",title="e. Aroma concentration in the liquid in pharynx cavity ($C_{FL}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,0],legend=False)
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y": df_in_vivo_sol_rare['CNA']},y_type="l",title="f. Aroma concentration in nasal cavity ($C_{NA}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,1],legend=False)
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y": df_in_vivo_sol_rare['CMS']},y_type="l",title="g. Intensity measured by PTR-MS ($C_{MS}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,0],legend=False)
gmet.plot_xy(df_in_vivo_sol_rare['time'], y={"y": QNA_func(df_in_vivo_sol_rare['time'])},y_type="l",title="h. Breath flowrate ($Q_{NA}$)",x_label="Time (s)",y_label="Flowrate (mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,1],legend=False)

event_lines = [
    (20, "forestgreen"),
    (40, "forestgreen"),
    (60, "tomato"),
    (80, "tomato")
]
for ax in axes.flat:
    for t, c in event_lines:
        ax.axvline(x=t, color=c, linestyle="--", alpha=0.8)

plt.tight_layout()
plt.show()

df_window = df_in_vivo_sol_rare[
    (df_in_vivo_sol_rare["time"] >= 59.5) &
    (df_in_vivo_sol_rare["time"] <= 60.5)
]
df_window

# Testing breath
#=====================================
QBreath=1
QNA_func3 = lambda t: 0.1*QBreath*np.sin(2*np.pi*((t+5)/5))
QNA_func4= lambda t: QBreath*np.sin(2*np.pi*((t+5)/5))
QNA_func= lambda t: 10*QBreath*np.sin(2*np.pi*((t+5)/5))
QNA_func1= lambda t: 50*QBreath*np.sin(2*np.pi*((t+5)/5))
QNA_func2= lambda t: 100*QBreath*np.sin(2*np.pi*((t+5)/5))

res_in_vivo_sol_test= exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial, KAL=Kaw_initial, kL=k_initial
)
res_in_vivo_sol_test1= exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func1, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial, KAL=Kaw_initial, kL=k_initial
)
res_in_vivo_sol_test2= exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func2, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, VFA=VFA,  VFL=VFL, VNA=VNA, tMS=tMS_initial, KAL=Kaw_initial, kL=k_initial
)
res_in_vivo_sol_test3= exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func3, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, VFA=VFA,VFL=VFL, VNA=VNA, tMS=tMS_initial, KAL=Kaw_initial, kL=k_initial
)
res_in_vivo_sol_test4= exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func4, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, VFA=VFA, VFL=VFL, VNA=VNA, tMS=tMS_initial, KAL=Kaw_initial, kL=k_initial
)
y0_init = {"VOL":VOL_ini, "COA":COA_ini, "COL":COL_ini, "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini, "CMS":CMS_ini}
df_in_vivo_sol_test=dm.run_model(model=res_in_vivo_sol_test, y0_dict=y0_init, df_event=swallow_event_rare, 
                                event_func_dict=res_in_vivo_sol_test["event_funcs"],
                                event_params={"VFA": VFA,"VNA":VNA,"VOL_m":VOL_m,"VOA":VOA},
                                t_start=0,t_end=100,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)
df_in_vivo_sol_test1=dm.run_model(model=res_in_vivo_sol_test1, y0_dict=y0_init, df_event=swallow_event_rare, 
                                event_func_dict=res_in_vivo_sol_test1["event_funcs"],
                                event_params={"VFA": VFA,"VNA":VNA,"VOL_m":VOL_m,"VOA":VOA},
                                t_start=0,t_end=100,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)
df_in_vivo_sol_test2=dm.run_model(model=res_in_vivo_sol_test2, y0_dict=y0_init, df_event=swallow_event_rare, 
                                event_func_dict=res_in_vivo_sol_test2["event_funcs"],
                                event_params={"VFA": VFA,"VNA":VNA,"VOL_m":VOL_m,"VOA":VOA},
                                t_start=0,t_end=100,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)
df_in_vivo_sol_test3=dm.run_model(model=res_in_vivo_sol_test3, y0_dict=y0_init, df_event=swallow_event_rare, 
                                event_func_dict=res_in_vivo_sol_test3["event_funcs"],
                                event_params={"VFA": VFA,"VNA":VNA,"VOL_m":VOL_m,"VOA":VOA},
                                t_start=0,t_end=100,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)
df_in_vivo_sol_test4=dm.run_model(model=res_in_vivo_sol_test4, y0_dict=y0_init, df_event=swallow_event_rare, 
                                event_func_dict=res_in_vivo_sol_test4["event_funcs"],
                                event_params={"VFA": VFA,"VNA":VNA,"VOL_m":VOL_m,"VOA":VOA},
                                t_start=0,t_end=100,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)
fig, axes = plt.subplots(3, 2, figsize=(12,8 ))
gmet.plot_xy(df_in_vivo_sol_test['time'], y={"y": df_in_vivo_sol_test['CMS']},y_type="l",title="Aroma release (QBreath=10)",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0])
gmet.plot_xy(df_in_vivo_sol_test2['time'], y={"y": df_in_vivo_sol_test2['CMS']},y_type="l",title="Aroma release (QBreath=100)",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,0])
gmet.plot_xy(df_in_vivo_sol_test3['time'], y={"y": df_in_vivo_sol_test3['CMS']},y_type="l",title="Aroma release (QBreath=0.1)",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_in_vivo_sol_test4['time'], y={"y": df_in_vivo_sol_test4['CMS']},y_type="l",title="Aroma release (QBreath=1)",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,1])
gmet.plot_xy(df_in_vivo_sol_test1['time'], y={"y": df_in_vivo_sol_test1['CMS']},y_type="l",title="Aroma release (QBreath=50)",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
gmet.plot_xy(df_in_vivo_sol_test1['time'], y={"y": QNA_func4(df_in_vivo_sol_test1['time'])},y_type="l",title="Breath",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,1])
plt.tight_layout()
plt.show()

#=================================
# In vivo protocols (gel)
#=================================
# Initial conditions
iso_interp= lambda t: QBreath*np.sin(2*np.pi*((t+5)/5))

scale_factor=1e+2
mass_factor=1e+3
T_ini=4
VOP_ini=2e-6*scale_factor**3
VOL_ini=1e-6*scale_factor**3
COA_ini=0
COL_ini=0
CFA_ini=0 # Concentration de l'air dans le pharynx
CFL_ini=0
CFP_ini=0 # Concentration du produit dans le pharynx
CNA_ini=0 # Concentration dans le nez
CMS_ini=0
ray_product_ini=8e-3*scale_factor# dans ce cas h=0.63
SP_ini=(np.pi*(ray_product_ini)**2)/(VOP_ini**(2/3))

T_mouth=36
VOA = 37.0e-6*scale_factor**3 # Air oral volume
VFA = 30.0e-6*scale_factor**3
VNA = 11.0e-6*scale_factor**3
# Instrumental characteristics
tMS_initial=0.88 #(s)

# Product characteristics
Kaw_initial=KOAL=2.3E-02 
k_initial=1*(2.32e-04)*scale_factor # m/s
COP=0.4*mass_factor/scale_factor**3 # Concentration du produit dans l'air oral
VOL_m=1e-6*scale_factor**3
# eFP = 0.1e-3*scale_factor
VFL=1e-6*scale_factor**3
# Dissolution parameters
kT=0.1 #thermal equilibration constant 1/20
alpha=0.5 # sensitivity of the dissolution velocity to temperature
v36=0.15e-3*scale_factor# speed of dissolution at 36 degrees
QSaliva = 0.025e-6*scale_factor**3# m3/s
QBreath=250e-6*scale_factor**3# m3/s

# Interfacial product/subject characteristics
AFAP = 60.0e-4*scale_factor**2
AOAP = 100.0e-4*scale_factor**2# aire
AOLP_coefficient=(np.pi*(0.8e-2*scale_factor)**2)/(VOP_ini**(2/3)) #=1/h =5
chew_factor=1.25

v_mouth=v36
v_chew=v36 #
v_succ=v36*2
v=v36*2

aolp_in_model=False
temp_in_model=True
fop="chew"
subject="Q407"
rep=1
QNA_func=iso_interp

tMS=tMS_initial
SP_ini=(np.pi*(0.8e-2*scale_factor)**2)/(VOP_ini**(2/3))
                                        
swallow_event_chew = pd.DataFrame({
    "time": [40, 60,80],
    "type": ["chew",
             "swallow", "total_swallow"],
})

# without AOLP
res_in_vivo_gel_chew=exmod.in_vivo_diffusion_solid_model(
    #static parameters
    tMS=tMS,kOL=k_initial, KOAL=Kaw_initial, v=v,QSaliva=QSaliva, AOAP=AOAP, VOA=VOA, AFAP=AFAP, VFA=VFA, VNA=VNA, VFL=VFL, COP=COP,AOLP_coefficient=AOLP_coefficient,
    # Fonctions dynamiques
    QNA_func=QNA_func )
y0_init_gel = {"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini, "CFA":CFA_ini, "CNA":CNA_ini,"CMS":CMS_ini, "CFL":CFL_ini}
# with AOLP
res_in_vivo_gel_chew=exmod.in_vivo_diffusion_solid_model(
  #  v=0.02,# Vitesse de diminution de l'aire
    #static parameters
    tMS=tMS,kOL=k_initial, KOAL=Kaw_initial, v=v,QSaliva=QSaliva, AOAP=AOAP, VOA=VOA, AFAP=AFAP, VFA=VFA,
      VNA=VNA, VFL=VFL, COP=COP,
      AOLP_coefficient=2,
      #VOL_m=VOL_m,
    # Fonctions dynamiques
     #AOLP_func=AOLP_func,
       QNA_func=QNA_func )

# Considering the surface area of the product
#==========================================
y0_dict_with_temp={"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini,
                "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini,"CMS":CMS_ini,"TP":T_ini,"SP":SP_ini}
res_in_vivo_gel_chew_with_temp=exmod.in_vivo_diffusion_solid_model_with_temp_and_surface(
tMS=tMS, kL=k_initial, KAL=Kaw_initial, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, 
VFA=VFA, VNA=VNA, VFL=VFL, COP=COP, 
QNA_func=iso_interp, kT=kT,T_mouth=T_mouth,v_mouth=v_mouth,alpha=alpha )  


df_in_vivo_gel_chew=dm.run_model(model=res_in_vivo_gel_chew_with_temp, y0_dict=y0_dict_with_temp,
                                df_event=swallow_event_chew, 
                                event_func_dict=res_in_vivo_gel_chew_with_temp.event_funcs,
                                event_params={"VFA": VFA,"VNA":VNA,"VOA":VOA,"VOL_m":VOL_m,"chew_factor":1.2},
                                t_start=0,t_end=120,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)


fig, axes = plt.subplots(5, 2, figsize=(12,8 ))
#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['SP']},y_type="l",title="a. Shape factor",x_label="Time (s)",y_label="No unity",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['TP']},y_type="l",title="b. Temperature of the product ($T_P$)",x_label="Time (s)",y_label="Temperature (°C)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,1],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['VOP']},y_type="l",title="c. Solid volume in oral cavity ($V_{OP}$)",x_label="Time (s)",y_label="Volume (mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['VOL']},y_type="l",title="d. Liquid volume in oral cavity ($V_{OL}$)",x_label="Time (s)",y_label="Volume (mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y":df_in_vivo_gel_chew['COL']},y_type="l",title="e. Aroma concentration in the liquid in oral cavity ($C_{OL}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,0],legend=False)
#gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": AOLP_func(df_in_vivo_gel_chew['time'])},y_type="l",title="d.AOLP_func ",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
#gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['AOLP']},y_type="l",title="d.AOLP_func ",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y":df_in_vivo_gel_chew['COA']},y_type="l",title="f. Aroma concentration in the air in oral cavity ($C_{OA}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,1],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CFA']},y_type="l",title="g. Aroma concentration in the air in pharynx cavity ($C_{FA}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,0],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CFL']},y_type="l",title="h. Aroma concentration in the liquid in pharynx cavity ($C_{FL}$)",x_label="Time (s)",y_label="Conc. (mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,1],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CNA']},y_type="l",title="i. Aroma concentration in nasal cavity ($C_{NA}$)",x_label="Time (s)",y_label="Conc.(mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[4,0],legend=False)
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CMS']},y_type="l",title="j. Intensity measured by PTRMS ($C_{MS}$)",x_label="Time (s)",y_label="Conc.(mg/mL)",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[4,1],legend=False)
event_lines = [
    (40, "forestgreen"),
    (60, "orange"),
    (80, "tomato"),
]
for ax in axes.flat:
    for t, c in event_lines:
        ax.axvline(x=t, color=c, linestyle="--", alpha=0.8)

plt.tight_layout()
plt.show()
plt.tight_layout()
plt.show()