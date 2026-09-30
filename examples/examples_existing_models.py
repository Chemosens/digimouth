import sys
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from digimouth import optimization as opt
from digimouth import graphical_methods as gmet
from digimouth import existing_models as exmod
from digimouth import dynamic_model as dm
import matplotlib.pyplot as plt
#==============================================
# In vitro model (convection)
#==============================================
Kaw_initial=7e-6  # 7.10-3 mL/L
k_initial=1.38e-04  # m/s
Cl_ini=2 #(g/m3) car  2.0e-2 g/L (1ppm = 1mg/L, donc 2ppm = 2mg/L = 2e-3 g/L = 2 g/m3)
d = 3.7e-2  # m
A_ini = np.pi * (d / 2) ** 2  # (m^2)
Dg_ini= 2.63e-6   # (m^3/s)
Vr=28e-6  # (m^3)
Vl=2e-6 # (m^3)
tMS_initial=1
model_invitro=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial, A_ini=A_ini,
                                              Dg_ini=Dg_ini, Vr=Vr, Vl=Vl, tMS=tMS_initial)
df_invitro=dm.resolution_odeint_generic(model=model_invitro, y0_dict={"Cg": 0, "Cl": Cl_ini, "CMS": 0},
                                        t_sim=np.linspace(0, 600, 300))
gmet.plot_xy(df_invitro["time"], df_invitro["CMS"], title="In vitro (convection model): CMS")

#==============================================
# In vivo model (solution)
#==============================================
QSaliva = 4e-2
AOAL = 100.0
VOA = 37.0
AFAL = 60.0
VFA = 30.0
VFL = VFA * 1e-3 # Volume du liquide dans le pharynx: dilution
VNA = 11.0
QNA_func_solution = lambda t: np.sin(3*t)
model_solution = exmod.in_vivo_diffusion_solution_model(
    QNA_func=QNA_func_solution, QSaliva=QSaliva, AOAL=AOAL, VOA=VOA, AFAL=AFAL, VFA=VFA, VFL=VFL, VNA=VNA,
    tMS=tMS_initial, KAL=Kaw_initial, kL=k_initial
)
y0_solution = {"VOL": 2, "COA": 0, "COL": 2, "CFA": 0, "CFL": 0, "CNA": 0, "CMS": 0}
df_solution = dm.resolution_odeint_generic(model=model_solution, y0_dict=y0_solution, t_sim=np.linspace(0, 50, 200))
gmet.plot_xy(df_solution["time"], df_solution["CMS"], title="In vivo (solution model): CMS")

# =============

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

T_mouth=36
VOA = 37.0e-6*scale_factor**3 # Air oral volume
VFA = 30.0e-6*scale_factor**3
VNA = 11.0e-6*scale_factor**3
# Instrumental characteristics
tMS_initial=0.88 #(s)

# Product characteristics
Kaw_initial=KOAL=2.3E-02 
k_initial=1*(2.32e-04)*scale_factor # m/s
COP=0.4*1e+3*mass_factor/scale_factor**3 # Concentration du produit dans l'air oral
VOL_m=1e-6*scale_factor**3
# eFP = 0.1e-3*scale_factor
VFL=1e-6*scale_factor**3
# Dissolution parameters
kT=0.1 #thermal equilibration constant 1/20
alpha=0.5 # sensitivity of the dissolution velocity to temperature
v36=0.05e-3*scale_factor# speed of dissolution at 36 degrees
QSaliva = 0.025e-6*scale_factor**3# m3/s
QBreath=250e-6*scale_factor**3# m3/s

# Interfacial product/subject characteristics
AFAP = 60.0e-4*scale_factor**2
AOAP = 100.0e-4*scale_factor**2# aire
AOLP_coefficient=(np.pi*(0.8e-2*scale_factor)**2)/(VOP_ini**(2/3)) #=1/h =5

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

kT=0.1 #thermal equilibration constant 1/20
alpha=0.5 # sensitivity of the dissolution velocity to temperature
v36=0.05e-3*scale_factor# speed of dissolution at 36 degrees

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


y0_dict_with_temp={"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini,
                "CFA":CFA_ini, "CFL":CFL_ini, "CNA":CNA_ini,"CMS":CMS_ini,"TP":T_ini}
res_in_vivo_gel_chew_with_temp=exmod.in_vivo_diffusion_solid_model_with_temp(
tMS=tMS, kL=k_initial, KAL=Kaw_initial, QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, 
VFA=VFA, VNA=VNA, VFL=VFL, COP=COP, AOLP_coefficient=AOLP_coefficient,
QNA_func=iso_interp, kT=kT,T_mouth=T_mouth,v_mouth=v_mouth,alpha=alpha )  

df_in_vivo_gel_chew=dm.run_model(model=res_in_vivo_gel_chew_with_temp, y0_dict=y0_dict_with_temp,
                                df_event=swallow_event_chew, 
                                event_func_dict=res_in_vivo_gel_chew_with_temp.event_funcs,
                                event_params={"VFA": VFA,"VNA":VNA,"VOA":VOA,"VOL_m":VOL_m,"chew_multiplier":1},
                                t_start=0,t_end=120,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)


fig, axes = plt.subplots(5, 2, figsize=(12,8 ))
#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['TP']},y_type="l",title="a. Temperature of the product",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['VOP']},y_type="l",title="b. Volume oral product",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,1])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['VOL']},y_type="l",title="c. VOL",x_label="Time (s)",y_label="vol. salivaire oral",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y":df_in_vivo_gel_chew['COL']},y_type="l",title="d. Product concentration (liquid) in oral cavity",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
#gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": AOLP_func(df_in_vivo_gel_chew['time'])},y_type="l",title="d.AOLP_func ",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
#gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['AOLP']},y_type="l",title="d.AOLP_func ",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y":df_in_vivo_gel_chew['COA']},y_type="l",title="e. Air concentration in oral cavity (gaz)",x_label="Time (s)",y_label="Concentration",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,0])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CFA']},y_type="l",title="f. Concentration in pharynx (air, CFA)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,1])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CFL']},y_type="l",title="g. Concentration in pharynx (product, CFL)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,0])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CNA']},y_type="l",title="h. Concentration in nasal cavity (CNA)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,1])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['CMS']},y_type="l",title="i. Concentration in PTRMS (CMS)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[4,0])
gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": QNA_func(df_in_vivo_gel_chew['time'])},y_type="l",title="j. Breath",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[4,1])
plt.tight_layout()
plt.show()


# without AOLP
res_in_vivo_gel_chew_surface=exmod.in_vivo_diffusion_solid_model_with_temp_and_surface(
    #static parameters
    tMS=tMS,kL=k_initial, KAL=Kaw_initial,QSaliva=QSaliva, AOAL=AOAP, VOA=VOA, AFAL=AFAP, VFA=VFA, VNA=VNA, VFL=VFL, COP=COP,
    # Fonctions dynamiques
    QNA_func=QNA_func , kT=kT,T_mouth=36,v_mouth=v_chew,alpha= alpha)
y0_init_gel_surface = {"VOP":VOP_ini, "VOL":VOL_ini, "COL":COL_ini, "COA":COA_ini, "CFA":CFA_ini, "CNA":CNA_ini,"CMS":CMS_ini, "CFL":CFL_ini,"TP":T_ini,"SP":1}

df_in_vivo_gel_surface=dm.run_model(model=res_in_vivo_gel_chew_surface, y0_dict=y0_init_gel_surface,
                                df_event=swallow_event_chew, 
                                event_func_dict=res_in_vivo_gel_chew_surface.event_funcs,
                                event_params={"VFA": VFA,"VNA":VNA,"VOA":VOA,"VOL_m":VOL_m,"chew_multiplier":1,"chew_factor":1.5},
                                t_start=0,t_end=120,n_points=1000, rtol=1e-4, atol=1e-9, mxstep=5000)



fig, axes = plt.subplots(5, 2, figsize=(12,8 ))
#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": df_in_vivo_gel_surface['TP']},y_type="l",title="a. Temperature of the product",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": df_in_vivo_gel_surface['VOP']},y_type="l",title="b. Volume oral product",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,1])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": df_in_vivo_gel_surface['VOL']},y_type="l",title="c. VOL",x_label="Time (s)",y_label="vol. salivaire oral",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y":df_in_vivo_gel_surface['COL']},y_type="l",title="d. Product concentration (liquid) in oral cavity",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y":df_in_vivo_gel_surface['SP']},y_type="l",title="j. Shape factor",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[4,1])


#gmet.plot_xy(df_in_vivo_gel_chew['time'], y={"y": df_in_vivo_gel_chew['AOLP']},y_type="l",title="d.AOLP_func ",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y":df_in_vivo_gel_surface['COA']},y_type="l",title="e. Air concentration in oral cavity (gaz)",x_label="Time (s)",y_label="Concentration",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,0])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": df_in_vivo_gel_surface['CFA']},y_type="l",title="f. Concentration in pharynx (air, CFA)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,1])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": df_in_vivo_gel_surface['CFL']},y_type="l",title="g. Concentration in pharynx (product, CFL)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,0])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": df_in_vivo_gel_surface['CNA']},y_type="l",title="h. Concentration in nasal cavity (CNA)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[3,1])
gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": df_in_vivo_gel_surface['CMS']},y_type="l",title="i. Concentration in PTRMS (CMS)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[4,0])
#gmet.plot_xy(df_in_vivo_gel_surface['time'], y={"y": QNA_func(df_in_vivo_gel_surface['time'])},y_type="l",title="j. Breath",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[4,1])
plt.tight_layout()
plt.show()