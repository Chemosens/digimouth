
import numpy as np
import pandas as pd
from digimouth import dynamic_model as dm
from digimouth import existing_models as exmod
from digimouth import graphical_methods as gmet
import matplotlib.pyplot as plt


#======================================
# In vitro protocols
#======================================
Kaw_initial=7e-3  # 7.10-3 mL/L
k_initial=1.38e-02  # cm/s
Cl_ini=2 #(g/cm3) car  2.0e-2 g/L (1ppm = 1mg/L, donc 2ppm = 2mg/L = 2e-3 g/L = 2 g/m3)
d = 3.7 # cm
A_ini = np.pi * (d / 2) ** 2  # (cm^2)
Dg_ini= 2.63   # (cm^3/s)
Vr=28  # (cm^3)
Vl=2 # (cm^3)
tMS_initial=1
t_sim = np.linspace(0, 100, 100)
y0_dict={"Cl": Cl_ini, "Cg": Kaw_initial*Cl_ini, "CMS": 0}
y0_dict_K1={"Cl": Cl_ini, "Cg": Kaw_initial/2*Cl_ini, "CMS": 0}
y0_dict_K3={"Cl": Cl_ini, "Cg": Kaw_initial*2*Cl_ini, "CMS": 0}
model_diffusion_invitro_original=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
df_model_original=dm.resolution_odeint_generic(model_diffusion_invitro_original, y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
fig, axes = plt.subplots(1, 3, figsize=(12,6 ))
#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_model_original['time'], y=df_model_original['Cl'],y_type="l",title="a. Concentration of aroma in solution",x_label="Time (s)",y_label="Concentration",new=False,ax=axes[0],y_lim=(0,2))
gmet.plot_xy(df_model_original['time'], y=df_model_original['Cg'],y_type="l",title="b. Concentration of aroma in headspace",x_label="Time (s)",y_label="Concentration",new=False,ax=axes[1],y_lim=(0,0.0145))
gmet.plot_xy(df_model_original['time'], y=df_model_original['CMS'],y_type="l",title="c. Concentration of aroma in PTR-MS",x_label="Time (s)",y_label="Concentration",new=False,ax=axes[2],y_lim=(0,0.0145))
plt.tight_layout()
plt.show()

# Impact of parameters variation
#=====================================================================
model_diffusion_invitro_k1=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial/2 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_k2=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_k3=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial*2 ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_K1=exmod.generate_convection_model(Kaw_initial=Kaw_initial/2, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_K2=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_K3=exmod.generate_convection_model(Kaw_initial=Kaw_initial*2, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_t1=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial/2)
model_diffusion_invitro_t2=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_t3=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial*2)

model_diffusion_invitro_D1=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini/2 ,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_D2=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_D3=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini*2,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_A1=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini/2,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_A2=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_A3=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini*2,Dg_ini=Dg_ini,Vr=Vr,Vl=Vl,tMS=tMS_initial)

df_model_k1=dm.resolution_odeint_generic(model_diffusion_invitro_k1,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_k2=dm.resolution_odeint_generic(model_diffusion_invitro_k2, y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_k3=dm.resolution_odeint_generic(model_diffusion_invitro_k3, y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_K1=dm.resolution_odeint_generic(model_diffusion_invitro_K1, y0_dict=y0_dict_K1, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_K2=dm.resolution_odeint_generic(model_diffusion_invitro_K2, y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_K3=dm.resolution_odeint_generic(model_diffusion_invitro_K3, y0_dict=y0_dict_K3, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_t1=dm.resolution_odeint_generic(model_diffusion_invitro_t1,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_t2=dm.resolution_odeint_generic(model_diffusion_invitro_t2,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_t3=dm.resolution_odeint_generic(model_diffusion_invitro_t3,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_D1=dm.resolution_odeint_generic(model_diffusion_invitro_D1,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_D2=dm.resolution_odeint_generic(model_diffusion_invitro_D2,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_D3=dm.resolution_odeint_generic(model_diffusion_invitro_D3,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

df_model_A1=dm.resolution_odeint_generic(model_diffusion_invitro_A1,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_A2=dm.resolution_odeint_generic(model_diffusion_invitro_A2,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)
df_model_A3=dm.resolution_odeint_generic(model_diffusion_invitro_A3,y0_dict=y0_dict, t_sim=t_sim, rtol=1e-6, atol=1e-12, mxstep=0)

p1=gmet.plot_xy(df_model_k1['time'], y={"y_k_1.38e-04":df_model_k1['CMS'],"y_k_7.38e-03": df_model_k2['CMS'],"y_k_7.38e-02": df_model_k3['CMS']},y_type="l",title="Impact de k sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False)
p2=gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS'],"y_k_3e-6": df_model_K2['CMS'],"y_k_3e-5": df_model_K3['CMS']},y_type="l",title="Impact de K sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
gmet.plot_xy(df_model_t1['time'], y={"y_t_1": df_model_t1['CMS'],"y_t_2": df_model_t2['CMS'],"y_t_3": df_model_t3['CMS']},y_type="l",title="Impact de t sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
gmet.plot_xy(df_model_t1['time'], y={"y_D_1": df_model_D1['CMS'],"y_D_2": df_model_D2['CMS'],"y_D_3": df_model_D3['CMS']},y_type="l",title="Impact de D sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
gmet.plot_xy(df_model_A1['time'], y={"y_A_1": df_model_A1['CMS'],"y_A_2": df_model_A2['CMS'],"y_A_3": df_model_A3['CMS']},y_type="l",title="Impact de A sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_type="l",y_col=['blue','red','green'])
gmet.plot_xy(df_model_k1['time'], y={"y_k_1.38e-04":df_model_k1['CMS'],"y_k_7.38e-03": df_model_k2['CMS'],"y_k_7.38e-02": df_model_k3['CMS']},y_type="l",title="Impact de k sur la libération in vitro",x_label="Temps (s)",y_label="Concentration mesurée (a.u.)",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1])
gmet.plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'])

fig, axes = plt.subplots(3, 2, figsize=(12,8 ))
#plot_xy(df_model_K1['time'], y={"y_K_7e-6":df_model_K1['CMS']/df_model_K1['CMS'].max(),"y_k_3e-6": df_model_K2['CMS']/df_model_K2['CMS'].max(),"y_k_3e-5": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="a. K_aw impact on aroma release",x_label="Temps (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_model_K1['time'], y={"$0.5*K_{aw}$":df_model_K1['CMS'],"$K_{aw}$": df_model_K2['CMS'],"$2*K_{aw}$": df_model_K3['CMS']},y_type="l",title="a. $K_{aw}$ impact on aroma release ($C_{MS}$)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[0,0])
gmet.plot_xy(df_model_K1['time'], y={"$0.5*K_{aw}$":df_model_K1['CMS']/df_model_K1['CMS'].max(),"$K_{aw}$": df_model_K2['CMS']/df_model_K2['CMS'].max(),"$2*K_{aw}$": df_model_K3['CMS']/df_model_K3['CMS'].max()},y_type="l",title="b. $K_{aw}$ impact on normalized aroma release ($C_{MS}$)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False, ax=axes[0,1])
gmet.plot_xy(df_model_A1['time'], y={"$0.5*A$": df_model_A1['CMS'],"$A$": df_model_A2['CMS'],"$2*A$": df_model_A3['CMS']},y_type="l",title="c. $A$ impact on aroma release ($C_{MS}$)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,0])
gmet.plot_xy(df_model_k1['time'], y={"$0.5k$":df_model_k1['CMS'],"$k$": df_model_k2['CMS'],"$2*k$": df_model_k3['CMS']},y_type="l",title="d. $k$ impact on aroma release ($C_{MS}$)",x_label="Time (s)",y_label="Concentration",opt_label="k=7.38e-3 m/s",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[1,1])
gmet.plot_xy(df_model_t1['time'], y={"$0.5*D_g$": df_model_D1['CMS'],"$D_g$": df_model_D2['CMS'],"$2*D_g$": df_model_D3['CMS']},y_type="l",title="e. $D_g$ impact on aroma release ($C_{MS}$)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,0])
gmet.plot_xy(df_model_t1['time'], y={"$0.5*t_{MS}$": df_model_t1['CMS'],"$t_{MS}$": df_model_t2['CMS'],"$2*t_{MS}$": df_model_t3['CMS']},y_type="l",title="f. $t_{MS}$ impact on aroma release ($C_{MS}$)",x_label="Time (s)",y_label="Concentration",opt_type="l",y_col=['blue','red','green'],new=False,ax=axes[2,1])
plt.tight_layout()
plt.show()

#======================================
# In vitro protocols with swallowing events and stages
#=====================================
df_event = pd.DataFrame({
    "time": [ 100],
    "type": ["open"],
    "value": [ 0.1],
    "details": [["Cg", "Cl"]]
})

#df_event=None
model_diffusion_invitro_k1=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Dg_ini=Dg_ini ,Vr=Vr,Vl=Vl,tMS=tMS_initial)
model_diffusion_invitro_k100=exmod.generate_convection_model(Kaw_initial=Kaw_initial, k_initial=k_initial*10 ,A_ini=A_ini,Dg_ini=Dg_ini ,Vr=Vr,Vl=Vl,tMS=tMS_initial)

# In vitro: testing the impact of swallowig events during diffusion
stages = [{ "t_start": 0,   "t_end": 50,   "model": model_diffusion_invitro_k1},
          { "t_start": 50,  "t_end": 150,  "model": model_diffusion_invitro_k100}]

stages_2 = [{ "t_start": 0,   "t_end": 30,   "model": model_diffusion_invitro_k100},
          { "t_start": 30,  "t_end": 60,  "model": model_diffusion_invitro_k1}]

res=dm.run_stages(stages, y0_dict=y0_dict, df_event=df_event,event_func_dict= model_diffusion_invitro_k1.event_funcs, event_params=None)
#gmet.plot_xy(res["time"],res["CMS"])
gmet.plot_xy(res["time"],res["Cg"],vertical_lines=[50,100],title="Impact of events and stages on aroma release on $C_g$",x_label="Time (s)",y_label="Concentration (g/m3)")
#gmet.plot_xy(res["time"],res["CMS"],vertical_lines=[10,15,20],title="Impact of events and stages on aroma release on $C_{MS}$",x_label="Time (s)",y_label="Concentration (g/m3)")
#res_2=dm.run_stages(stages_2, y0_dict=y0_dict, df_event=df_event,event_func_dict= model_diffusion_invitro_k1.event_funcs, event_params=None)
#gmet.plot_xy(res_2["time"],res_2["Cg"],vertical_lines=[10,15,20],title="Impact of events and stages on aroma release on $C_g$",x_label="Time (s)",y_label="Concentration (g/m3)")
#gmet.plot_xy(res_2["time"],res_2["CMS"],vertical_lines=[10,15,20],title="Impact of events and stages on aroma release on $C_{MS}$",x_label="Time (s)",y_label="Concentration (g/m3)")


#=======================
model_diffusion_invitro_k1_no_ptr=exmod.generate_convection_model_without_measure(Kaw_initial=Kaw_initial, k_initial=k_initial ,A_ini=A_ini,Vr=Vr,Vl=Vl)
model_diffusion_invitro_k100_no_ptr=exmod.generate_convection_model_without_measure(Kaw_initial=Kaw_initial, k_initial=k_initial*100 ,A_ini=A_ini,Vr=Vr,Vl=Vl)
stages_no_ptr = [{ "t_start": 0,   "t_end": 20,   "model": model_diffusion_invitro_k1_no_ptr},
          { "t_start": 20,  "t_end": 50,  "model": model_diffusion_invitro_k100_no_ptr}]
stages_no_ptr_2 = [{ "t_start": 0,   "t_end": 20,   "model": model_diffusion_invitro_k100_no_ptr},
          { "t_start": 20,  "t_end": 50,  "model": model_diffusion_invitro_k1_no_ptr}]
stages_no_ptr_3 = [{ "t_start": 0,   "t_end": 20,   "model": model_diffusion_invitro_k1_no_ptr},
          { "t_start": 20,  "t_end": 50,  "model": model_diffusion_invitro_k1_no_ptr}]

y0_dict_no_ptr={"Cl": Cl_ini, "Cg": 0}
res=dm.run_stages(stages_no_ptr, y0_dict=y0_dict_no_ptr, df_event=df_event,event_func_dict= model_diffusion_invitro_k1_no_ptr.event_funcs, event_params=None)
res_2=dm.run_stages(stages_no_ptr_2, y0_dict=y0_dict_no_ptr, df_event=df_event,event_func_dict= model_diffusion_invitro_k1_no_ptr.event_funcs, event_params=None)
res_3=dm.run_stages(stages_no_ptr_3, y0_dict=y0_dict_no_ptr, df_event=df_event,event_func_dict= model_diffusion_invitro_k1_no_ptr.event_funcs, event_params=None)

fig, axes= plt.subplots(1,3)
gmet.plot_xy(res["time"],res["Cg"],new=False,ax=axes[0],title="a. k1-k100",x_label="Time (s)",y_label="Concentration", vertical_lines=[15,20,35])
gmet.plot_xy(res_2["time"],res_2["Cg"],new=False,ax=axes[1],title="b. k100-k1",x_label="Time (s)",y_label="Concentration", vertical_lines=[15,20,35])
gmet.plot_xy(res_3["time"],res_3["Cg"],new=False,ax=axes[2],title="c. k1-k1",x_label="Time (s)",y_label="Concentration", vertical_lines=[15,20,35])
plt.show()