import numpy as np

from .dynamic_model import Model


def generate_convection_model(Kaw_initial, k_initial, A_ini, Dg_ini,  Vr, Vl, tMS):
    def dCg_dt(y, t, Kaw, k, Vr, A, Dg):
        return ((k * A) / Vr) * (Kaw * y["Cl"] - y["Cg"]) - y["Cg"] *Dg/Vr
    def dCl_dt(y, t, Kaw, k, Vl, A):
        return - ((k * A) / Vl) * (Kaw * y["Cl"] - y["Cg"])
    def dCMS_dt(y, t, tMS):
        return (y["Cg"] - y["CMS"]) / tMS
    funcs_dict_release = {
        "Cg": dCg_dt,
        "Cl": dCl_dt,
        "CMS": dCMS_dt
    }
    params_dict_release = {
        "Cg": {"Kaw": Kaw_initial, "k": k_initial , "Vr": Vr, "A": A_ini, "Dg": Dg_ini},
        "Cl": {"Kaw": Kaw_initial, "k": k_initial , "Vl": Vl, "A": A_ini},
        "CMS": {"tMS": tMS}
    }
    def open_step(y_current, params_current=None):
        return {
            "Cg": 0,
            "Cl": y_current["Cl"],
            "CMS": y_current["CMS"],
        }
    return Model(params_dict_release, funcs_dict_release, 
                 event_funcs={"open": open_step}, name="convection_model")


def in_vivo_diffusion_solution_model(
    QNA_func, QSaliva, AOAL, VOA, AFAL, VFA, VFL, VNA, tMS, KAL, kL
):
    def dVOL_dt(y, t, QSaliva):
        return QSaliva
    def dCOA_dt(y, t, AOAL, kL, VOA, KAL):
        COA, COL = y["COA"], y["COL"]
        return AOAL * kL * (COL - COA / KAL) / VOA
    def dCOL_dt(y, t, AOAL, kL, QSaliva, KAL):
        COA, COL,VOL = y["COA"], y["COL"],y["VOL"]
        COA = max(y["COA"], 0.0)
        COL = max(y["COL"], 0.0)
        # Protection numérique compatible avec des volumes exprimés en m³.
        VOL = max(y["VOL"], np.finfo(float).tiny)
        return -AOAL * kL * (COL - COA / KAL) / VOL - QSaliva * COL / VOL
    def dCFA_dt(y, t, AFAL, kL, VFA, QNA_func,  KAL):
        CFA, CFL, CNA = max(0, y["CFA"]), max(0, y["CFL"]), max(0, y["CNA"])
        QNAt = QNA_func(t)
        term_from_FL = AFAL * kL * (CFL - CFA / KAL) / VFA
        term_from_inspi = (QNAt >= 0) * QNAt * (CNA - CFA) / VFA
        term_from_expi = (QNAt < 0) * (-QNAt) * (0.0 - CFA) / VFA
        return term_from_FL + term_from_inspi + term_from_expi
    def dCFL_dt(y, t, AFAL, kL, VFL, KAL):
        CFA, CFL = max(0,y["CFA"]),max(0, y["CFL"])
        if CFL > 0:
            return -AFAL * kL * (CFL - CFA / KAL) / VFL
        else:
            return 0.0
    def dCNA_dt(y, t, QNA_func, VNA):
        CFA=max(y["CFA"], 0.0)
        CNA = max(y["CNA"], 0.0)
        QNAt = QNA_func(t)
        term2 = (QNAt >= 0) * QNAt * (0.0 - CNA) / VNA
        term3 = (QNAt < 0) * (-QNAt) * (CFA - CNA) / VNA
        return term2 + term3
    def dCMS_dt(y, t, tMS):
        CNA = max(y["CNA"], 0.0)
        CMS = y["CMS"]
        return (CNA - CMS) / tMS
    # Fonction dict
    funcs_dict = {
        "VOL": dVOL_dt,
        "COA": dCOA_dt,
        "COL": dCOL_dt,
        "CFA": dCFA_dt,
        "CFL": dCFL_dt,
        "CNA": dCNA_dt,
        "CMS": dCMS_dt,
    }
    # ░Paramètres associés
    params_dict = {
        "VOL": {"QSaliva": QSaliva},
        "COA": {"AOAL": AOAL, "kL": kL, "VOA": VOA, "KAL": KAL},
        "COL": {"AOAL": AOAL, "kL": kL, "QSaliva": QSaliva, "KAL": KAL},
        "CFA": {"AFAL": AFAL, "kL": kL, "VFA": VFA, "QNA_func": QNA_func    , "KAL": KAL},
        "CFL": {"AFAL": AFAL, "kL": kL, "VFL": VFL, "KAL": KAL},
        "CNA": {"QNA_func": QNA_func, "VNA": VNA},
        "CMS": {"tMS": tMS},
    }
    #  Conditions initiales
    def swallow_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        COL= y0_current["COL"]
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VNA"] + params_current["VFA"])
        VOL_m=params_current["VOL_m"]
        return {
            "VOL": VOL_m ,  # ancien :y0_current["VOL"]
            "COA": CFN ,
            "COL": COL ,
            "CFA": CFN,
            "CFL": y0_current["COL"],  # ancien y0_current["CFP"]
            "CNA": CFN,
            "CMS": y0_current["CMS"]
        }
    def mimic_chew(y0_current, params_current=None):
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VNA"] + params_current["VFA"])
        return {
            "VOL": y0_current["VOL"] ,  # ancien :y0_current["VOL"]
            "COA": CFN ,
            "COL": y0_current["COL"] ,
            "CFA": CFN,
            "CFL": y0_current["CFL"],  # ancien y0_current["CFP"]
            "CNA": CFN,
            "CMS": y0_current["CMS"]
        }
    return Model(params_dict, funcs_dict, 
                 event_funcs={"swallow": swallow_step, "jaw_move": mimic_chew}, 
                 name="in_vivo_diffusion_solution_model")


def in_vivo_diffusion_solid_model_with_temp(
    # Conditions initiales
    # static parameters
    tMS, kL, KAL, QSaliva, AOAL, VOA, AFAL, VFA, VNA, VFL, COP, AOLP_coefficient,
    # Fonctions dynamiques
     QNA_func ,
    kT,T_mouth,v_mouth,alpha # QOA_func, , phi_NM_func
):
    """
    Modèle in vivo pour un produit solide en bouche (Doyennette 2014)
    Retourne un dict avec funcs_dict, params_dict, y0_dict
    Description:
Modélise le comportement d'un produit solide en bouche (échange masse/volume entre
produit, salive, pharynx, et air) via un système d'EDO. Conçu pour être utilisé
avec un solveur ODE qui accepte des dictionnaires y, params et des événements.
Variables d'état (y):
VOP : Volume du produit dans la cavité orale (float, volume)
VOL: Volume salivaire oral (float, volume) + Volume de produit déplacé vers le pharynx (float, volume)
COL : Concentration du produit dans le liquide oral (float, masse/volume)
COA : Concentration du produit dans l'air oral/pharyngé (float, masse/volume)
CFA : Concentration dans l'air pharyngé/respiratoire (float, masse/volume)
CNA : Concentration dans l'air nasal (float, masse/volume)
CNM : Concentration dans le mucus/niveau nasal (ou autre réservoir nasal) (float)
Paramètres d'entrée:
VOP_ini, VOL_ini,  COL_ini, COA_ini, CFA_ini, CNA_ini, CNM_ini:
conditions initiales (floats).
iso_interp: fonction d'interpolation d'isolement (callable(t) -> float).
(NB : dans l'implémentation actuelle iso_interp apparaît dans la signature
mais n'est pas utilisé dans les équations du modèle.)
QSaliva : débit salivaire (float, volume / temps).
AOAL : surface d'échange oral-pharyngé (float, surface).
VOA, VFA, VFP, VNA : volumes des compartiments air oral, air pharyngé, pharynx, nasal (floats).
AFAL : surface d'échange pharynx-air (float).
kL : coefficient de transfert masse produit → air oral (float).
KAL : coefficient de partage / partition produit/air (float).
v : coefficient volumique relatif (float, volume/time/surface — utilisé avec AOLP_func).
AOLP_func: callable(t) -> float, aire orale de contact du produit (même signature que les autres fonctions dynamiques).
QOA_func: callable(t) -> float, débit/respi oral (positif vers l'extérieur si défini).
QNA_func: callable(t) -> float, débit nasal.
phi_NM_func: callable(t) -> float, flux nasal → mucus (masse/temps).
Signature retournée:
Retourne un dict structuré pour le solveur:
"funcs_dict": mapping variable -> fonction dy/dt (signatures: func(y, t, **params)).
"params_dict": mapping variable -> dict des paramètres nécessaires à la fonction.
"y0_dict": mapping variable -> valeur initiale.
"event_funcs" (optionnel) : mapping d'événement -> fonction évènementielle qui prend
l'état actuel (y0_current) et params_current, et renvoie un nouveau dictionnaire d'états
(post-événement). Dans le modèle renvoyé, on fournit swallow et chew.
    """
    #  A = B*(VOP)^{2/3} => je trouve B avec les conditions initiales (A0 = k*B0 )
    #  Si on mache, on multiplie le B par quelque chose: le volume reste constant mais l'aire augmente
    # ------------------------------
    # Equations différentielles
    # ------------------------------
    # mouth ===========================
    # solid product-------------------
    def AOLP_fun(vop,AOLP_coefficient):
        return AOLP_coefficient* vop**(2/3)
    def dVOP_dt(y, t, v_mouth, alpha, AOLP_coefficient,T_mouth):
        VOP = y["VOP"]
        TP = y["TP"]
        if VOP <= 0:
            return 0.0
        AOLP = AOLP_fun(VOP, AOLP_coefficient)
        # version linéaire
        v_eff = v_mouth * (1 + alpha * (TP - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        return -v_eff * AOLP
        # liquid product -----------------
    # volume of saliva
    def dVOL_dt(y, t, QSaliva, v_mouth,alpha, AOLP_coefficient,T_mouth):
        TP = y["TP"]
        if y.get("VOL", 0.0) <= 0.0:
            return 0.0
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOP", 0.0) <= 0.0:
            # No product volume lost, only saliva is added
            return QSaliva
        AOLP = AOLP_fun(y["VOP"],AOLP_coefficient)
        v_eff = v_mouth * (1 + alpha * (TP - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        dVODP = v_eff * AOLP
        dVOS = QSaliva
        return dVODP + dVOS
    # volume of dissolved product
    # concentration in liquid phase
    def dCOL_dt(y, t,  kL, AOAL, KAL, COP, v_mouth,alpha, AOLP_coefficient,T_mouth):
        """
        d(VOL*COL)/dt = phi_OLP - phi_OAL
        phi_OLP = v * AOLP * COP
        phi_OAL = kL * AOAL * (COL - COALeta)
        """
        # pour simplification on prend VOL = VOS + VOPD
        TP = y["TP"]
        VOL = y["VOL"]
        if VOL <= 0.0:
            # Défensif : si le volume est nul, on évite la division et on garde la concentration
            # inchangée (dCOL/dt = 0). Cela évite NaN au démarrage quand VOS=VOPD=0.
            return 0.0
        COL, COA = y["COL"], y["COA"]  # concentration oral gaz
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOP", 0.0) <= 0.0:
            AOLP = 0.0
        else:
            AOLP = AOLP_fun(y["VOP"],AOLP_coefficient)
        v_eff = v_mouth * (1 + alpha * (TP - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        phi_OLP = v_eff * AOLP  * COP  # flux of diffusion of solid
        phi_OAL = kL * AOAL * (COL - COA / KAL)  # flux of diffusion liquid => gaz
        dVOL_dt = QSaliva + v_eff * AOLP 
        term_supp = COL * dVOL_dt  # terme de dilution/concentration
        return (phi_OLP - phi_OAL - term_supp) / VOL
    # concentration in oral cavity (gaz)
    def dCOA_dt(y, t, kL, AOAL, KAL, VOA):
        # QOA = QOA_func(t)
        COL, COA = y["COL"], y["COA"]
        phi_OAL = kL * AOAL * (COL - COA / KAL)
        # Flux oral vers pharynx selon le signe de QOA
        # flux_respi = QOA * (y["CFA"] - COA) if QOA >= 0 else 0
        return (phi_OAL) / VOA  # +flux_respi/VOA
    # Pharynx ==========================
    # Concentration aroma
    def dCFA_dt(y, t, AFAL, kL, VFA, QNA_func, KAL):
        # QOA = QOA_func(t)
        QNA = QNA_func(t)
        phi_FAL = AFAL * kL * (y["CFL"] - y["CFA"] / KAL)
        # Flux selon respiration
        # flux_air = (QOA * (y["COA"] - y["CFA"])) if QOA < 0 else 0
        flux_respi = (QNA * (y["CNA"] - y["CFA"])) if QNA >= 0 else (-QNA * (0 - y["CFA"]))
        return (phi_FAL + flux_respi) / VFA  # +flux_air/VFA
    # concentration product in pharynx
    def dCFL_dt(y, t, AFAL, kL, VFL,KAL):
        phi_FAL = - kL * AFAL * (y["CFL"] - y["CFA"] / KAL)
        return (phi_FAL) / VFL
    # Nose==================
    def dCNA_dt(y, t, QNA_func, VNA):
        QNA = QNA_func(t)
        # phi_NM = phi_NM_func(t)
        flux_respi = (QNA * (0 - y["CNA"])) if QNA >= 0 else (-QNA * (y["CFA"] - y["CNA"]))
        return (flux_respi) / VNA  # +phi_NM/flux_respi
    # mucosal adsorption
    # def dCNM_dt(y, t, phi_NM_func):
    #    phi_NM = phi_NM_func(t)
    #    return -phi_NM
    # PTRMS ===================
    def dCMS_dt(y, t, tMS):
        return (y["CNA"] - y["CMS"]) / tMS
    def dTP_dt(y, t, kT, T_mouth=36):
        TP = y["TP"]
        return kT * (T_mouth - TP)
    # ------------------------------
    # Dictionnaires pour le solveur
    # ------------------------------
    funcs_dict = {
        "VOP": dVOP_dt,
        "VOL": dVOL_dt,
        "COL": dCOL_dt,
        "COA": dCOA_dt,
        "CFA": dCFA_dt,
        "CFL": dCFL_dt,
        "CNA": dCNA_dt,
        "CMS": dCMS_dt,
        "TP":dTP_dt
    }
    params_dict = {
        "VOP": {"v_mouth":v_mouth,"alpha":alpha,"AOLP_coefficient": AOLP_coefficient,"T_mouth":T_mouth},
        "VOL": {"QSaliva": QSaliva, "v_mouth":v_mouth,"alpha":alpha,"AOLP_coefficient": AOLP_coefficient,"T_mouth":T_mouth},
        "COL": { "kL": kL, "AOAL": AOAL, "KAL": KAL, "COP": COP, "v_mouth":v_mouth,"alpha":alpha,"AOLP_coefficient": AOLP_coefficient,"T_mouth":T_mouth},
        "COA": {"kL": kL, "AOAL": AOAL, "KAL": KAL, "VOA": VOA},
        "CFA": {"AFAL": AFAL, "kL": kL, "VFA": VFA, "QNA_func": QNA_func, "KAL": KAL},
        "CFL": {"AFAL": AFAL, "kL": kL, "VFL": VFL,"KAL" : KAL},
        "CNA": {"QNA_func": QNA_func, "VNA": VNA},
        #    "CNM": {"phi_NM_func": phi_NM_func},
        "CMS": {"tMS": tMS},
        "TP":{"kT": kT, "T_mouth":T_mouth}
    }

    def opening_velum(y0_current, params_current=None,params_model=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": y0_current["VOL"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["CFL"],
            "CNA": CFN,
            #   "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP":y0_current["TP"]
        }
  
    def chew(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
                * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": y0_current["VOL"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["CFL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP": y0_current["TP"],
        }
    def swallow_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP":y0_current["TP"]
        }

    def total_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": 0,
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP":y0_current["TP"]
        }

    return Model(params_dict, funcs_dict, 
                 event_funcs={"swallow": swallow_step, "chew": chew, "total_swallow": total_step,"open_velum": opening_velum}, 
                 name="in_vivo_diffusion_gel_model")


#=======Other model examples
def in_vivo_diffusion_solid_model(
    # Conditions initiales
    # static parameters
    tMS, kOL, KOAL, v, QSaliva, AOAP, VOA, AFAP, VFA, VNA, VFL, COP, AOLP_coefficient,
    # Fonctions dynamiques
     QNA_func  # QOA_func, , phi_NM_func
):
    """
    Modèle in vivo pour un produit solide en bouche (Doyennette 2014)
    Retourne un dict avec funcs_dict, params_dict, y0_dict
    Description:
Modélise le comportement d'un produit solide en bouche (échange masse/volume entre
produit, salive, pharynx, et air) via un système d'EDO. Conçu pour être utilisé
avec un solveur ODE qui accepte des dictionnaires y, params et des événements.
Variables d'état (y):
VOP : Volume du produit dans la cavité orale (float, volume)
VOL: Volume salivaire oral (float, volume) + Volume de produit déplacé vers le pharynx (float, volume)
COL : Concentration du produit dans le liquide oral (float, masse/volume)
COA : Concentration du produit dans l'air oral/pharyngé (float, masse/volume)
CFA : Concentration dans l'air pharyngé/respiratoire (float, masse/volume)
CNA : Concentration dans l'air nasal (float, masse/volume)
CNM : Concentration dans le mucus/niveau nasal (ou autre réservoir nasal) (float)
Paramètres d'entrée:
VOP_ini, VOL_ini,  COL_ini, COA_ini, CFA_ini, CNA_ini, CNM_ini:
conditions initiales (floats).
iso_interp: fonction d'interpolation d'isolement (callable(t) -> float).
(NB : dans l'implémentation actuelle iso_interp apparaît dans la signature
mais n'est pas utilisé dans les équations du modèle.)
QSaliva : débit salivaire (float, volume / temps).
AOAP : surface d'échange oral-pharyngé (float, surface).
VOA, VFA, VFP, VNA : volumes des compartiments air oral, air pharyngé, pharynx, nasal (floats).
AFAP : surface d'échange pharynx-air (float).
kOL : coefficient de transfert masse produit → air oral (float).
KOAL : coefficient de partage / partition produit/air (float).
v : coefficient volumique relatif (float, volume/time/surface — utilisé avec AOLP_func).
AOLP_func: callable(t) -> float, aire orale de contact du produit (même signature que les autres fonctions dynamiques).
QOA_func: callable(t) -> float, débit/respi oral (positif vers l'extérieur si défini).
QNA_func: callable(t) -> float, débit nasal.
phi_NM_func: callable(t) -> float, flux nasal → mucus (masse/temps).
Signature retournée:
Retourne un dict structuré pour le solveur:
"funcs_dict": mapping variable -> fonction dy/dt (signatures: func(y, t, **params)).
"params_dict": mapping variable -> dict des paramètres nécessaires à la fonction.
"y0_dict": mapping variable -> valeur initiale.
"event_funcs" (optionnel) : mapping d'événement -> fonction évènementielle qui prend
l'état actuel (y0_current) et params_current, et renvoie un nouveau dictionnaire d'états
(post-événement). Dans le modèle renvoyé, on fournit swallow et chew.
    """
    #  A = B*(VOP)^{2/3} => je trouve B avec les conditions initiales (A0 = k*B0 )
    #  Si on mache, on multiplie le B par quelque chose: le volume reste constant mais l'aire augmente
    # ------------------------------
    # Equations différentielles
    # ------------------------------
    # mouth ===========================
    # solid product-------------------
    def AOLP_fun(vop,AOLP_coefficient):
        return AOLP_coefficient* vop**(2/3)
    def dVOP_dt(y, t, v, AOLP_coefficient):
        # Si VOP déjà nul et la dérivée veut le rendre négatif => stopper (VÉRIFIER D'ABORD pour éviter NaN avec VOP**(2/3))
        if y.get("VOP", 0.0) <= 0.0:
             return 0.0
        AOLP = AOLP_fun(vop=y["VOP"],AOLP_coefficient=AOLP_coefficient) # AOLP func doit dépendre du VOP avec A = B*VOP^(2/3) : def AOLP_func(VOP,B)=B*VOP^(2/3) (avec B calculé sur les conditions initiales, qui varie si le bonbon est fragmenté)
        rate = -v * AOLP
        return rate
    # liquid product -----------------
    # volume of saliva
    def dVOL_dt(y, t, QSaliva, v, AOLP_coefficient):
        if y.get("VOL", 0.0) <= 0.0:
            return 0.0
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOP", 0.0) <= 0.0:
            # No product volume lost, only saliva is added
            return QSaliva
        AOLP = AOLP_fun(y["VOP"],AOLP_coefficient)
        dVODP = v * AOLP
        dVOS = QSaliva
        return dVODP + dVOS
    # volume of dissolved product
    # concentration in liquid phase
    def dCOL_dt(y, t, v, kOL, AOAP, KOAL, COP, AOLP_coefficient):
        """
        d(VOL*COL)/dt = phi_OLP - phi_OAL
        phi_OLP = v * AOLP * COP
        phi_OAL = kOL * AOAL * (COL - COALeta)
        """
        # pour simplification on prend VOL = VOS + VOPD
        VOL = y["VOL"]
        if VOL <= 0.0:
            # Défensif : si le volume est nul, on évite la division et on garde la concentration
            # inchangée (dCOL/dt = 0). Cela évite NaN au démarrage quand VOS=VOPD=0.
            return 0.0
        COL, COA = y["COL"], y["COA"]  # concentration oral gaz
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOP", 0.0) <= 0.0:
            AOLP = 0.0
        else:
            AOLP = AOLP_fun(y["VOP"],AOLP_coefficient)
        phi_OLP = v * AOLP  * COP  # flux of diffusion of solid
        phi_OAL = kOL * AOAP * (COL - COA / KOAL)  # flux of diffusion liquid => gaz
        dVOL_dt = QSaliva + v * AOLP 
        term_supp = COL * dVOL_dt  # terme de dilution/concentration
        return (phi_OLP - phi_OAL - term_supp) / VOL
    # concentration in oral cavity (gaz)
    def dCOA_dt(y, t, kOL, AOAP, KOAL, VOA):
        # QOA = QOA_func(t)
        COL, COA = y["COL"], y["COA"]
        phi_OAL = kOL * AOAP * (COL - COA / KOAL)
        # Flux oral vers pharynx selon le signe de QOA
        # flux_respi = QOA * (y["CFA"] - COA) if QOA >= 0 else 0
        return (phi_OAL) / VOA  # +flux_respi/VOA
    # Pharynx ==========================
    # Concentration aroma
    def dCFA_dt(y, t, AFAP, kOL, VFA, QNA_func, KOAL):
        # QOA = QOA_func(t)
        QNA = QNA_func(t)
        phi_FAL = AFAP * kOL * (y["CFL"] - y["CFA"] / KOAL)
        # Flux selon respiration
        # flux_air = (QOA * (y["COA"] - y["CFA"])) if QOA < 0 else 0
        flux_respi = (QNA * (y["CNA"] - y["CFA"])) if QNA >= 0 else (-QNA * (0 - y["CFA"]))
        return (phi_FAL + flux_respi) / VFA  # +flux_air/VFA
    # concentration product in pharynx
    def dCFL_dt(y, t, AFAP, kOL, VFL,KOAL):
        phi_FAL = - kOL * AFAP * (y["CFL"] - y["CFA"] / KOAL)
        return (phi_FAL) / VFL
    # Nose==================
    def dCNA_dt(y, t, QNA_func, VNA):
        QNA = QNA_func(t)
        # phi_NM = phi_NM_func(t)
        flux_respi = (QNA * (0 - y["CNA"])) if QNA >= 0 else (-QNA * (y["CFA"] - y["CNA"]))
        return (flux_respi) / VNA  # +phi_NM/flux_respi
    # mucosal adsorption
    # def dCNM_dt(y, t, phi_NM_func):
    #    phi_NM = phi_NM_func(t)
    #    return -phi_NM
    # PTRMS ===================
    def dCMS_dt(y, t, tMS):
        return (y["CNA"] - y["CMS"]) / tMS
    # ------------------------------
    # Dictionnaires pour le solveur
    # ------------------------------
    funcs_dict = {
        "VOP": dVOP_dt,
        "VOL": dVOL_dt,
        "COL": dCOL_dt,
        "COA": dCOA_dt,
        "CFA": dCFA_dt,
        "CFL": dCFL_dt,
        "CNA": dCNA_dt,
        "CMS": dCMS_dt
    }
    params_dict = {
        "VOP": {"v": v, "AOLP_coefficient": AOLP_coefficient},
        "VOL": {"QSaliva": QSaliva,  "v": v, "AOLP_coefficient": AOLP_coefficient},
        "COL": {"v": v, "kOL": kOL, "AOAP": AOAP, "KOAL": KOAL, "COP": COP, "AOLP_coefficient": AOLP_coefficient},
        "COA": {"kOL": kOL, "AOAP": AOAP, "KOAL": KOAL, "VOA": VOA},
        "CFA": {"AFAP": AFAP, "kOL": kOL, "VFA": VFA, "QNA_func": QNA_func, "KOAL": KOAL},
        "CFL": {"AFAP": AFAP, "kOL": kOL, "VFL": VFL,"KOAL" : KOAL},
        "CNA": {"QNA_func": QNA_func, "VNA": VNA},
        #    "CNM": {"phi_NM_func": phi_NM_func},
        "CMS": {"tMS": tMS}
    }
    def swallow_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"]
        }
    def total_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": 0,
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"]
        }
    def chew(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": y0_current["VOL"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["CFL"],
            "CNA": CFN,
            #   "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"]
        }
    return Model(params_dict, funcs_dict, 
                 event_funcs={"swallow": swallow_step, "chew": chew, "total_swallow": total_step}, 
                 name="in_vivo_diffusion_gel_model")


def generate_convection_model_without_measure(Kaw_initial, k_initial, A_ini,  Vr, Vl):
    def dCg_dt(y, t, Kaw, k, Vr, A):
        return ((k * A) / Vr) * (Kaw * y["Cl"] - y["Cg"]) 
    def dCl_dt(y, t, Kaw, k, Vl, A):
        return - ((k * A) / Vl) * (Kaw * y["Cl"] - y["Cg"])
    funcs_dict_release = {
        "Cg": dCg_dt,
        "Cl": dCl_dt,
    }
    params_dict_release = {
        "Cg": {"Kaw": Kaw_initial, "k": k_initial , "Vr": Vr, "A": A_ini},
        "Cl": {"Kaw": Kaw_initial, "k": k_initial , "Vl": Vl, "A": A_ini},
    }
    def open_step(y_current, params_current=None):
        return {
            "Cg": 0,
            "Cl": y_current["Cl"],
        }
    return Model(params_dict_release, funcs_dict_release, 
                 event_funcs={"open": open_step}, name="convection_model_without_measure")



def in_vivo_diffusion_solid_model_with_temp_and_surface(
    # Conditions initiales
    # static parameters
    tMS, kL, KAL, QSaliva, AOAL, VOA, AFAL, VFA, VNA, VFL, COP, 
    # Fonctions dynamiques
     QNA_func ,
    kT,T_mouth,v_mouth,alpha # QOA_func, , phi_NM_func
):
    """
    Modèle in vivo pour un produit solide en bouche (Doyennette 2014)
    Retourne un dict avec funcs_dict, params_dict, y0_dict
    Description:
Modélise le comportement d'un produit solide en bouche (échange masse/volume entre
produit, salive, pharynx, et air) via un système d'EDO. Conçu pour être utilisé
avec un solveur ODE qui accepte des dictionnaires y, params et des événements.
Variables d'état (y):
VOP : Volume du produit dans la cavité orale (float, volume)
VOL: Volume salivaire oral (float, volume) + Volume de produit déplacé vers le pharynx (float, volume)
COL : Concentration du produit dans le liquide oral (float, masse/volume)
COA : Concentration du produit dans l'air oral/pharyngé (float, masse/volume)
CFA : Concentration dans l'air pharyngé/respiratoire (float, masse/volume)
CNA : Concentration dans l'air nasal (float, masse/volume)
CNM : Concentration dans le mucus/niveau nasal (ou autre réservoir nasal) (float)
Paramètres d'entrée:
VOP_ini, VOL_ini,  COL_ini, COA_ini, CFA_ini, CNA_ini, CNM_ini:
conditions initiales (floats).
iso_interp: fonction d'interpolation d'isolement (callable(t) -> float).
(NB : dans l'implémentation actuelle iso_interp apparaît dans la signature
mais n'est pas utilisé dans les équations du modèle.)
QSaliva : débit salivaire (float, volume / temps).
AOAL : surface d'échange oral-pharyngé (float, surface).
VOA, VFA, VFP, VNA : volumes des compartiments air oral, air pharyngé, pharynx, nasal (floats).
AFAL : surface d'échange pharynx-air (float).
kL : coefficient de transfert masse produit → air oral (float).
KAL : coefficient de partage / partition produit/air (float).
v : coefficient volumique relatif (float, volume/time/surface — utilisé avec AOLP_func).
AOLP_func: callable(t) -> float, aire orale de contact du produit (même signature que les autres fonctions dynamiques).
QOA_func: callable(t) -> float, débit/respi oral (positif vers l'extérieur si défini).
QNA_func: callable(t) -> float, débit nasal.
phi_NM_func: callable(t) -> float, flux nasal → mucus (masse/temps).
Signature retournée:
Retourne un dict structuré pour le solveur:
"funcs_dict": mapping variable -> fonction dy/dt (signatures: func(y, t, **params)).
"params_dict": mapping variable -> dict des paramètres nécessaires à la fonction.
"y0_dict": mapping variable -> valeur initiale.
"event_funcs" (optionnel) : mapping d'événement -> fonction évènementielle qui prend
l'état actuel (y0_current) et params_current, et renvoie un nouveau dictionnaire d'états
(post-événement). Dans le modèle renvoyé, on fournit swallow et chew.
    """
    #  A = B*(VOP)^{2/3} => je trouve B avec les conditions initiales (A0 = k*B0 )
    #  Si on mache, on multiplie le B par quelque chose: le volume reste constant mais l'aire augmente
    # ------------------------------
    # Equations différentielles
    # ------------------------------
    # mouth ===========================
    # solid product-------------------
    def AOLP_fun(y,AOAL):
        VOP = y["VOP"]
        if VOP <= 0:
            return 0.0
        return min(y["SP"] * VOP**(2/3),AOAL)
    def dVOP_dt(y, t, v_mouth, alpha, T_mouth,AOAL):
        VOP = y["VOP"]
        TP = y["TP"]
        if VOP <= 0:
            return 0.0
        AOLP = AOLP_fun(y,AOAL)
        # version linéaire
        v_eff = v_mouth * (1 + alpha * (TP - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        return -v_eff * AOLP
        # liquid product -----------------
    # volume of saliva
    def dVOL_dt(y, t, QSaliva, v_mouth,alpha, T_mouth,AOAL):
        TP = y["TP"]
        if y.get("VOL", 0.0) <= 0.0:
            return 0.0
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOL", 0.0) <= 0.0:
            # No product volume lost, only saliva is added
            return QSaliva
        AOLP = AOLP_fun(y,AOAL)
        v_eff = v_mouth * (1 + alpha * (TP - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        dVODP = v_eff * AOLP
        dVOS = QSaliva
        return dVODP + dVOS
    # volume of dissolved product
    # concentration in liquid phase
    def dCOL_dt(y, t,  kL, AOAL, KAL, COP, v_mouth,alpha, T_mouth):
        """
        d(VOL*COL)/dt = phi_OLP - phi_OAL
        phi_OLP = v * AOLP * COP
        phi_OAL = kL * AOAL * (COL - COALeta)
        """
        # pour simplification on prend VOL = VOS + VOPD
        TP = y["TP"]
        VOL = y["VOL"]
        if VOL <= 0.0:
            # Défensif : si le volume est nul, on évite la division et on garde la concentration
            # inchangée (dCOL/dt = 0). Cela évite NaN au démarrage quand VOS=VOPD=0.
            return 0.0
        COL, COA = y["COL"], y["COA"]  # concentration oral gaz
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOP", 0.0) <= 0.0:
            AOLP = 0.0
        else:
            AOLP = AOLP_fun(y,AOAL)
        v_eff = v_mouth * (1 + alpha * (TP - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        phi_OLP = v_eff * AOLP  * COP  # flux of diffusion of solid
        phi_OAL = kL * AOAL * (COL - COA / KAL)  # flux of diffusion liquid => gaz
        dVOL_dt = QSaliva + v_eff * AOLP 
        term_supp = COL * dVOL_dt  # terme de dilution/concentration
        return (phi_OLP - phi_OAL - term_supp) / VOL
    # concentration in oral cavity (gaz)
    def dCOA_dt(y, t, kL, AOAL, KAL, VOA):
        # QOA = QOA_func(t)
        COL, COA = y["COL"], y["COA"]
        phi_OAL = kL * AOAL * (COL - COA / KAL)
        # Flux oral vers pharynx selon le signe de QOA
        # flux_respi = QOA * (y["CFA"] - COA) if QOA >= 0 else 0
        return (phi_OAL) / VOA  # +flux_respi/VOA
    # Pharynx ==========================
    # Concentration aroma
    def dCFA_dt(y, t, AFAL, kL, VFA, QNA_func, KAL):
        # QOA = QOA_func(t)
        QNA = QNA_func(t)
        phi_FAL = AFAL * kL * (y["CFL"] - y["CFA"] / KAL)
        # Flux selon respiration
        # flux_air = (QOA * (y["COA"] - y["CFA"])) if QOA < 0 else 0
        flux_respi = (QNA * (y["CNA"] - y["CFA"])) if QNA >= 0 else (-QNA * (0 - y["CFA"]))
        return (phi_FAL + flux_respi) / VFA  # +flux_air/VFA
    # concentration product in pharynx
    def dCFL_dt(y, t, AFAL, kL, VFL,KAL):
        phi_FAL = - kL * AFAL * (y["CFL"] - y["CFA"] / KAL)
        return (phi_FAL) / VFL
    # Nose==================
    def dCNA_dt(y, t, QNA_func, VNA):
        QNA = QNA_func(t)
        # phi_NM = phi_NM_func(t)
        flux_respi = (QNA * (0 - y["CNA"])) if QNA >= 0 else (-QNA * (y["CFA"] - y["CNA"]))
        return (flux_respi) / VNA  # +phi_NM/flux_respi
    # mucosal adsorption
    # def dCNM_dt(y, t, phi_NM_func):
    #    phi_NM = phi_NM_func(t)
    #    return -phi_NM
    # PTRMS ===================
    def dCMS_dt(y, t, tMS):
        return (y["CNA"] - y["CMS"]) / tMS
    def dTP_dt(y, t, kT, T_mouth=36):
        TP = y["TP"]
        return kT * (T_mouth - TP)
    def dSP_dt(y, t): 
        return 0
    # ------------------------------
    # Dictionnaires pour le solveur
    # ------------------------------
    funcs_dict = {
        "VOP": dVOP_dt,
        "VOL": dVOL_dt,
        "COL": dCOL_dt,
        "COA": dCOA_dt,
        "CFA": dCFA_dt,
        "CFL": dCFL_dt,
        "CNA": dCNA_dt,
        "CMS": dCMS_dt,
        "TP":dTP_dt,
        "SP": dSP_dt
    }
    params_dict = {
        "VOP": {"v_mouth":v_mouth,"alpha":alpha,"T_mouth":T_mouth,"AOAL": AOAL},
        "VOL": {"QSaliva": QSaliva, "v_mouth":v_mouth,"alpha":alpha,"T_mouth":T_mouth,"AOAL": AOAL},
        "COL": { "kL": kL, "AOAL": AOAL, "KAL": KAL, "COP": COP, "v_mouth":v_mouth,"alpha":alpha,"T_mouth":T_mouth,"AOAL": AOAL},
        "COA": {"kL": kL, "AOAL": AOAL, "KAL": KAL, "VOA": VOA},
        "CFA": {"AFAL": AFAL, "kL": kL, "VFA": VFA, "QNA_func": QNA_func, "KAL": KAL},
        "CFL": {"AFAL": AFAL, "kL": kL, "VFL": VFL,"KAL" : KAL},
        "CNA": {"QNA_func": QNA_func, "VNA": VNA},
        #    "CNM": {"phi_NM_func": phi_NM_func},
        "CMS": {"tMS": tMS},
        "TP":{"kT": kT, "T_mouth":T_mouth},
        "SP": {}
    }

    def opening_velum(y0_current, params_current=None,params_model=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": y0_current["VOL"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["CFL"],
            "CNA": CFN,
            #   "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP":y0_current["TP"],  
            "SP":y0_current["SP"] 
        }
  
    def chew(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
                * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": y0_current["VOL"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["CFL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP": y0_current["TP"],
            "SP": y0_current["SP"]*params_current["chew_factor"]
        }
    def swallow_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP":y0_current["TP"],
            "SP": y0_current["SP"]
             }

    def total_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": 0,
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "TP":y0_current["TP"],
            "SP": 0
        }

    return Model(params_dict, funcs_dict, 
                 event_funcs={"swallow": swallow_step, "chew": chew, "total_swallow": total_step,"open_velum": opening_velum}, 
                 name="in_vivo_diffusion_gel_model")


def in_vivo_diffusion_solid_model_with_temp_and_AOLP(
    # Conditions initiales
    # static parameters
    tMS, kL, KAL, QSaliva, AOAL, VOA, AFAL, VFA, VNA, VFL, COP, AOLP_coefficient,
    # Fonctions dynamiques
     QNA_func ,
    kT,T_mouth,v_mouth,alpha, AOLP_fun # QOA_func, , phi_NM_func
):
    """
    Modèle in vivo pour un produit solide en bouche (Doyennette 2014)
    Retourne un dict avec funcs_dict, params_dict, y0_dict
    Description:
Modélise le comportement d'un produit solide en bouche (échange masse/volume entre
produit, salive, pharynx, et air) via un système d'EDO. Conçu pour être utilisé
avec un solveur ODE qui accepte des dictionnaires y, params et des événements.
Variables d'état (y):
VOP : Volume du produit dans la cavité orale (float, volume)
VOL: Volume salivaire oral (float, volume) + Volume de produit déplacé vers le pharynx (float, volume)
COL : Concentration du produit dans le liquide oral (float, masse/volume)
COA : Concentration du produit dans l'air oral/pharyngé (float, masse/volume)
CFA : Concentration dans l'air pharyngé/respiratoire (float, masse/volume)
CNA : Concentration dans l'air nasal (float, masse/volume)
CNM : Concentration dans le mucus/niveau nasal (ou autre réservoir nasal) (float)
Paramètres d'entrée:
VOP_ini, VOL_ini,  COL_ini, COA_ini, CFA_ini, CNA_ini, CNM_ini:
conditions initiales (floats).
iso_interp: fonction d'interpolation d'isolement (callable(t) -> float).
(NB : dans l'implémentation actuelle iso_interp apparaît dans la signature
mais n'est pas utilisé dans les équations du modèle.)
QSaliva : débit salivaire (float, volume / temps).
AOAL : surface d'échange oral-pharyngé (float, surface).
VOA, VFA, VFP, VNA : volumes des compartiments air oral, air pharyngé, pharynx, nasal (floats).
AFAL : surface d'échange pharynx-air (float).
kL : coefficient de transfert masse produit → air oral (float).
KAL : coefficient de partage / partition produit/air (float).
v : coefficient volumique relatif (float, volume/time/surface — utilisé avec AOLP_func).
AOLP_func: callable(t) -> float, aire orale de contact du produit (même signature que les autres fonctions dynamiques).
QOA_func: callable(t) -> float, débit/respi oral (positif vers l'extérieur si défini).
QNA_func: callable(t) -> float, débit nasal.
phi_NM_func: callable(t) -> float, flux nasal → mucus (masse/temps).
Signature retournée:
Retourne un dict structuré pour le solveur:
"funcs_dict": mapping variable -> fonction dy/dt (signatures: func(y, t, **params)).
"params_dict": mapping variable -> dict des paramètres nécessaires à la fonction.
"y0_dict": mapping variable -> valeur initiale.
"event_funcs" (optionnel) : mapping d'événement -> fonction évènementielle qui prend
l'état actuel (y0_current) et params_current, et renvoie un nouveau dictionnaire d'états
(post-événement). Dans le modèle renvoyé, on fournit swallow et chew.
    """
    #  A = B*(VOP)^{2/3} => je trouve B avec les conditions initiales (A0 = k*B0 )
    #  Si on mache, on multiplie le B par quelque chose: le volume reste constant mais l'aire augmente
    # ------------------------------
    # Equations différentielles
    # ------------------------------
    # mouth ===========================
    # solid product-------------------
    def dVOP_dt(y, t, v_mouth, alpha, AOLP_coefficient,T_mouth):
        VOP = y["VOP"]
        T = y["T"]
        if VOP <= 0:
            return 0.0
        AOLP = AOLP_fun(t)
        # version linéaire
        v_eff = v_mouth * (1 + alpha * (T - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        return -v_eff * AOLP
        # liquid product -----------------
    # volume of saliva
    def dVOL_dt(y, t, QSaliva, v_mouth,alpha, AOLP_coefficient,T_mouth):
        T = y["T"]
        if y.get("VOL", 0.0) <= 0.0:
            return 0.0
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOP", 0.0) <= 0.0:
            # No product volume lost, only saliva is added
            return QSaliva
        AOLP = AOLP_fun(t)
        v_eff = v_mouth * (1 + alpha * (T - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        dVODP = v_eff * AOLP
        dVOS = QSaliva
        return dVODP + dVOS
    # volume of dissolved product
    # concentration in liquid phase
    def dCOL_dt(y, t,  kL, AOAL, KAL, COP, v_mouth,alpha, AOLP_coefficient,T_mouth):
        """
        d(VOL*COL)/dt = phi_OLP - phi_OAL
        phi_OLP = v * AOLP * COP
        phi_OAL = kL * AOAL * (COL - COALeta)
        """
        # pour simplification on prend VOL = VOS + VOPD
        T = y["T"]
        VOL = y["VOL"]
        if VOL <= 0.0:
            # Défensif : si le volume est nul, on évite la division et on garde la concentration
            # inchangée (dCOL/dt = 0). Cela évite NaN au démarrage quand VOS=VOPD=0.
            return 0.0
        COL, COA = y["COL"], y["COA"]  # concentration oral gaz
        # Check VOP before computing AOLP to avoid NaN with VOP**(2/3)
        if y.get("VOP", 0.0) <= 0.0:
            AOLP = 0.0
        else:
            AOLP = AOLP_fun(t)
        v_eff = v_mouth * (1 + alpha * (T - T_mouth))
        # sécurité
        v_eff = max(v_eff, 0)
        phi_OLP = v_eff * AOLP  * COP  # flux of diffusion of solid
        phi_OAL = kL * AOAL * (COL - COA / KAL)  # flux of diffusion liquid => gaz
        dVOL_dt = QSaliva + v_eff * AOLP 
        term_supp = COL * dVOL_dt  # terme de dilution/concentration
        return (phi_OLP - phi_OAL - term_supp) / VOL
    # concentration in oral cavity (gaz)
    def dCOA_dt(y, t, kL, AOAL, KAL, VOA):
        # QOA = QOA_func(t)
        COL, COA = y["COL"], y["COA"]
        phi_OAL = kL * AOAL * (COL - COA / KAL)
        # Flux oral vers pharynx selon le signe de QOA
        # flux_respi = QOA * (y["CFA"] - COA) if QOA >= 0 else 0
        return (phi_OAL) / VOA  # +flux_respi/VOA
    # Pharynx ==========================
    # Concentration aroma
    def dCFA_dt(y, t, AFAL, kL, VFA, QNA_func, KAL):
        # QOA = QOA_func(t)
        QNA = QNA_func(t)
        phi_FAL = AFAL * kL * (y["CFL"] - y["CFA"] / KAL)
        # Flux selon respiration
        # flux_air = (QOA * (y["COA"] - y["CFA"])) if QOA < 0 else 0
        flux_respi = (QNA * (y["CNA"] - y["CFA"])) if QNA >= 0 else (-QNA * (0 - y["CFA"]))
        return (phi_FAL + flux_respi) / VFA  # +flux_air/VFA
    # concentration product in pharynx
    def dCFL_dt(y, t, AFAL, kL, VFL,KAL):
        phi_FAL = - kL * AFAL * (y["CFL"] - y["CFA"] / KAL)
        return (phi_FAL) / VFL
    # Nose==================
    def dCNA_dt(y, t, QNA_func, VNA):
        QNA = QNA_func(t)
        # phi_NM = phi_NM_func(t)
        flux_respi = (QNA * (0 - y["CNA"])) if QNA >= 0 else (-QNA * (y["CFA"] - y["CNA"]))
        return (flux_respi) / VNA  # +phi_NM/flux_respi
    # mucosal adsorption
    # def dCNM_dt(y, t, phi_NM_func):
    #    phi_NM = phi_NM_func(t)
    #    return -phi_NM
    # PTRMS ===================
    def dCMS_dt(y, t, tMS):
        return (y["CNA"] - y["CMS"]) / tMS
    def dT_dt(y, t, kT, T_mouth=36):
        T = y["T"]
        return kT * (T_mouth - T)
    # ------------------------------
    # Dictionnaires pour le solveur
    # ------------------------------
    funcs_dict = {
        "VOP": dVOP_dt,
        "VOL": dVOL_dt,
        "COL": dCOL_dt,
        "COA": dCOA_dt,
        "CFA": dCFA_dt,
        "CFL": dCFL_dt,
        "CNA": dCNA_dt,
        "CMS": dCMS_dt,
        "T":dT_dt
    }
    params_dict = {
        "VOP": {"v_mouth":v_mouth,"alpha":alpha,"AOLP_coefficient": AOLP_coefficient,"T_mouth":T_mouth},
        "VOL": {"QSaliva": QSaliva, "v_mouth":v_mouth,"alpha":alpha,"AOLP_coefficient": AOLP_coefficient,"T_mouth":T_mouth},
        "COL": { "kL": kL, "AOAL": AOAL, "KAL": KAL, "COP": COP, "v_mouth":v_mouth,"alpha":alpha,"AOLP_coefficient": AOLP_coefficient,"T_mouth":T_mouth},
        "COA": {"kL": kL, "AOAL": AOAL, "KAL": KAL, "VOA": VOA},
        "CFA": {"AFAL": AFAL, "kL": kL, "VFA": VFA, "QNA_func": QNA_func, "KAL": KAL},
        "CFL": {"AFAL": AFAL, "kL": kL, "VFL": VFL,"KAL" : KAL},
        "CNA": {"QNA_func": QNA_func, "VNA": VNA},
        #    "CNM": {"phi_NM_func": phi_NM_func},
        "CMS": {"tMS": tMS},
        "T":{"kT": kT, "T_mouth":T_mouth}
    }

    def opening_velum(y0_current, params_current=None,params_model=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": y0_current["VOL"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["CFL"],
            "CNA": CFN,
            #   "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "T":y0_current["T"]
        }
  
    def chew(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
                * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "T": y0_current["T"],
        }
    def swallow_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": y0_current["VOP"],
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "T":y0_current["T"]
        }

    def total_step(y0_current, params_current=None):
        """
        Renvoie les dictionnaires pour la résolution d'une étape
        y_current : dict des conditions initiales
        """
        if params_current is None:
            raise ValueError("params_current cannot be None: it must contain at least 'VFA' and 'VNA'.")
        # Check that mandatory parameters are present
        missing = [p for p in ("VFA", "VNA", "VOL_m") if p not in params_current]
        if missing:
            raise KeyError(f"The following parameters are missing in params_current: {', '.join(missing)}")
        CFN = (y0_current["COA"] * params_current["VOA"] + y0_current["CFA"] * params_current["VFA"] + params_current["VNA"]
               * y0_current["CNA"]) / (params_current["VOA"] + params_current["VFA"] + params_current["VNA"])
        return {
            "VOP": 0,
            "VOL": params_current["VOL_m"],
            "COL": y0_current["COL"],
            "COA": CFN,
            "CFA": CFN,
            "CFL": y0_current["COL"],
            "CNA": CFN,
            # "CNM": y0_current["VOP"],
            "CMS": y0_current["CMS"],
            "T":y0_current["T"]
        }

    return Model(params_dict, funcs_dict, 
                 event_funcs={"swallow": swallow_step, "chew": chew, "total_swallow": total_step,"open_velum": opening_velum}, 
                 name="in_vivo_diffusion_gel_model")

