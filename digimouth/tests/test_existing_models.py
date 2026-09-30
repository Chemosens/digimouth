"""Catalogue de modèles (digimouth.existing_models) : construction, intégration, événements et
convention de noms des paramètres.

Convention actuelle :
  - modèle solution et modèles solides « with_temp* » : AOAL, AFAL, KAL, kL ;
  - modèle solide sans température : AOAP, AFAP, KOAL, kOL.
"""
import inspect

import numpy as np
import pytest

from digimouth import dynamic_model as dm
from digimouth import existing_models as exmod

RENAMED = [
    "in_vivo_diffusion_solution_model",
    "in_vivo_diffusion_solid_model_with_temp",
    "in_vivo_diffusion_solid_model_with_temp_and_surface",
    "in_vivo_diffusion_solid_model_with_temp_and_AOLP",
]
LEGACY_NAMES = {
    "in_vivo_diffusion_solid_model": {"AOAP", "AFAP", "KOAL", "kOL"},
}
OLD_NAMES = {"AOAP", "AFAP", "KOAL", "kOL", "KAP", "kp"}
NEW_NAMES = {"AOAL", "AFAL", "KAL", "kL"}


def breathing(t):
    """Oscille autour de 0 : exerce les branches inspiration et expiration."""
    return 0.02 * np.sin(2 * np.pi * t / 3.0)


# Valeurs (SI) pour chaque nom d'argument connu ; un nom inconnu fait échouer le test,
# ce qui signale qu'une signature a changé et que ce fichier doit suivre.
VALUES = dict(
    tMS=1.0, QSaliva=5e-2, VOA=4e-5, VFA=3e-5, VNA=1.1e-5, VFL=1e-6, COP=1e3,
    AOAL=1e-4, AFAL=6e-5, KAL=1e-3, kL=1e-6, AOAP=1e-4, AFAP=6e-5, KOAL=1e-3, kOL=1e-6,
    v=1e-4, AOLP_coefficient=8e4, kT=0.5, T_mouth=36.0, v_mouth=2e-8, alpha=0.05,
    QNA_func=breathing, AOLP_fun=lambda t: 1e-4,
)
LIQUID_VALUES = dict(VALUES, KAL=2e-3, kL=0.14)
Y0 = dict(T=20.0, TP=20.0, SP=5e4, VOP=1e-7, VOL=1e-8, COL=0.0, COA=0.0, CFA=0.0, CFL=0.0, CNA=0.0, CMS=0.0)


def build(name):
    func = getattr(exmod, name)
    values = LIQUID_VALUES if name == "in_vivo_diffusion_solution_model" else VALUES
    return func(**{arg: values[arg] for arg in inspect.signature(func).parameters})


def initial_state(model, name):
    y0 = {var: Y0.get(var, 1e-3) for var in model.funcs_dict}
    if name == "in_vivo_diffusion_solution_model":
        y0.update(VOL=1e-6, COL=2.0)
    return y0


@pytest.mark.unit
@pytest.mark.parametrize("name", RENAMED + ["in_vivo_diffusion_solid_model"])
def test_model_integrates_to_finite_values(name):
    model = build(name)
    df = dm.resolution_odeint_generic(model, y0_dict=initial_state(model, name), t_sim=np.linspace(0, 20, 120))
    assert set(df.columns) == set(model.funcs_dict) | {"time"}
    assert np.isfinite(df.to_numpy(dtype=float)).all()
    assert df["CMS"].min() > -1e-9          # une concentration mesurée n'est jamais (sensiblement) négative


@pytest.mark.unit
@pytest.mark.parametrize("name", RENAMED)
def test_renamed_models_use_the_new_parameter_names(name):
    args = set(inspect.signature(getattr(exmod, name)).parameters)
    assert NEW_NAMES <= args
    assert not (OLD_NAMES & args)


@pytest.mark.unit
@pytest.mark.parametrize("name, expected", sorted(LEGACY_NAMES.items()))
def test_other_models_keep_their_historical_names(name, expected):
    args = set(inspect.signature(getattr(exmod, name)).parameters)
    assert expected <= args
    assert not (NEW_NAMES & args)


@pytest.mark.unit
@pytest.mark.parametrize("name", RENAMED + ["in_vivo_diffusion_solid_model"])
def test_params_dict_names_are_arguments_of_their_equation(name):
    """generic_model retrouve les paramètres par NOM : chaque clé de params_dict doit être un argument."""
    model = build(name)
    assert set(model.params_dict) == set(model.funcs_dict)
    for var, params in model.params_dict.items():
        arguments = set(inspect.signature(model.funcs_dict[var]).parameters)
        assert set(params) <= arguments, (var, set(params) - arguments)


@pytest.mark.unit
def test_solution_swallow_mixes_the_gases_and_resets_the_liquid():
    model = build("in_vivo_diffusion_solution_model")
    state = dict(VOL=0.5, COA=1.0, COL=2.0, CFA=3.0, CFL=4.0, CNA=5.0, CMS=6.0)
    params = dict(VOA=4e-5, VFA=3e-5, VNA=1.1e-5, VOL_m=1e-3)
    out = model.event_funcs["swallow"](dict(state), params_current=params)
    assert set(out) == set(state)
    assert out["VOL"] == params["VOL_m"]                       # le liquide est ramené au volume résiduel
    assert out["CFL"] == state["COL"] and out["COL"] == state["COL"]
    assert out["COA"] == out["CFA"] == out["CNA"]              # les trois compartiments gazeux sont mélangés
    before = state["COA"] * params["VOA"] + state["CFA"] * params["VFA"] + state["CNA"] * params["VNA"]
    after = out["COA"] * params["VOA"] + out["CFA"] * params["VFA"] + out["CNA"] * params["VNA"]
    assert after == pytest.approx(before)                      # le mélange conserve la masse gazeuse
    assert out["CMS"] == state["CMS"]                          # l'instrument n'est pas touché


@pytest.mark.unit
def test_solution_jaw_move_keeps_the_liquid():
    model = build("in_vivo_diffusion_solution_model")
    state = dict(VOL=0.5, COA=1.0, COL=2.0, CFA=3.0, CFL=4.0, CNA=5.0, CMS=6.0)
    out = model.event_funcs["jaw_move"](dict(state), params_current=dict(VOA=4e-5, VFA=3e-5, VNA=1.1e-5, VOL_m=1e-3))
    assert (out["VOL"], out["COL"], out["CFL"], out["CMS"]) == (state["VOL"], state["COL"], state["CFL"], state["CMS"])
    assert out["COA"] == out["CFA"] == out["CNA"]


@pytest.mark.unit
def test_solid_events_preserve_the_set_of_state_variables():
    model = build("in_vivo_diffusion_solid_model_with_temp_and_surface")
    state = {var: Y0.get(var, 1e-3) for var in model.funcs_dict}
    params = dict(VOA=4e-5, VFA=3e-5, VNA=1.1e-5, VOL_m=1e-3, chew_factor=1.25)
    assert set(model.event_funcs) >= {"swallow", "chew"}
    for event, handler in model.event_funcs.items():
        assert set(handler(dict(state), params_current=params)) == set(state), event
