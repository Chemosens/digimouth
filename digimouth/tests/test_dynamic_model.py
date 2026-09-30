"""Moteur d'ODE (digimouth.dynamic_model) sur un problème dont on connaît la solution :
la décroissance exponentielle dy/dt = -k y, y(t) = y0 exp(-k t)."""
import numpy as np
import pandas as pd
import pytest

from digimouth import dynamic_model as dm

K = 0.5


def decay_model(k=K):
    return dm.Model(
        {"CMS": {"k": k}},
        {"CMS": lambda y, t, k: -k * y["CMS"]},
        event_funcs={"reset": lambda y, params_current=None: {"CMS": 1.0}},
        name="decay",
    )


@pytest.mark.unit
def test_odeint_matches_the_analytic_solution():
    t = np.linspace(0, 5, 51)
    df = dm.resolution_odeint_generic(decay_model(), {"CMS": 1.0}, t_sim=t)
    assert list(df.columns) == ["time", "CMS"]
    assert np.allclose(df["time"], t)
    assert np.abs(df["CMS"] - np.exp(-K * t)).max() < 1e-5


@pytest.mark.unit
def test_solve_ivp_agrees_with_odeint():
    t = np.linspace(0, 5, 51)
    a = dm.resolution_odeint_generic(decay_model(), {"CMS": 1.0}, t_sim=t)
    b = dm.resolution_odeint_generic(decay_model(), {"CMS": 1.0}, t_sim=t, solver="solve_ivp")
    assert np.allclose(a["CMS"], b["CMS"], atol=1e-5)


@pytest.mark.unit
def test_inconsistent_initial_conditions_are_rejected():
    with pytest.raises(ValueError, match="incoh"):
        dm.resolution_odeint_generic(decay_model(), {"AUTRE": 1.0}, t_sim=np.linspace(0, 1, 5))


@pytest.mark.unit
def test_run_model_applies_an_instantaneous_event():
    model = decay_model()
    events = pd.DataFrame({"time": [2.0], "type": ["reset"]})
    df = dm.run_model(model, {"CMS": 0.3}, t_start=0, t_end=5, df_event=events,
                      event_func_dict=model.event_funcs, n_points=60)
    assert df["time"].is_monotonic_increasing
    # la ligne ajoutée à l'instant de l'événement porte l'état remis à 1
    assert df.loc[df["time"] == 2.0, "CMS"].iloc[-1] == 1.0
    # ensuite l'état décroît depuis 1 pendant les 3 s restantes
    assert df["CMS"].iloc[-1] == pytest.approx(np.exp(-K * 3), rel=1e-4)


@pytest.mark.unit
def test_unknown_event_is_reported():
    model = decay_model()
    events = pd.DataFrame({"time": [1.0], "type": ["inexistant"]})
    with pytest.raises(ValueError, match="inconnu"):
        dm.run_model(model, {"CMS": 1.0}, t_start=0, t_end=2, df_event=events,
                     event_func_dict=model.event_funcs, n_points=20)


@pytest.mark.unit
def test_run_stages_chains_stages_without_a_gap():
    model = decay_model()
    stages = [{"t_start": 0, "t_end": 2, "model": model}, {"t_start": 2, "t_end": 5, "model": model}]
    df = dm.run_stages(stages, {"CMS": 1.0}, n_points=40)
    assert df["time"].is_monotonic_increasing
    assert df["time"].iloc[0] == 0 and df["time"].iloc[-1] == 5
    assert df["CMS"].iloc[-1] == pytest.approx(np.exp(-K * 5), rel=1e-4)


@pytest.mark.unit
def test_model_keeps_the_dictionary_style_access():
    model = decay_model()
    assert model["params_dict"] is model.params_dict
    assert model["funcs_dict"] is model.funcs_dict
    assert "event_funcs" in model
