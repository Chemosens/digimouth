"""Ajustement d'un protocole (digimouth.optimization.optim_param_protocol) : phases successives et
événements instantanés, sur des données synthétiques dont on connaît les paramètres vrais
(décroissance exponentielle dC/dt = -k C)."""
import numpy as np
import pandas as pd
import pytest

from digimouth import dynamic_model as dm
from digimouth import optimization as opt


def decay_model(k):
    return dm.Model({"CMS": {"k": k}}, {"CMS": lambda y, t, k: -k * y["CMS"]},
                    event_funcs={"refill": lambda y, params_current=None: {"CMS": 1.0}}, name="decay")


T = np.linspace(0, 10, 81)


@pytest.mark.unit
def test_without_events_it_agrees_with_optim_param_dynamic():
    y = np.exp(-0.5 * T)
    reference = opt.optim_param_dynamic(T, y, {"CMS": 1.0}, decay_model(0.2), param_keys_to_optimize=["k"],
                                        initial_params_to_optimize=[0.3], bounds=([0.01], [5.0]),
                                        normalization=False)
    fit = opt.optim_param_protocol(T, y, decay_model(0.2), {"CMS": 1.0}, params={"k": 0.3},
                           bounds={"k": (0.01, 5.0)}, normalization=False, n_points=400)
    assert fit["success"]
    assert fit["opt_params"]["k"] == pytest.approx(reference["opt_param"][0], abs=1e-3)
    assert set(fit["conf_intervals"]) == {"k"}
    low, high = fit["conf_intervals"]["k"]
    assert low < fit["opt_params"]["k"] < high


@pytest.mark.unit
def test_a_parameter_can_differ_between_stages():
    stages = [{"name": "before", "t_start": 0, "t_end": 5, "model": decay_model(0.5)},
              {"name": "after", "t_start": 5, "t_end": 10, "model": decay_model(0.1)}]
    truth = dm.run_stages(stages, {"CMS": 1.0}, n_points=400).drop_duplicates("time", keep="last")
    y = np.interp(T, truth["time"], truth["CMS"])
    start = [{**st, "model": decay_model(0.3)} for st in stages]
    fit = opt.optim_param_protocol(T, y, start, {"CMS": 1.0}, params={"k@before": 0.3, "k@after": 0.3},
                           bounds={"k@before": (0.01, 5), "k@after": (0.01, 5)}, normalization=False, n_points=400)
    assert fit["opt_params"]["k@before"] == pytest.approx(0.5, abs=1e-3)
    assert fit["opt_params"]["k@after"] == pytest.approx(0.1, abs=1e-3)
    assert fit["opt_stages"][1]["model"].params_dict["CMS"]["k"] == pytest.approx(0.1, abs=1e-3)


@pytest.mark.unit
def test_a_parameter_without_suffix_is_shared_by_all_stages():
    y = np.exp(-0.5 * T)
    stages = [{"name": "a", "t_start": 0, "t_end": 5, "model": decay_model(0.2)},
              {"name": "b", "t_start": 5, "t_end": 10, "model": decay_model(0.9)}]
    fit = opt.optim_param_protocol(T, y, stages, {"CMS": 1.0}, params={"k": 0.3}, bounds={"k": (0.01, 5)},
                           normalization=False, n_points=400)
    assert fit["opt_params"]["k"] == pytest.approx(0.5, abs=1e-3)
    assert all(st["model"].params_dict["CMS"]["k"] == pytest.approx(0.5, abs=1e-3) for st in fit["opt_stages"])


@pytest.mark.unit
def test_events_are_taken_into_account():
    model = decay_model(0.5)
    events = pd.DataFrame({"time": [3.0, 7.0], "type": ["refill", "refill"]})
    truth = dm.run_model(model, {"CMS": 1.0}, t_start=0, t_end=10, df_event=events,
                         event_func_dict=model.event_funcs, n_points=400).drop_duplicates("time", keep="last")
    y = np.interp(T, truth["time"], truth["CMS"])
    fit = opt.optim_param_protocol(T, y, decay_model(0.2), {"CMS": 1.0}, params={"k": 0.3}, bounds={"k": (0.01, 5)},
                           events=events, normalization=False, n_points=400)
    assert fit["opt_params"]["k"] == pytest.approx(0.5, abs=1e-3)
    assert fit["rmse"] < 1e-3
    # sans les événements, le même ajustement ne peut pas reproduire la courbe
    without = opt.optim_param_protocol(T, y, decay_model(0.2), {"CMS": 1.0}, params={"k": 0.3}, bounds={"k": (0.01, 5)},
                               normalization=False, n_points=400)
    assert without["rmse"] > 10 * fit["rmse"]


@pytest.mark.unit
def test_several_experiments_share_the_parameters():
    y1 = np.exp(-0.5 * T)
    y2 = 2.0 * np.exp(-0.5 * T)                    # même cinétique, autre condition initiale
    fit = opt.optim_param_protocol([T, T], [y1, y2], decay_model(0.2), [{"CMS": 1.0}, {"CMS": 2.0}],
                           params={"k": 0.3}, bounds={"k": (0.01, 5)}, normalization=False, n_points=400)
    assert fit["opt_params"]["k"] == pytest.approx(0.5, abs=1e-3)


@pytest.mark.unit
def test_unknown_parameter_or_stage_is_reported():
    stages = [{"name": "a", "t_start": 0, "t_end": 10, "model": decay_model(0.2)}]
    y = np.exp(-0.5 * T)
    with pytest.raises(ValueError, match="not a parameter"):
        opt.optim_param_protocol(T, y, stages, {"CMS": 1.0}, params={"kx": 0.3})
    with pytest.raises(ValueError, match="no stage named"):
        opt.optim_param_protocol(T, y, stages, {"CMS": 1.0}, params={"k@z": 0.3})
    with pytest.raises(ValueError, match="within the stages"):
        opt.optim_param_protocol(np.linspace(0, 20, 5), np.ones(5), stages, {"CMS": 1.0}, params={"k": 0.3})
