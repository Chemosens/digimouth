"""Ajustement de paramètres (digimouth.optimization) sur des données synthétiques dont on connaît le
paramètre vrai : décroissance exponentielle y = exp(-k t) avec k = 0,5."""
import numpy as np
import pytest

from digimouth import dynamic_model as dm
from digimouth import optimization as opt

K_TRUE = 0.5
T = np.linspace(0, 10, 60)
Y = np.exp(-K_TRUE * T)


def decay_model(k):
    return dm.Model({"CMS": {"k": k}}, {"CMS": lambda y, t, k: -k * y["CMS"]}, name="decay")


@pytest.mark.unit
def test_residuals_vanish_at_the_true_parameter():
    at_truth = opt.calculate_residuals([K_TRUE], T, Y, {"CMS": 1.0}, decay_model(0.2), ["k"], normalization=False)
    elsewhere = opt.calculate_residuals([0.2], T, Y, {"CMS": 1.0}, decay_model(0.5), ["k"], normalization=False)
    assert len(at_truth) == len(T)
    assert np.abs(at_truth).max() < 1e-5
    assert np.abs(elsewhere).max() > 0.1


@pytest.mark.unit
def test_diagnostic_plot_works_for_any_model(tmp_path):
    """Le tracé de diagnostic (save_path) ne suppose plus un modèle de convection (Cg, Kaw, tMS) :
    son titre affiche les paramètres ajustés, quels qu'ils soient."""
    import matplotlib
    matplotlib.use("Agg")
    figure = tmp_path / "fit.png"
    with_plot = opt.calculate_residuals([K_TRUE], T, Y, {"CMS": 1.0}, decay_model(0.2), ["k"],
                                        normalization=False, save_path=str(figure))
    without_plot = opt.calculate_residuals([K_TRUE], T, Y, {"CMS": 1.0}, decay_model(0.2), ["k"],
                                           normalization=False)
    assert figure.is_file()
    assert np.array_equal(with_plot, without_plot)          # le tracé ne change pas les résidus
    opt.calculate_residuals_multi([K_TRUE], [T, T], [Y, Y], {"CMS": 1.0}, decay_model(0.2), ["k"],
                                  normalization=False, save_path=str(tmp_path / "multi"))
    assert (tmp_path / "multi_0").exists() or (tmp_path / "multi_0.png").exists()


@pytest.mark.unit
def test_fit_recovers_the_decay_rate():
    fit = opt.optim_param_dynamic(T, Y, {"CMS": 1.0}, decay_model(0.2), param_keys_to_optimize=["k"],
                                  initial_params_to_optimize=[0.3], bounds=([0.01], [5.0]), normalization=False)
    assert fit["opt_param"][0] == pytest.approx(K_TRUE, abs=1e-3)
    assert fit["rmse"] < 1e-4
    assert {"opt_param", "ssr", "rmse", "opt_params_dict", "opt_model", "stderr", "conf_intervals"} <= set(fit)


@pytest.mark.unit
def test_fit_stops_at_the_bounds():
    fit = opt.optim_param_dynamic(T, Y, {"CMS": 1.0}, decay_model(0.2), param_keys_to_optimize=["k"],
                                  initial_params_to_optimize=[0.1], bounds=([0.01], [0.3]), normalization=False)
    assert fit["opt_param"][0] == pytest.approx(0.3, abs=1e-6)     # la vraie valeur (0,5) est hors bornes
