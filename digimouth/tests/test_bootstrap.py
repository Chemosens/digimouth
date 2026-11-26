"""Bootstrap of the residuals (digimouth.optimization.bootstrap_params): independent ("iid"),
block ("block") and parametric autoregressive ("ar1") resampling, on synthetic exponential
decays y = exp(-k t) whose noise is correlated in time."""
import numpy as np
import pytest

from digimouth import dynamic_model as dm
from digimouth import optimization as opt

K_TRUE = 0.5
T = np.linspace(0, 10, 200)


def decay_model(k):
    return dm.Model({"CMS": {"k": k}}, {"CMS": lambda y, t, k: -k * y["CMS"]}, name="decay")


def ar1_noise(rng, rho, sd, n):
    noise = np.empty(n)
    noise[0] = rng.normal(0, sd)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + rng.normal(0, sd * np.sqrt(1 - rho ** 2))
    return noise


def boot(y, resampling, n_boot=150, **kwargs):
    return opt.bootstrap_params(T, y, {"CMS": 1.0}, decay_model(0.3), ["k"], [0.3],
                                bounds=([0.01], [5.0]), normalization=False, n_boot=n_boot,
                                random_state=1, resampling=resampling, **kwargs)


@pytest.fixture(scope="module")
def correlated_curve():
    rng = np.random.default_rng(0)
    return np.exp(-K_TRUE * T) + ar1_noise(rng, 0.8, 0.02, T.size)


@pytest.mark.unit
def test_correlated_resampling_widens_the_intervals(correlated_curve):
    iid = boot(correlated_curve, "iid")
    block = boot(correlated_curve, "block", block_length=15)
    ar1 = boot(correlated_curve, "ar1")
    assert ar1["rho"] == pytest.approx(0.8, abs=0.15)
    # with rho = 0.8 the independent bootstrap underestimates the spread by about 3
    assert ar1["stderr_boot"][0] > 2 * iid["stderr_boot"][0]
    assert block["stderr_boot"][0] > 1.5 * iid["stderr_boot"][0]
    assert block["block_length"] == 15 and iid["block_length"] is None


@pytest.mark.unit
def test_ar1_spread_matches_the_true_spread():
    """Spread of k over many curves with fresh AR(1) noise vs the AR(1) bootstrap of one curve."""
    rng = np.random.default_rng(3)
    fits = []
    for _ in range(150):
        y = np.exp(-K_TRUE * T) + ar1_noise(rng, 0.8, 0.02, T.size)
        fits.append(opt.optim_param_dynamic(T, y, {"CMS": 1.0}, decay_model(0.3), ["k"], [0.3],
                                            bounds=([0.01], [5.0]), normalization=False)["opt_param"][0])
    true_sd = np.std(fits, ddof=1)
    ar1 = boot(np.exp(-K_TRUE * T) + ar1_noise(rng, 0.8, 0.02, T.size), "ar1", n_boot=200)
    assert ar1["stderr_boot"][0] == pytest.approx(true_sd, rel=0.5)


@pytest.mark.unit
def test_bootstrap_works_on_the_normalised_scale(correlated_curve):
    """With normalization=True the noise is built on the normalised scale, whatever the
    amplitude of the raw signal: the spread of k must not depend on that amplitude."""
    def run(scale):
        return opt.bootstrap_params(T, scale * correlated_curve, {"CMS": 1.0}, decay_model(0.3), ["k"], [0.3],
                                    bounds=([0.01], [5.0]), normalization=True, n_boot=60,
                                    random_state=2, resampling="block", block_length=10)["stderr_boot"][0]
    assert run(1000.0) == pytest.approx(run(1.0), rel=1e-3)
    assert run(1000.0) > 1e-3


@pytest.mark.unit
def test_bootstrap_is_reproducible_and_checks_its_arguments(correlated_curve):
    a = boot(correlated_curve, "block", n_boot=20, block_length=10)
    b = boot(correlated_curve, "block", n_boot=20, block_length=10)
    assert np.array_equal(a["samples"], b["samples"])
    with pytest.raises(ValueError, match="resampling"):
        boot(correlated_curve, "blocks", n_boot=2)
    with pytest.raises(ValueError, match="block_length"):
        boot(correlated_curve, "block", n_boot=2, block_length=0)
