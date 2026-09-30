"""Fonctions utilitaires (digimouth.utils)."""
import numpy as np
import pandas as pd
import pytest

from digimouth import utils


@pytest.mark.unit
def test_r2_score_matches_the_textbook_formula():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    assert utils.r2_score(y_true, y_pred) == pytest.approx(0.9486081370449679)   # valeur de référence de scikit-learn
    assert utils.r2_score(y_true, y_true) == 1.0
    # Données constantes : même convention que scikit-learn (1 si prédiction parfaite, 0 sinon)
    assert utils.r2_score(np.ones(3), np.ones(3)) == 1.0
    assert utils.r2_score(np.ones(3), np.array([1.0, 2.0, 1.0])) == 0.0


@pytest.mark.unit
def test_staircase_multiplies_by_the_factor_at_each_chew():
    chews = np.array([1.0, 2.0, 3.0])
    values = [utils.escalier_func(t, chews, multiplier=1.05) for t in (0.5, 1.5, 2.5, 3.5)]
    assert values == pytest.approx([1.0, 1.05, 1.05 ** 2, 1.05 ** 3])


@pytest.mark.unit
def test_swallows_are_moved_to_the_next_zero_crossing_of_the_breath():
    time = np.linspace(0, 10, 100)
    breath = pd.DataFrame({"time": time, "intensity": np.sin(time * 2 * np.pi / 5)})   # période 5 s
    swallows = pd.Series([2.0, 4.0, 8.0])
    adjusted = utils.adjust_swallow_with_breath(breath, swallows, adjust="next_zero", rising_factor=0.66)
    assert len(adjusted) == len(swallows)
    assert (adjusted.to_numpy() >= swallows.to_numpy()).all()       # « next » : jamais avant l'instant d'origine
    assert adjusted.is_monotonic_increasing
