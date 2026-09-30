"""Tests de digimouth.graphical_methods.plot_xy, avec de vraies assertions
(contrairement aux scripts d'exploration de ce dossier, ignorés par
conftest.py). Backend matplotlib forcé en Agg pour ne jamais tenter d'ouvrir
une fenêtre (plt.show()/backend Tk indisponible sur certaines machines)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from digimouth import graphical_methods as gmet


@pytest.fixture(autouse=True)
def cleanup():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# --- Basic functionality tests ---
def test_plot_xy_simple_vectors(show=False, save_path=None, new=True):
    """Test basic x, y vector plotting."""

    x = [0, 1, 2, 3]
    y = [0, 1, 4, 9]
    fig, ax = gmet.plot_xy(x, y, show=show, save_path=save_path, new=new)
    assert fig is not None
    assert ax is not None
    assert len(ax.lines) > 0


def test_plot_xy_simple_vectors_reuses_an_existing_ax():
    """new=False on a fresh call still returns a usable ax (no crash)."""
    fig, ax = gmet.plot_xy([0, 1, 2, 3], [0, 1, 4, 9], show=False, new=False)
    assert fig is not None
    assert len(ax.lines) > 0


def test_plot_xy_dict_input(show=False, new=False, ax=None):
    """Test plotting with dict inputs (x and y both dicts)."""
    x = {"a": [0, 1, 2], "b": [0, 1, 2]}
    y = {"a": [0, 1, 4], "b": [0, 2, 4]}
    fig, ax = gmet.plot_xy(x, y, show=show, new=new, ax=ax)
    assert fig is not None
    assert len(ax.lines) == 2  # Two series plotted


def test_plot_xy_dict_y_scalar_x():
    """Test dict y with scalar x (broadcasted to all keys)."""
    x = [0, 1, 2]
    y = {"series1": [0, 1, 4], "series2": [0, 2, 4]}
    fig, ax = gmet.plot_xy(x, y, show=False)
    assert fig is not None
    assert len(ax.lines) == 2


def test_plot_xy_sequence_of_vectors():
    """Test plotting list of y vectors with single x."""
    x = [0, 1, 2]
    y = [[0, 1, 4], [0, 2, 4], [1, 1, 1]]
    fig, ax = gmet.plot_xy(x, y, show=False)
    assert fig is not None
    assert len(ax.lines) == 3


def test_plot_xy_sequence_of_vectors_with_sequence_x():
    """Test multiple x and y vectors (paired)."""
    x = [[0, 1, 2], [0, 1, 2, 3]]
    y = [[0, 1, 4], [0, 1, 4, 9]]
    fig, ax = gmet.plot_xy(x, y, show=False)
    assert fig is not None
    assert len(ax.lines) == 2


def test_plot_xy_without_plot_for_several_graph_on_the_same_ax():
    """Test plotting multiple series on same ax without creating new plot."""
    x = [0, 1, 2]
    y1 = [0, 1, 4]
    y2 = [0, 2, 4]
    fig, ax = gmet.plot_xy(x, y1, show=False)
    # new=False est requis pour réutiliser l'ax fourni : par défaut (new=True)
    # plot_xy ouvre toujours une nouvelle figure, même si ax est passé
    # (cf. sa docstring).
    fig, ax = gmet.plot_xy(x, y2, show=False, ax=ax, new=False)
    assert fig is not None
    assert len(ax.lines) == 2
