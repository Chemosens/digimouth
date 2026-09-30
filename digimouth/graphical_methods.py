import numpy as np
import matplotlib.pyplot as plt

def plot_xy(
    x, y,
    vertical_lines=None,
    opt_x=None,
    opt_y=None,
    # `new` controls creation of a fresh Figure/Axis when an `ax` is passed.
    # If new=False the function will plot on the supplied axis (or current
    # axes when `ax` is None) without opening a new figure. Default is True.
    y_type="l",                 # "l"=ligne, "p"=points, "b"=both
    opt_type="p",
    y_col="blue",
    opt_col="green",
    vertical_col="red",
    title="Graph XY",
    x_lim=None,
    y_lim=None,
    x_label=None,
    y_label=None,
    opt_label="opt(x)",
    plotly=False,
    ax=None,
    new=True,
    show=None,
    save_path=None,
    dpi=300,
    legend=True,width=1920,height=1080):
        # determine whether to show the figure based on backend if show is None
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import cycle
    backend = plt.get_backend().lower() 
    interactive_backends = {"tkagg", "qt5agg", "qt4agg", "macosx", "gtk3agg"}
    if show is None:
        show_flag = (backend in interactive_backends)
    else:
        show_flag = bool(show)
    def plot_with_type(xd, yd, plot_type, color, label):
        style = {"l": "-", "p": "o", "b": "-o"}[plot_type]
        ax.plot(xd, yd, style, color=color, label=label)
    def make_color_map(keys, base):
        if isinstance(base, dict):
            return {k: base.get(k, "C0") for k in keys}
        if isinstance(base, (list, tuple)):
            cols = list(base)
            if len(cols) < len(keys):
                cols = list(cycle(cols))[: len(keys)]
            return {k: cols[i] for i, k in enumerate(keys)}
        return {k: base for k in keys}   
    internal_fig = False
    fig = None
    # determine whether we need to create a fresh figure/axis
    if width is not None and height is not None:
        figsize = (width / dpi, height / dpi)
    else:
        figsize = (8, 5)
    if new or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        internal_fig = True
    else:
        # reuse provided axis or current one
        if ax is None:
            ax = plt.gca()
        fig = ax.figure
    ax.set_title(title)
    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label)
    # --- y is dict mode (preferred) ---
    if isinstance(y, dict):
        keys = list(y.keys())
        # ensure x mapping
        if isinstance(x, dict):
            if set(x.keys()) != set(keys):
                raise ValueError("When x and y are dicts they must have the same keys.")
            x_map = x
        else:
            x_map = {k: x for k in keys}
        y_colors = make_color_map(keys, y_col)
        for k in keys:
            xi = np.asarray(x_map[k]) if x_map[k] is not None else np.array([])
            yi = np.asarray(y[k]) if y[k] is not None else np.array([])
            plot_with_type(xi, yi, y_type, y_colors[k], label=str(k))
    # --- y is sequence of series (list/tuple/ndarray of 1D arrays) ---
    else:
        # detect list-of-vectors
        is_seq_of_vecs = isinstance(y, (list, tuple)) and all(hasattr(el, "__len__") for el in y)
        if is_seq_of_vecs:
            y_list = list(y)
            # x is a list of one x-vector per series (paired) only if each of
            # its elements is itself a sequence — otherwise a same-length x
            # (e.g. one point per series) would wrongly be treated as
            # per-series x-scalars instead of a single shared x-axis.
            is_x_seq_of_vecs = (
                isinstance(x, (list, tuple)) and len(x) == len(y_list)
                and all(hasattr(xi, "__len__") for xi in x)
            )
            if is_x_seq_of_vecs:
                x_list = list(x)
            else:
                x_list = [x] * len(y_list)
            colors = make_color_map(range(len(y_list)), y_col)
            for i, (xi, yi) in enumerate(zip(x_list, y_list)):
                plot_with_type(np.asarray(xi), np.asarray(yi), y_type, colors[i], label=f"y[{i}]")
        else:
            # simple vector
            plot_with_type(np.asarray(x), np.asarray(y), y_type, y_col, "y(x)")
    #if opt_y is not None:
    # determine x used for opt
    if opt_x is None:
        if isinstance(x, dict):
            opt_x = next(iter(x.values()))
        elif isinstance(x, (list, tuple)):
            opt_x = x[0]
        else:
            opt_x = x
    # opt_y dict mode
    if opt_y is None:
        pass
    elif isinstance(opt_y, dict):
        opt_colors = make_color_map(opt_y.keys(), opt_col)
        for k, oy in opt_y.items():
            if isinstance(opt_x, dict):
                xi = opt_x.get(k, next(iter(opt_x.values())))
            else:
                xi = opt_x
            plot_with_type(
                np.asarray(xi),
                np.asarray(oy),
                opt_type,
                opt_colors[k],
                label=f"{opt_label}_{k}",
            )
    # opt_y list of curves
    elif isinstance(opt_y, (list, tuple)):
        colors = make_color_map(range(len(opt_y)), opt_col)
        for i, oy in enumerate(opt_y):
            if isinstance(opt_x, (list, tuple)):
                xi = opt_x[i]
            else:
                xi = opt_x
            plot_with_type(
                np.asarray(xi),
                np.asarray(oy),
                opt_type,
                colors[i],
                label=f"{opt_label}[{i}]",
            )
    # opt_y single vector
    else:
        plot_with_type(
            np.asarray(opt_x),
            np.asarray(opt_y),
            opt_type,
            opt_col,
            opt_label,
        )
    # vertical lines and final touches
    if vertical_lines is not None:
        for i, v in enumerate(vertical_lines):
            label = f"x={v:.2f}" if i == 0 else None
            ax.axvline(x=v, color=vertical_col, linestyle="--", alpha=0.7, label=label)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.grid(True, linestyle=":")
    if legend:
        ax.legend()
    # save if requested
    if save_path is not None:
        try:
            fig.savefig(save_path,dpi=dpi)
            plt.close(fig)     # close after saving to free resources
        except Exception:
            pass
    # show only when interactive and allowed
    if show_flag and internal_fig and not plotly:
        plt.show()
    # close figure to free resources when requested
    return fig, ax


def plot_grid(
    x_list,
    y_list,
    titles=None,
    ncols=5,
    figsize=None,
    sharex=False,
    sharey=False,
    **plot_kwargs,
):
    """Display multiple x/y series arranged in a grid of subplots.

    Parameters
    ----------
    x_list : sequence
        List of x vectors (can be the same object repeated if all curves share
        the same abscissa).
    y_list : sequence
        List of y vectors or sequences. Length must equal len(x_list).
    titles : sequence, optional
        Titles for each subplot; if provided its length should match the data
        lists.
    ncols : int, default 5
        Number of columns in the grid. Rows are computed automatically.
    figsize : tuple, optional
        Figure size (width, height); if None a sensible default is chosen.
    sharex, sharey : bool
        Whether to share axes between plots.
    **plot_kwargs : dict
        Passed through to :func:`plot_xy` for each subplot (e.g. ``y_type``,
        ``opt``...).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray
        Array of subplot axes (flattened if more than one row).
    """
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list must have the same length")
    nseries = len(x_list)
    nrows = int(np.ceil(nseries / ncols))
    if figsize is None:
        figsize = (ncols * 3, nrows * 2.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             sharex=sharex, sharey=sharey)
    axes = np.atleast_1d(axes).flatten()
    for idx, (xi, yi) in enumerate(zip(x_list, y_list)):
        ax = axes[idx]
        fig_i, _ = plot_xy(xi, yi, ax=ax, **plot_kwargs)
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx])
    # hide any unused axes
    for j in range(nseries, axes.size):
        axes[j].axis('off')
    fig.tight_layout()
    return fig, axes

