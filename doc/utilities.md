# Utilities

## Signal helpers (`digimouth.utils`)

| Function | Purpose |
|---|---|
| `r2_score(y_true, y_pred)` | Coefficient of determination R² (same results as scikit-learn, without depending on it) |
| `compare_curves(t1, x1, t2, x2, time_window=(0, 120), norm=True)` | Interpolates curve 2 on the times of curve 1 and returns `(r2, rmse, residuals)`: R² and RMSE on the time window, residuals on all times of curve 1. **Also opens a plot** (see [design_principles.md](design_principles.md)) |
| `adjust_swallow_with_breath(breath_eval, t_degs, adjust="next_zero", rising_factor=0.66, display_graph=False)` | Moves swallow times onto the zero crossings of the breathing signal |
| `detect_start(f_sel, ion=..., method="max_intensity_ratio", ...)` | Detects the start of the release peak of one ion in one PTR-MS file |
| `recale_time_on_peak(df, file_col="file", time_col="time", intensity_col="intensity", ...)` | Shifts the time axis of each file so that t = 0 is the start of its peak |
| `escalier_func(t, chew_times, multiplier=1.05)` | Step function multiplied by `multiplier` at each chew (used for chewing-driven surfaces) |
| `AOLP_func_param(t, chew_times, r0, v, A0, multiplier=1.05)` | Product surface as a function of time and chews |

Expected data layout: tables are pandas DataFrames with a `time` column and an `intensity`
column (breathing signal, aroma signal). Swallow times are a pandas Series.

## Plotting helpers (`digimouth.graphical_methods`)

| Function | Purpose |
|---|---|
| `plot_xy(x, y, vertical_lines=None, opt_x=None, opt_y=None, ...)` | One curve, optionally with a second series (e.g. data points) and vertical lines (events) |
| `plot_grid(x_list, y_list, titles=None, ncols=5, ...)` | Several curves in a grid of subplots; returns `(fig, axes)` |

These helpers only draw; they never change results. Pass `show=False` / a file path where
available so that scripts can run without a screen.
