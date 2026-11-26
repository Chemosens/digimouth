# %% [markdown]
# # Appendix: parameter uncertainty with time-correlated residuals (in vitro, PTR-MS)
#
# The convection model is fitted (k and tMS, Kaw fixed) on each of the three in vitro
# replicates. PTR-MS kinetics are sampled every 0.5 s and their residuals are correlated in
# time, so confidence intervals that treat time points as independent are too narrow.
# For each replicate, four ways of computing a 95 % confidence interval are compared:
#
# - **Jacobian**: local least-squares Jacobian + Student t (assumes independent residuals);
# - **iid bootstrap**: residuals resampled one by one (assumes independent residuals);
# - **AR(1) bootstrap**: parametric bootstrap, Gaussian first-order autoregressive noise with
#   the standard deviation and lag-1 autocorrelation of the residuals
#   (Efron & Tibshirani, 1993; Davison & Hinkley, 1997);
# - **block bootstrap**: moving block bootstrap (Kunsch, 1989), blocks of consecutive
#   residuals resampled with replacement and laid end to end; block lengths of 3 to 24
#   points (1.5 to 12 s), around the usual n^(1/3) rule (6 points; Hall, Horowitz & Jing, 1995).
#
# In every bootstrap the fitted curve is kept unchanged: only the noise is resampled, and the
# model is fitted again on "fitted curve + artificial noise". Intervals are the 2.5th and
# 97.5th percentiles of the refitted parameters.
#
# **Retained method: block bootstrap with blocks of 24 points (12 s).** It makes no assumption
# on the form of the correlation, and the block length is chosen from the autocorrelation of
# the residuals (section 1). The AR(1) bootstrap is not retained because the observed
# autocorrelation persists much longer than that of an AR(1) process.
#
# Runtime: about 6 minutes with N_BOOT = 500.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import digimouth
from digimouth import optimization as opt
from digimouth import existing_models as exmod
from digimouth import utils

try:
    EXAMPLES_DIR = Path(__file__).resolve().parent
except NameError:
    EXAMPLES_DIR = Path(digimouth.__file__).resolve().parents[1] / "examples"

N_BOOT = 500                    # bootstrap samples per method and replicate
BLOCK_LENGTHS = [3, 6, 12, 24]  # points per block (0.5 s between points)
RETAINED = "block bootstrap (L=24)"
SEED = 0
SAVE_DIR = None                 # e.g. "output" to save the figures and the tables

# %% [markdown]
# ## Data: three replicates, aligned on the start of the release, first 120 s

# %%
df = pd.read_csv(EXAMPLES_DIR / "data" / "in_vitro_isoamyl_acetate_5min.csv")
df = utils.recale_time_on_peak(df=df, file_col="file", time_col="time", intensity_col="intensity",
                               baseline_max_time=30, threshold_std=4, min_time_after=0)
df = df[df["time"] < 120]
curves = {name: (g["time"].to_numpy(), g["intensity"].to_numpy())
          for name, g in df.sort_values("time").groupby("file")}
print({name: len(t) for name, (t, _) in curves.items()}, "points per replicate")

# %% [markdown]
# ## Model: in vitro convection model, k and tMS fitted (same settings as digimouth_optim_in_vitro.py)

# %%
Kaw = 2.3e-2                       # fixed, from GC measurements
Cl_ini = 2                         # g/m3
A = np.pi * (3.7e-2 / 2) ** 2      # m2
Dg = 2.633e-6                      # m3/s
Vr, Vl = 48e-6, 2e-6               # m3
model = exmod.generate_convection_model(Kaw_initial=Kaw, k_initial=1.38e-4, A_ini=A, Dg_ini=Dg,
                                        Vr=Vr, Vl=Vl, tMS=1)
y0 = {"Cl": Cl_ini, "Cg": Kaw * Cl_ini, "CMS": 0}
keys = ["k", "tMS"]
units = {"k": "m/s", "tMS": "s"}
start = [1.38e-4, 1.0]
bounds = ([1e-5, 1e-3], [1e-3, 2])


def run_bootstrap(t, y, resampling, block_length=None):
    return opt.bootstrap_params(t, y, y0, model, keys, start, bounds=bounds, normalization=True,
                                n_boot=N_BOOT, random_state=SEED, resampling=resampling,
                                block_length=block_length)


def save(fig, filename):
    if SAVE_DIR:
        Path(SAVE_DIR).mkdir(exist_ok=True)
        fig.savefig(Path(SAVE_DIR) / filename, dpi=300)

# %% [markdown]
# ## 1. Residuals and their autocorrelation
#
# Top: residuals of the fit (normalised scale). Bottom: their autocorrelation, compared with
# the autocorrelation of an AR(1) process with the same lag-1 value (red). Dotted lines:
# approximate 95 % limits for independent residuals. The observed autocorrelation decreases
# much more slowly than the AR(1) one, and the residuals show slow systematic deviations from
# the fitted curve (model-data mismatch), not only measurement noise.

# %%
max_lag = 60                      # points, i.e. 30 s
fig, axes = plt.subplots(2, 3, figsize=(13, 6))
for j, (name, (t, y)) in enumerate(curves.items()):
    fit = opt.optim_param_dynamic(t, y, y0, model, keys, start, bounds=bounds, normalization=True)
    r = opt.calculate_residuals(fit["opt_param"], t, y, y0, model, keys, normalization=True)
    r = r - r.mean()
    rho = np.corrcoef(r[:-1], r[1:])[0, 1]
    lags = np.arange(0, max_lag + 1)
    acf = [1.0] + [np.corrcoef(r[:-h], r[h:])[0, 1] for h in lags[1:]]
    limit = 1.96 / np.sqrt(len(r))
    axes[0, j].plot(t, r, lw=0.8)
    axes[0, j].axhline(0, c="k", lw=0.5)
    axes[0, j].set_title(f"{name}: residuals")
    axes[0, j].set_xlabel("Time (s)")
    axes[0, j].set_ylabel("Residual (normalised)")
    axes[1, j].bar(lags * 0.5, acf, width=0.35, label="Residuals")
    axes[1, j].plot(lags * 0.5, rho ** lags, "r-", label=f"AR(1), lag-1 = {rho:.2f}")
    axes[1, j].axhline(limit, ls=":", c="grey")
    axes[1, j].axhline(-limit, ls=":", c="grey")
    axes[1, j].axvline(12, ls="--", c="k", lw=0.8, label="Retained block length (12 s)")
    axes[1, j].set_title(f"{name}: autocorrelation of the residuals")
    axes[1, j].set_xlabel("Lag (s)")
    axes[1, j].set_ylabel("Autocorrelation")
    axes[1, j].legend(fontsize=7)
fig.tight_layout()
save(fig, "appendix_residual_autocorrelation.png")
plt.show()

# %% [markdown]
# ## 2. Confidence intervals with the four methods

# %%
rows = []
for name, (t, y) in curves.items():
    runs = [("iid bootstrap", run_bootstrap(t, y, "iid"))]
    runs.append(("AR(1) bootstrap", run_bootstrap(t, y, "ar1")))
    runs += [(f"block bootstrap (L={L})", run_bootstrap(t, y, "block", L)) for L in BLOCK_LENGTHS]
    fit = runs[0][1]["original_fit"]
    for i, key in enumerate(keys):
        rows.append({"replicate": name, "method": "Jacobian", "parameter": key, "estimate": fit["opt_param"][i],
                     "low": fit["conf_intervals"][key][0], "high": fit["conf_intervals"][key][1]})
        for method, b in runs:
            rows.append({"replicate": name, "method": method, "parameter": key, "estimate": fit["opt_param"][i],
                         "low": b["conf_intervals_boot"][key][0], "high": b["conf_intervals_boot"][key][1]})
    print(name, "done; lag-1 autocorrelation =", round(runs[0][1]["rho"], 2),
          "; failed fits:", sum(b["n_failed"] for _, b in runs))

table = pd.DataFrame(rows)
table["width"] = table["high"] - table["low"]
jacobian_width = table[table["method"] == "Jacobian"].set_index(["replicate", "parameter"])["width"]
table["width / Jacobian"] = [w / jacobian_width[(r, p)] for w, r, p in
                             zip(table["width"], table["replicate"], table["parameter"])]
pd.set_option("display.width", 160)
for key in keys:
    print(f"\n=== {key} ({units[key]}) ===")
    print(table[table["parameter"] == key].drop(columns="parameter").to_string(index=False, float_format="%.4g"))
if SAVE_DIR:
    Path(SAVE_DIR).mkdir(exist_ok=True)
    table.to_csv(Path(SAVE_DIR) / "appendix_confidence_intervals_all_methods.csv", index=False)

# %% [markdown]
# ## 3. Figure: 95 % confidence intervals per replicate and method
#
# Vertical ticks: estimated value. The retained method (block bootstrap, L = 24 points) is drawn in black.

# %%
methods = list(dict.fromkeys(table["method"]))
colors = dict(zip(methods, plt.cm.viridis(np.linspace(0, 0.9, len(methods)))))
colors[RETAINED] = "black"
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, key in zip(axes, keys):
    sub = table[table["parameter"] == key]
    for i, name in enumerate(curves):
        for j, method in enumerate(methods):
            row = sub[(sub["replicate"] == name) & (sub["method"] == method)].iloc[0]
            ypos = i * (len(methods) + 2) + j
            ax.plot([row["low"], row["high"]], [ypos, ypos], color=colors[method],
                    lw=4 if method == RETAINED else 2.5, label=method if i == 0 else None)
            ax.plot(row["estimate"], ypos, "|", color="red", ms=8)
    ax.set_yticks([i * (len(methods) + 2) + (len(methods) - 1) / 2 for i in range(len(curves))])
    ax.set_yticklabels(list(curves))
    ax.invert_yaxis()
    ax.set_xlabel(f"{key} ({units[key]})")
    ax.set_title(f"95 % confidence intervals of {key}")
axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout()
save(fig, "appendix_confidence_intervals_methods.png")
plt.show()

# %% [markdown]
# ## 4. Retained intervals (block bootstrap, L = 24 points) and between-replicate variability
#
# The within-replicate intervals are conditional on the model and, because of the
# model-data mismatch seen in section 1, should be read as lower bounds. The variability
# across the three replicates is reported descriptively only (three replicates).

# %%
retained = table[table["method"] == RETAINED].copy()
retained["value"] = [f"{est:.3g} [{lo:.3g}, {hi:.3g}]"
                     for est, lo, hi in zip(retained["estimate"], retained["low"], retained["high"])]
summary = retained.pivot(index="replicate", columns="parameter", values="value")
summary.columns = [f"{key} ({units[key]}), estimate [95 % CI]" for key in summary.columns]
print(summary.to_string())
estimates = retained.pivot(index="replicate", columns="parameter", values="estimate")
print("\nBetween replicates (descriptive):")
for key in keys:
    print(f"  {key}: mean = {estimates[key].mean():.4g} {units[key]}, "
          f"range = [{estimates[key].min():.4g}, {estimates[key].max():.4g}]")
if SAVE_DIR:
    summary.to_csv(Path(SAVE_DIR) / "appendix_retained_intervals.csv")
