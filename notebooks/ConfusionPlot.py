import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import r2_score

# ============================================================
# STYLE
# ============================================================

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("..").resolve() / "dp_arve" # uncomment the right line

BASE_DIR = Path("..").resolve() / "analysis"

DATA_DIR = BASE_DIR / "data/Data Fornisseurs"

obs_path = DATA_DIR / "ARVE/2170_Abfluss_10-Min-Mittel_1999-01-01_2024-12-31.csv"

ml_path = DATA_DIR / "Hydrique_model/Archive prévisions Hydrique ML Arve-Bout du Monde.json"

hyd_path = DATA_DIR / "Hydrique_model/Archive prévisions Hydrique hydrique-curve Arve-Bout du Monde.json"

cnr_path = DATA_DIR / "SIG-CNR_model/Previsions_CNR_20_25.csv"

ofev_path = BASE_DIR / "outputs/OFEV_probabilistic/station2170_q_quantiles_by_model.csv"

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_dir = BASE_DIR / "outputs" / "scatter_confusion_0_48h"

output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================

THRESHOLDS = [400, 600, 800]

MAX_LEADTIME = 48

# ============================================================
# LOAD OBSERVED DATA
# ============================================================

obs = pd.read_csv(
    obs_path,
    sep=";",
    encoding="cp1252",
    skiprows=8
)

obs.columns = obs.columns.str.strip()

obs["Zeitstempel"] = pd.to_datetime(obs["Zeitstempel"])

obs["Wert"] = pd.to_numeric(
    obs["Wert"],
    errors="coerce"
)

obs = obs.rename(columns={"Wert": "Q_obs"})

obs = obs[["Zeitstempel", "Q_obs"]]

obs = obs.set_index("Zeitstempel")

# hourly aggregation
obs_h = obs.resample("1h").mean()

# ============================================================
# LOAD HYDRIQUE ML
# ============================================================

with open(ml_path) as f:
    data = json.load(f)

records = []

for ftime, ts in data["ForecastFirstDate2Timeseries"].items():

    for vtime, q in ts.items():

        records.append([ftime, vtime, q])

ml = pd.DataFrame(
    records,
    columns=["forecast_time", "valid_time", "Q_ml"]
)

ml["forecast_time"] = pd.to_datetime(ml["forecast_time"])

ml["valid_time"] = pd.to_datetime(ml["valid_time"])

# compute leadtime
ml["lead_time_h"] = (
    ml["valid_time"] - ml["forecast_time"]
).dt.total_seconds() / 3600

# keep only 0-48h forecasts
ml = ml[
    (ml["lead_time_h"] >= 0) &
    (ml["lead_time_h"] <= MAX_LEADTIME)
]

ml = ml.set_index("valid_time")

ml = ml[~ml.index.duplicated(keep="first")]

ml = ml.sort_index()

# ============================================================
# LOAD HYDRIQUE PHYSIQUE
# ============================================================

with open(hyd_path) as f:
    data = json.load(f)

records = []

for ftime, ts in data["ForecastFirstDate2Timeseries"].items():

    for vtime, q in ts.items():

        records.append([ftime, vtime, q])

hyd = pd.DataFrame(
    records,
    columns=["forecast_time", "valid_time", "Q_hyd"]
)

hyd["forecast_time"] = pd.to_datetime(hyd["forecast_time"])

hyd["valid_time"] = pd.to_datetime(hyd["valid_time"])

# compute leadtime
hyd["lead_time_h"] = (
    hyd["valid_time"] - hyd["forecast_time"]
).dt.total_seconds() / 3600

# keep only 0-48h forecasts
hyd = hyd[
    (hyd["lead_time_h"] >= 0) &
    (hyd["lead_time_h"] <= MAX_LEADTIME)
]

hyd = hyd.set_index("valid_time")

hyd = hyd[~hyd.index.duplicated(keep="first")]

hyd = hyd.sort_index()

# ============================================================
# LOAD CNR
# ============================================================

cnr = pd.read_csv(cnr_path)

cnr["DateTime_PREV"] = pd.to_datetime(
    cnr["DateTime_PREV"]
)

cnr["Date_Prevision"] = pd.to_datetime(
    cnr["Date_Prevision"]
)

cnr = cnr.rename(columns={"Q": "Q_cnr"})

# compute leadtime
cnr["lead_time_h"] = (
    cnr["Date_Prevision"] - cnr["DateTime_PREV"]
).dt.total_seconds() / 3600

# keep only 0-48h forecasts
cnr = cnr[
    (cnr["lead_time_h"] >= 0) &
    (cnr["lead_time_h"] <= MAX_LEADTIME)
]

cnr["valid_time"] = cnr["Date_Prevision"]

cnr = cnr.set_index("valid_time")

cnr = cnr[~cnr.index.duplicated(keep="first")]

cnr = cnr.sort_index()

# ============================================================
# LOAD OFEV
# ============================================================

ofev = pd.read_csv(ofev_path)

ofev.columns = ofev.columns.str.strip()

ofev["issue_time"] = pd.to_datetime(ofev["issue_time"])

ofev["valid_time"] = pd.to_datetime(ofev["valid_time"])

# compute leadtime
ofev["lead_time_h"] = (
    ofev["valid_time"] - ofev["issue_time"]
).dt.total_seconds() / 3600

# keep only 0-48h forecasts
ofev = ofev[
    (ofev["lead_time_h"] >= 0) &
    (ofev["lead_time_h"] <= MAX_LEADTIME)
]

# median aggregation
ofev = (
    ofev
    .groupby("valid_time")["Q_p75"]
    .median()
    .reset_index()
)

ofev = ofev.rename(columns={"Q_p75": "Q_ofev"})

ofev = ofev.set_index("valid_time")

ofev = ofev[~ofev.index.duplicated(keep="first")]

ofev = ofev.sort_index()

# ============================================================
# COMMON PERIOD
# ============================================================

series = [
    ml[["Q_ml"]],
    hyd[["Q_hyd"]],
    cnr[["Q_cnr"]],
    ofev[["Q_ofev"]]
]

start_common = max(
    s.index.min() for s in series
)

end_common = min(
    s.index.max() for s in series
)

obs_plot = obs_h.loc[start_common:end_common]

ml_plot = ml.loc[start_common:end_common]

hyd_plot = hyd.loc[start_common:end_common]

cnr_plot = cnr.loc[start_common:end_common]

ofev_plot = ofev.loc[start_common:end_common]

# ============================================================
# SCATTER CONFUSION FUNCTION
# ============================================================

def plot_scatter_confusion(
    df,
    model_name,
    threshold,
    output_dir
):

    obs = df["Q_obs"].values

    sim = df.iloc[:, 1].values

    # remove NaN
    mask = ~np.isnan(obs) & ~np.isnan(sim)

    obs = obs[mask]

    sim = sim[mask]

    # ============================================================
    # CONFUSION QUADRANTS
    # ============================================================

    TP_mask = (obs >= threshold) & (sim >= threshold)

    FN_mask = (obs >= threshold) & (sim < threshold)

    FP_mask = (obs < threshold) & (sim >= threshold)

    # TN intentionally ignored

    TP = np.sum(TP_mask)

    FN = np.sum(FN_mask)

    FP = np.sum(FP_mask)

    # IMPORTANT:
    # only TP + FN + FP considered
    total = TP + FN + FP

    TP_p = 100 * TP / total if total > 0 else 0

    FN_p = 100 * FN / total if total > 0 else 0

    FP_p = 100 * FP / total if total > 0 else 0

    # ============================================================
    # R²
    # ============================================================

    r2 = r2_score(obs, sim)

    # ============================================================
    # FIGURE
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        sim,
        obs,
        s=8,
        alpha=0.30
    )

    lim = max(
        np.max(obs),
        np.max(sim),
        threshold
    ) + 50

    # y=x
    ax.plot(
        [0, lim],
        [0, lim],
        linestyle="--",
        linewidth=1.5,
        color="black"
    )

    # threshold lines
    ax.axvline(
        threshold,
        linestyle="--",
        color="red",
        linewidth=2
    )

    ax.axhline(
        threshold,
        linestyle="--",
        color="red",
        linewidth=2
    )

    # ============================================================
    # TEXT STYLE
    # ============================================================

    text_style = dict(
        fontsize=16,
        fontweight="bold",
        bbox=dict(
            facecolor="white",
            alpha=0.90,
            boxstyle="round"
        )
    )

    # ============================================================
    # TP
    # ============================================================

    ax.text(
        threshold + (lim - threshold) * 0.35,
        threshold + (lim - threshold) * 0.35,
        f"TP\nn = {TP}\n{TP_p:.1f} %",
        ha="center",
        va="center",
        color="darkgreen",
        **text_style
    )

    # ============================================================
    # FN
    # ============================================================

    ax.text(
        threshold * 0.5,
        threshold + (lim - threshold) * 0.35,
        f"FN\nn = {FN}\n{FN_p:.1f} %",
        ha="center",
        va="center",
        color="darkorange",
        **text_style
    )

    # ============================================================
    # FP
    # ============================================================

    ax.text(
        threshold + (lim - threshold) * 0.35,
        threshold * 0.5,
        f"FP\nn = {FP}\n{FP_p:.1f} %",
        ha="center",
        va="center",
        color="crimson",
        **text_style
    )

    # ============================================================
    # TN ZONE LABEL
    # ============================================================

    ax.text(
        threshold * 0.5,
        threshold * 0.5,
        "TN\nignored",
        ha="center",
        va="center",
        color="gray",
        fontsize=16
    )

    # ============================================================
    # R²
    # ============================================================

    ax.text(
        0.03,
        0.97,
        f"$R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(
            facecolor="white",
            alpha=0.90,
            boxstyle="round"
        )
    )

    # ============================================================
    # AXES
    # ============================================================

    ax.set_xlim(0, lim)

    ax.set_ylim(0, lim)

    ax.set_xlabel("Simulated discharge [m³/s]", fontsize=16)

    ax.set_ylabel("Observed discharge [m³/s]", fontsize=16)

    ax.set_title(
        f"{model_name}, threshold = {threshold} m³/s\n"
        f"Forecasts between 0h and 48h",
            fontsize=18
    )

    plt.tight_layout()

    # ============================================================
    # SAVE
    # ============================================================

    plt.savefig(
        output_dir / f"scatter_confusion_{model_name}_{threshold}.png",
        dpi=300
    )

    plt.close()

# ============================================================
# MODELS
# ============================================================

models = [
    ("Hydrique_ML", ml_plot[["Q_ml"]]),
    ("Hydrique_physique", hyd_plot[["Q_hyd"]]),
    ("SIG_CNR", cnr_plot[["Q_cnr"]]),
    ("FOEN", ofev_plot[["Q_ofev"]])
]

# ============================================================
# MAIN LOOP
# ============================================================

for model_name, model_df in models:

    print(f"\nProcessing {model_name}")

    # merge obs + model
    df = pd.merge(
        obs_plot,
        model_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    df = df.dropna()

    for thr in THRESHOLDS:

        print(f"Threshold = {thr} m3/s")

        plot_scatter_confusion(
            df=df,
            model_name=model_name,
            threshold=thr,
            output_dir=output_dir
        )

print("\nFinished.")

