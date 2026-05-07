import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
#ajout de matrice de confusion

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path('..').resolve() / 'dp_arve'
DATA_DIR = BASE_DIR / 'data/Data Fornisseurs'

obs_path = DATA_DIR / 'ARVE/2170_Abfluss_10-Min-Mittel_1999-01-01_2024-12-31.csv'
ml_path  = DATA_DIR / 'Hydrique_model/Archive prévisions Hydrique ML Arve-Bout du Monde.json'
hyd_path = DATA_DIR / 'Hydrique_model/Archive prévisions Hydrique hydrique-curve Arve-Bout du Monde.json'
cnr_path = DATA_DIR / 'SIG-CNR_model/Previsions_CNR_20_25.csv'
ofev_path = BASE_DIR / 'outputs/ofev_median_of_medians.csv'

output_dir = BASE_DIR / "outputs"/"peak_windows"
output_dir.mkdir(parents=True, exist_ok=True)

out_floods = output_dir / "flood_timeseries"
out_floods.mkdir(exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

FLOOD_THR = 400

LEAD_WINDOWS = [
    (0,6),
    (6,12),
    (12,18),
    (18,24),
    (24,36),
    (36,48)
]

# ============================================================
# METRICS
# ============================================================
def nse(obs, sim):
    if len(obs) < 2:
        return np.nan
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

def kge(obs, sim):
    if len(obs) < 2:
        return np.nan
    r = np.corrcoef(obs, sim)[0,1]
    alpha = np.std(sim)/np.std(obs)
    beta = np.mean(sim)/np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def compute_peak_metrics(df):

    if len(df) == 0:
        return {}

    sim = df["sim_peak"].values
    obs = df["obs_peak"].values

    return {
        "RMSE_peak": np.sqrt(np.mean((sim - obs)**2)),
        "Bias_peak_%": np.mean((sim - obs) / obs) * 100,
        "MAPE_peak_%": np.mean(np.abs((sim - obs) / obs)) * 100,
        "Timing_error_h": np.median(df["timing_error"]),
        "NSE_peak": nse(obs, sim),
        "KGE_peak": kge(obs, sim)
    }

# ============================================================
# EVENT DETECTION (pour plots uniquement)
# ============================================================

def get_flood_events(series, thr, gap_hours=12):

    mask = series > thr
    groups = (mask != mask.shift()).cumsum()

    events = []
    for _, g in series[mask].groupby(groups):
        events.append((g.index.min(), g.index.max()))

    # merge events proches
    merged = []
    for e in events:
        if not merged:
            merged.append(e)
        else:
            prev = merged[-1]
            gap = (e[0] - prev[1]).total_seconds()/3600
            if gap <= gap_hours:
                merged[-1] = (prev[0], e[1])
            else:
                merged.append(e)

    return merged

# ============================================================
# PEAK EXTRACTION
# ============================================================

def extract_peaks(model_df, obs_df, start_lt, end_lt, col_name):

    results = []

    for ftime in model_df["forecast_time"].unique():

        sub = model_df[model_df["forecast_time"] == ftime].copy()

        sub["lt"] = (sub["valid_time"] - ftime).dt.total_seconds()/3600

        window = sub[(sub["lt"] >= start_lt) & (sub["lt"] < end_lt)]
        

        if len(window) < 3:
            continue
        
        # alignement robuste
        df_merge = pd.merge_asof(
            window.sort_values("valid_time"),
            obs_df.reset_index().sort_values("Zeitstempel"),
            left_on="valid_time",
            right_on="Zeitstempel",
            direction="nearest",
            tolerance=pd.Timedelta("30min")
        )

        if len(df_merge) < 3:
            continue

        sim = df_merge[col_name].values
        obs = df_merge["Q_obs"].values
        idx = df_merge["valid_time"].values

        if np.all(np.isnan(sim)) or np.all(np.isnan(obs)):
            continue

        sim_peak = np.max(sim)
        obs_peak = np.max(obs)

        t_sim = idx[np.argmax(sim)]
        t_obs = idx[np.argmax(obs)]

        timing_error = (t_sim - t_obs) / np.timedelta64(1, 'h')

        results.append({
            "sim_peak": sim_peak,
            "obs_peak": obs_peak,
            "timing_error": timing_error
        })

    print("MODEL:", col_name)
    print("WINDOW:", start_lt, end_lt)
    print("rows after merge:", len(df_merge))
    print("window size:", len(window))
    print("obs size:", len(obs_df))

    return pd.DataFrame(results, columns=["sim_peak", "obs_peak", "timing_error"])

# ============================================================
# LOAD OBS
# ============================================================

obs = pd.read_csv(obs_path, sep=";", encoding="cp1252", skiprows=8)
obs.columns = obs.columns.str.strip()

obs["Zeitstempel"] = pd.to_datetime(obs["Zeitstempel"])
obs["Wert"] = pd.to_numeric(obs["Wert"], errors="coerce")

obs = obs[["Zeitstempel","Wert"]].rename(columns={"Wert":"Q_obs"})
obs = obs.set_index("Zeitstempel")

obs_h = obs.resample("1h").mean()

print("RAW MAX:", obs.index.max())
print("RAW MIN:", obs.index.min())

print("NaN count:", obs["Q_obs"].isna().sum())
print("Total rows:", len(obs))

# ============================================================
# LOAD ML & HYD
# ============================================================

def load_json_model(path, col):

    with open(path) as f:
        data = json.load(f)

    records = []
    for ftime, ts in data["ForecastFirstDate2Timeseries"].items():
        for vtime, q in ts.items():
            records.append([ftime, vtime, q])

    df = pd.DataFrame(records, columns=["forecast_time","valid_time",col])
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    return df

ml  = load_json_model(ml_path, "Q_ml")
hyd = load_json_model(hyd_path, "Q_hyd")

# ============================================================
# LOAD CNR
# ============================================================

cnr = pd.read_csv(cnr_path)

cnr["DateTime_PREV"] = pd.to_datetime(cnr["DateTime_PREV"])
cnr["Date_Prevision"] = pd.to_datetime(cnr["Date_Prevision"])

cnr = cnr.rename(columns={"Q":"Q_cnr"})

cnr["forecast_time"] = cnr["DateTime_PREV"]
cnr["valid_time"] = cnr["Date_Prevision"]

cnr = cnr[["forecast_time","valid_time","Q_cnr"]]

# ============================================================
# LOAD OFEV
# ============================================================
ofev = pd.read_csv(ofev_path)
ofev.columns = ofev.columns.str.strip()

print(ofev.columns)
print(ofev.head())
# Convertir directement les colonnes existantes (déjà renommées dans le fichier)
ofev["forecast_time"] = pd.to_datetime(ofev["forecast_time"])
ofev["valid_time"] = pd.to_datetime(ofev["valid_time"])

# Pas besoin de recalculer Q_ofev, car le fichier contient déjà la médiane des médianes
ofev = ofev[["forecast_time", "valid_time", "Q_ofev"]]

START_DATE = "2020-06-11"

obs_h = obs_h.loc[START_DATE:]

ml  = ml[ml["valid_time"] >= START_DATE]
hyd = hyd[hyd["valid_time"] >= START_DATE]
cnr = cnr[cnr["valid_time"] >= START_DATE]
ofev = ofev[ofev["valid_time"] >= START_DATE]
# ============================================================
# MAIN LOOP
# ============================================================

all_results = []

models = [
    ("Hydrique_ML", ml, "Q_ml"),
    ("Hydrique_Curve", hyd, "Q_hyd"),
    ("SIG_CNR", cnr, "Q_cnr"),
    ("OFEV", ofev, "Q_ofev")
]

for (lt_start, lt_end) in LEAD_WINDOWS:

    print(f"\n=== WINDOW {lt_start}-{lt_end}h ===")

    for name, df_model, col in models:

        peaks = extract_peaks(df_model, obs_h, lt_start, lt_end, col)

        # filtre crues
        peaks_flood = peaks[peaks["obs_peak"] > FLOOD_THR]

        metrics = compute_peak_metrics(peaks_flood)

        metrics.update({
            "model": name,
            "window": f"{lt_start}-{lt_end}",
            "n_events": len(peaks_flood)
        })

        all_results.append(metrics)

# ============================================================
# SAVE METRICS
# ============================================================

df_results = pd.DataFrame(all_results)
df_results.to_csv(output_dir / "peak_metrics.csv", index=False)

print(df_results)

# ============================================================
# PLOTS DES CRUES
# ============================================================

def filter_window(df_model, col, start_lt, end_lt):

    df = df_model.copy()

    df["lead_time"] = (
        df["valid_time"] - df["forecast_time"]
    ).dt.total_seconds()/3600

    df = df[
        (df["lead_time"] >= start_lt) &
        (df["lead_time"] < end_lt)
    ]

    # plusieurs forecasts → médiane
    df = df.groupby("valid_time")[col].median()

    return df

events = get_flood_events(obs_h["Q_obs"], FLOOD_THR)

for i, (start, end) in enumerate(events):

    # fenêtre autour de la crue
    start_win = start - pd.Timedelta(days=2)
    end_win   = end + pd.Timedelta(days=2)

    for (lt_start, lt_end) in LEAD_WINDOWS:

        plt.figure(figsize=(16,6))

        # =========================
        # OBS
        # =========================
        plt.plot(
            obs_h.loc[start_win:end_win].index,
            obs_h.loc[start_win:end_win]["Q_obs"],
            color="black",
            label="Observed"
        )

        # =========================
        # MODELS FILTRÉS PAR WINDOW
        # =========================
        for name, df_model, col in models:

            ts = filter_window(df_model, col, lt_start, lt_end)

            if len(ts) == 0:
                continue

            ts = ts.loc[start_win:end_win]

            plt.plot(ts.index, ts.values, label=name)

        # =========================
        # STYLE
        # =========================
        plt.title(
            f"Flood {start.strftime('%Y-%m-%d')} | "
            f"LT window {lt_start}-{lt_end}h"
        )

        plt.xlim(start_win, end_win)
        plt.legend()

        # option : zoom vertical
        # plt.ylim(0, 1500)

        plt.tight_layout()

        fname = (
            f"flood_{i}_"
            f"{start.strftime('%Y%m%d')}_"
            f"LT_{lt_start}_{lt_end}.png"
        )

        plt.savefig(out_floods / fname, dpi=200)
        plt.close()



print("DONE")