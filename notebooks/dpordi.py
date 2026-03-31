import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# 1 PATHS
# ============================================================
BASE_DIR = Path('..').resolve()
BASE_DIR = BASE_DIR / 'dp_arve'
DATA_DIR = BASE_DIR / 'data/Data Fornisseurs'

obs_path = DATA_DIR / 'ARVE/2170_Abfluss_10-Min-Mittel_1999-01-01_2024-12-31.csv'
ml_path = DATA_DIR / 'Hydrique_model/Archive prévisions Hydrique ML Arve-Bout du Monde.json'
phys_path = DATA_DIR / 'SIG-CNR_model/Prévisions QArve.xlsx'
ofev_path = DATA_DIR / 'forecasts_OFEV_models.csv'

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

LEAD_TIMES = [6, 12, 24, 48]

# ============================================================
# METRICS
# ============================================================
def nse(obs, sim):
    return 1 - np.sum((obs-sim)**2) / np.sum((obs-np.mean(obs))**2)

def kge(obs, sim):
    r = np.corrcoef(obs, sim)[0,1]
    alpha = np.std(sim)/np.std(obs)
    beta = np.mean(sim)/np.mean(obs)
    return 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)

def peak_error(obs, sim):
    return (np.max(sim)-np.max(obs))/np.max(obs)*100

def peak_timing(obs, sim, index):
    t_obs = index[np.argmax(obs)]
    t_sim = index[np.argmax(sim)]
    return (t_sim - t_obs).total_seconds()/3600

def volume_error(obs, sim):
    return (np.sum(sim)-np.sum(obs))/np.sum(obs)*100

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

# ============================================================
# LOAD ML
# ============================================================
with open(ml_path) as f:
    data = json.load(f)

records = []
for ftime, ts in data["ForecastFirstDate2Timeseries"].items():
    for vtime, q in ts.items():
        records.append([ftime, vtime, q])

ml = pd.DataFrame(records, columns=["forecast_time","valid_time","Q_ml"])
ml["forecast_time"] = pd.to_datetime(ml["forecast_time"])
ml["valid_time"] = pd.to_datetime(ml["valid_time"])

ml["lead_time_h"] = (
    ml["valid_time"]-ml["forecast_time"]
).dt.total_seconds()/3600

# ============================================================
# LOAD SIG-CNR
# ============================================================
phys = pd.read_excel(phys_path)

phys["Date"] = pd.to_datetime(phys["Date"], dayfirst=True)
phys["datetime"] = phys["Date"] + pd.to_timedelta(phys["H"]-1, unit="h")

phys = phys.rename(columns={"Prévisions Arve (CNR)":"Q_phys"})
phys = phys.set_index("datetime")

phys_h = phys[["Q_phys"]]

# ============================================================
# LOAD OFEV
# ============================================================
ofev = pd.read_csv(ofev_path)
ofev.columns = ofev.columns.str.strip()

ofev["forecast_date"] = pd.to_datetime(ofev["forecast_date"])
ofev["datetime"] = pd.to_datetime(ofev["datetime"])

ofev["discharge_m3s"] = (
    ofev["discharge_m3s"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
)

# ============================================================
# LOOP LEAD TIMES
# ============================================================
all_metrics = []

for LT in LEAD_TIMES:

    print("\n============================")
    print(f"LEAD TIME = {LT} h")
    print("============================")

    out_lt = output_dir / f"leadtime_{LT}h"
    out_lt.mkdir(exist_ok=True)

    # =========================
    # ML
    # =========================
    ml_lt = ml[np.round(ml["lead_time_h"]) == LT]
    ml_lt = ml_lt.set_index("valid_time")

    ml_h = ml_lt[["Q_ml"]].resample("1h").mean()

    # =========================
    # OFEV (MEDIAN ensemble)
    # =========================
    ofev_lt = ofev[np.round(ofev["lead_time_h"]) == LT]

    ofev_med = (
        ofev_lt
        .groupby("datetime")["discharge_m3s"]
        .median()
        .to_frame()
    )

    ofev_med = ofev_med.rename(columns={"discharge_m3s":"Q_ofev"})
    ofev_h = ofev_med.resample("1h").mean()

    # =========================
    # ALIGN
    # =========================
    df = (
        obs_h
        .join(ml_h, how="inner")
        .join(phys_h, how="inner")
        .join(ofev_h, how="inner")
    ).dropna()

    print("Common points:", len(df))

    # =========================
    # METRICS
    # =========================
    for name, col in [
        ("Hydrique_ML","Q_ml"),
        ("SIG_CNR","Q_phys"),
        ("OFEV","Q_ofev")
    ]:

        obs_v = df["Q_obs"].values
        sim_v = df[col].values

        all_metrics.append({
            "lead_time":LT,
            "model":name,
            "NSE":nse(obs_v,sim_v),
            "KGE":kge(obs_v,sim_v),
            "Peak_%":peak_error(obs_v,sim_v),
            "Timing_h":peak_timing(obs_v,sim_v,df.index),
            "Volume_%":volume_error(obs_v,sim_v)
        })

    # =========================
    # TIMESERIES
    # =========================
    plt.figure(figsize=(12,5))

    plt.plot(df.index, df["Q_obs"], label="Observed", color="black")
    plt.plot(df.index, df["Q_ml"], label="Hydrique ML")
    plt.plot(df.index, df["Q_phys"], label="SIG-CNR")
    plt.plot(df.index, df["Q_ofev"], label="OFEV (median)")

    plt.legend()
    plt.ylabel("Discharge (m³/s)")
    plt.title(f"Lead time {LT} h")

    plt.tight_layout()
    plt.savefig(out_lt / f"timeseries_LT{LT}.png", dpi=200)
    plt.close()

    # =========================
    # SCATTER
    # =========================
    for name, col in [
        ("ML","Q_ml"),
        ("SIG","Q_phys"),
        ("OFEV","Q_ofev")
    ]:

        plt.figure(figsize=(5,5))

        plt.scatter(df["Q_obs"], df[col], s=5, alpha=0.3)

        m = max(df["Q_obs"].max(), df[col].max())
        plt.plot([0,m],[0,m],"k--")

        plt.xlabel("Observed")
        plt.ylabel(name)
        plt.title(f"{name} LT {LT}h")

        plt.tight_layout()
        plt.savefig(out_lt / f"scatter_{name}_LT{LT}.png", dpi=200)
        plt.close()

# ============================================================
# SAVE METRICS
# ============================================================
metrics = pd.DataFrame(all_metrics)
metrics.to_csv(output_dir / "metrics_leadtime_comparison.csv", index=False)

print("\nSaved metrics:")
print(metrics)

print("\nScript finished")