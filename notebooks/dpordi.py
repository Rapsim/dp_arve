import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path('..').resolve() / 'dp_arve'
DATA_DIR = BASE_DIR / 'data/Data Fornisseurs'

obs_path = DATA_DIR / 'ARVE/2170_Abfluss_10-Min-Mittel_1999-01-01_2024-12-31.csv'
ml_path = DATA_DIR / 'Hydrique_model/Archive prévisions Hydrique ML Arve-Bout du Monde.json'
hyd_path = DATA_DIR / 'Hydrique_model/Archive prévisions Hydrique hydrique-curve Arve-Bout du Monde.json'
cnr_path = DATA_DIR / 'SIG-CNR_model/Previsions_CNR_20_25.csv'
ofev_path = DATA_DIR / 'forecasts_OFEV_models.csv'

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

LEAD_TIMES = [6,12,24,36,48]
FLOOD_THR = 400

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

def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim)**2))

def relative_volume_error(obs, sim):
    return (np.sum(sim) - np.sum(obs)) / np.sum(obs) * 100

def mape_high_flows(obs, sim, thr):
    mask = obs > thr
    if np.sum(mask) == 0:
        return np.nan
    return 100 * np.mean(np.abs((obs[mask] - sim[mask]) / obs[mask]))

# ============================================================
# EVENT DETECTION
# ============================================================

def get_flood_events(series, thr):
    mask = series > thr
    groups = (mask != mask.shift()).cumsum()
    events = []

    for _, g in series[mask].groupby(groups):
        events.append((g.index.min(), g.index.max()))

    return events

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
ml["lead_time_h"] = (ml["valid_time"]-ml["forecast_time"]).dt.total_seconds()/3600
ml = ml.set_index("valid_time")

# ============================================================
# LOAD HYD
# ============================================================

with open(hyd_path) as f:
    data = json.load(f)

records = []
for ftime, ts in data["ForecastFirstDate2Timeseries"].items():
    for vtime, q in ts.items():
        records.append([ftime, vtime, q])

hyd = pd.DataFrame(records, columns=["forecast_time","valid_time","Q_hyd"])
hyd["forecast_time"] = pd.to_datetime(hyd["forecast_time"])
hyd["valid_time"] = pd.to_datetime(hyd["valid_time"])
hyd["lead_time_h"] = (hyd["valid_time"]-hyd["forecast_time"]).dt.total_seconds()/3600
hyd = hyd.set_index("valid_time")

# ============================================================
# LOAD CNR
# ============================================================

cnr = pd.read_csv(cnr_path)

cnr["DateTime_PREV"] = pd.to_datetime(cnr["DateTime_PREV"])
cnr["Date_Prevision"] = pd.to_datetime(cnr["Date_Prevision"])

cnr = cnr.rename(columns={"Q":"Q_cnr"})

cnr["lead_time_h"] = (
    cnr["Date_Prevision"] - cnr["DateTime_PREV"]
).dt.total_seconds()/3600

cnr["valid_time"] = cnr["Date_Prevision"]
cnr = cnr.set_index("valid_time")

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
    .str.replace(",",".")
    .astype(float)
)

ofev["lead_time_h"] = (
    ofev["datetime"] - ofev["forecast_date"]
).dt.total_seconds()/3600

ofev = ofev.rename(columns={
    "datetime":"valid_time",
    "discharge_m3s":"Q_ofev"
})

ofev = ofev.set_index("valid_time")

# ============================================================
# LOOP LEAD TIMES
# ============================================================

all_metrics = []

for LT in LEAD_TIMES:

    print(f"\n========== LT {LT}h ==========")

    out_lt = output_dir / f"leadtime_{LT}h"
    out_lt.mkdir(exist_ok=True)

    ml_lt = ml[np.round(ml["lead_time_h"])==LT][["Q_ml"]]
    hyd_lt = hyd[np.round(hyd["lead_time_h"])==LT][["Q_hyd"]]
    cnr_lt = cnr[np.round(cnr["lead_time_h"])==LT][["Q_cnr"]]
    ofev_lt = ofev[np.round(ofev["lead_time_h"])==LT][["Q_ofev"]]

    ofev_lt = ofev_lt.groupby(ofev_lt.index).median()

    # ============================================================
    # PERIODE COMMUNE
    # ============================================================

    series = [s for s in [ml_lt, hyd_lt, cnr_lt, ofev_lt] if len(s)>0]

    start_common = max(s.index.min() for s in series)
    end_common   = min(s.index.max() for s in series)

    obs_plot  = obs_h.loc[start_common:end_common]
    ml_plot   = ml_lt.loc[start_common:end_common]
    hyd_plot  = hyd_lt.loc[start_common:end_common]
    cnr_plot  = cnr_lt.loc[start_common:end_common]
    ofev_plot = ofev_lt.loc[start_common:end_common]

    # ============================================================
    # TIMESERIES COMPLETE
    # ============================================================

    plt.figure(figsize=(16,6))

    plt.plot(obs_plot.index, obs_plot["Q_obs"], color="black", label="Observed")
    plt.plot(ml_plot.index, ml_plot["Q_ml"], label="Hydrique ML")
    plt.plot(hyd_plot.index, hyd_plot["Q_hyd"], label="Hydrique Curve")
    plt.plot(cnr_plot.index, cnr_plot["Q_cnr"], label="SIG-CNR")
    plt.plot(ofev_plot.index, ofev_plot["Q_ofev"], label="OFEV")

    plt.xlim(start_common, end_common)

    plt.legend()
    plt.title(f"Timeseries LT {LT}h")
    plt.tight_layout()
    plt.savefig(out_lt / f"timeseries_LT{LT}.png", dpi=200)
    plt.close()

    
    # ============================================================
    # FLOOD EVENTS TIMESERIES
    # ============================================================

    events = get_flood_events(obs_plot["Q_obs"], FLOOD_THR)

    for i,(start,end) in enumerate(events):

        # fenêtre ±3 jours
        start_win = start - pd.Timedelta(days=3)
        end_win   = end + pd.Timedelta(days=3)

        fig, ax = plt.subplots(figsize=(16,6))

        # plots
        ax.plot(obs_plot.loc[start_win:end_win].index,
                obs_plot.loc[start_win:end_win,"Q_obs"],
                color="black", label="Observed")

        if len(ml_plot)>0:
            ax.plot(ml_plot.loc[start_win:end_win].index,
                    ml_plot.loc[start_win:end_win,"Q_ml"],
                    label="Hydrique ML")

        if len(hyd_plot)>0:
            ax.plot(hyd_plot.loc[start_win:end_win].index,
                    hyd_plot.loc[start_win:end_win,"Q_hyd"],
                    label="Hydrique Curve")

        if len(cnr_plot)>0:
            ax.plot(cnr_plot.loc[start_win:end_win].index,
                    cnr_plot.loc[start_win:end_win,"Q_cnr"],
                    label="SIG-CNR")

        if len(ofev_plot)>0:
            ax.plot(ofev_plot.loc[start_win:end_win].index,
                    ofev_plot.loc[start_win:end_win,"Q_ofev"],
                    label="OFEV")

        # limites strictes
        ax.set_xlim(start_win, end_win)

        # titre avec debut/fin du plot
        title = (
            f"Flood {start_win.strftime('%d/%m')} - "
            f"{end_win.strftime('%d/%m %Y')}"
        )

        ax.set_title(title)

        # format axe temps
        ax.xaxis.set_major_formatter(
            plt.matplotlib.dates.DateFormatter('%d/%m\n%H:%M')
        )

        ax.legend()
        plt.tight_layout()

        # nom fichier avec dates
        fname = (
            f"flood_"
            f"{start_win.strftime('%Y%m%d')}_"
            f"{end_win.strftime('%Y%m%d')}_"
            f"LT{LT}.png"
        )

        plt.savefig(out_lt / fname, dpi=200)
        plt.close()

    # ============================================================
    # METRICS + SCATTER
    # ============================================================

    for name,model in [
        ("Hydrique_ML",ml_plot),
        ("Hydrique_Curve",hyd_plot),
        ("SIG_CNR",cnr_plot),
        ("OFEV",ofev_plot)
    ]:

        df = pd.merge(obs_plot, model, left_index=True,
                      right_index=True, how="inner")

        if len(df)<10:
            continue

        obs_v = df["Q_obs"].values
        sim_v = df.iloc[:,1].values

        all_metrics.append({
            "lead_time": LT,
            "model": name,

            # global metrics
            "NSE": nse(obs_v, sim_v),
            "KGE": kge(obs_v, sim_v),
            "RMSE": rmse(obs_v, sim_v),

            # flood-focused metrics
            "REQ_%": peak_error(obs_v, sim_v),
            "TP_h": peak_timing(obs_v, sim_v, df.index),
            "RER_%": relative_volume_error(obs_v, sim_v),

            # high-flow metric
            "MAPE_high_%": mape_high_flows(obs_v, sim_v, FLOOD_THR)
        })

        plt.figure(figsize=(5,5))

        plt.scatter(obs_v,sim_v,s=5,alpha=0.3)

        m=max(obs_v.max(),sim_v.max())
        plt.plot([0,m],[0,m],"k--")

        plt.xlabel("Observed")
        plt.ylabel(name)
        plt.title(f"{name} LT{LT}")

        plt.tight_layout()
        plt.savefig(out_lt / f"scatter_{name}_LT{LT}.png", dpi=200)
        plt.close()

# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(all_metrics)
metrics.to_csv(output_dir / "metrics_all_models.csv", index=False)

print(metrics)
print("finished")