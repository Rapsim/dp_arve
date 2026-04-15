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

def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim)**2))

def mape_high_flows(obs, sim, thr):
    mask = obs > thr
    if np.sum(mask) == 0:
        return np.nan
    return 100 * np.mean(np.abs((obs[mask] - sim[mask]) / obs[mask]))

# ============================================================
# EVENT METRICS (CORRIGÉ)
# ============================================================

def peak_error_event(obs, sim):
    return (np.max(sim) - np.max(obs)) / np.max(obs) * 100

def peak_timing_event(obs, sim, index):
    t_obs = index[np.argmax(obs)]
    t_sim = index[np.argmax(sim)]
    return (t_sim - t_obs).total_seconds() / 3600

def volume_error_event(obs, sim):
    return (np.sum(sim) - np.sum(obs)) / np.sum(obs) * 100

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
# LOAD ML / HYD / CNR / OFEV (IDENTIQUE)
# ============================================================

def load_json_model(path, colname):
    with open(path) as f:
        data = json.load(f)

    records = []
    for ftime, ts in data["ForecastFirstDate2Timeseries"].items():
        for vtime, q in ts.items():
            records.append([ftime, vtime, q])

    df = pd.DataFrame(records, columns=["forecast_time","valid_time",colname])
    df["forecast_time"] = pd.to_datetime(df["forecast_time"])
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df["lead_time_h"] = (df["valid_time"]-df["forecast_time"]).dt.total_seconds()/3600

    return df.set_index("valid_time")

ml = load_json_model(ml_path, "Q_ml")
hyd = load_json_model(hyd_path, "Q_hyd")

cnr = pd.read_csv(cnr_path)
cnr["DateTime_PREV"] = pd.to_datetime(cnr["DateTime_PREV"])
cnr["Date_Prevision"] = pd.to_datetime(cnr["Date_Prevision"])
cnr = cnr.rename(columns={"Q":"Q_cnr"})
cnr["lead_time_h"] = (cnr["Date_Prevision"] - cnr["DateTime_PREV"]).dt.total_seconds()/3600
cnr["valid_time"] = cnr["Date_Prevision"]
cnr = cnr.set_index("valid_time")

ofev = pd.read_csv(ofev_path)
ofev.columns = ofev.columns.str.strip()
ofev["forecast_date"] = pd.to_datetime(ofev["forecast_date"])
ofev["datetime"] = pd.to_datetime(ofev["datetime"])
ofev["discharge_m3s"] = ofev["discharge_m3s"].astype(str).str.replace(",",".").astype(float)
ofev["lead_time_h"] = (ofev["datetime"] - ofev["forecast_date"]).dt.total_seconds()/3600
ofev = ofev.rename(columns={"datetime":"valid_time","discharge_m3s":"Q_ofev"})
ofev = ofev.set_index("valid_time")

# ============================================================
# LOOP
# ============================================================

all_metrics = []

for LT in LEAD_TIMES:

    print(f"\n========== LT {LT}h ==========")

    ml_lt = ml[np.round(ml["lead_time_h"])==LT][["Q_ml"]]
    hyd_lt = hyd[np.round(hyd["lead_time_h"])==LT][["Q_hyd"]]
    cnr_lt = cnr[np.round(cnr["lead_time_h"])==LT][["Q_cnr"]]
    
    # filtre NORAIN
    ofev = ofev[ofev["model"] != "NORAIN"]  
    # sélection lead time
    ofev_lt = ofev[np.round(ofev["lead_time_h"])==LT]
    # médiane propre
    ofev_lt = ofev_lt.groupby(ofev_lt.index)["Q_ofev"].median().to_frame()

    series = [s for s in [ml_lt, hyd_lt, cnr_lt, ofev_lt] if len(s)>0]

    start_common = max(s.index.min() for s in series)
    end_common   = min(s.index.max() for s in series)

    obs_plot  = obs_h.loc[start_common:end_common]
    ml_plot   = ml_lt.loc[start_common:end_common]
    hyd_plot  = hyd_lt.loc[start_common:end_common]
    cnr_plot  = cnr_lt.loc[start_common:end_common]
    ofev_plot = ofev_lt.loc[start_common:end_common]

    # ============================================================
    # METRICS
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

        # ================= EVENT-BASED =================

        events = get_flood_events(df["Q_obs"], FLOOD_THR)

        req_list, tp_list, rer_list = [], [], []

        for start, end in events:

            # fenêtre élargie (CRUCIAL)
            start_win = start - pd.Timedelta(days=2)
            end_win   = end + pd.Timedelta(days=2)

            df_event = df.loc[start_win:end_win]

            if len(df_event) < 5:
                continue

            obs_e = df_event["Q_obs"].values
            sim_e = df_event.iloc[:,1].values
            idx_e = df_event.index

            # ignorer si pas de crue simulée
            if np.max(sim_e) < FLOOD_THR:
                continue

            req_list.append(peak_error_event(obs_e, sim_e))
            tp_list.append(peak_timing_event(obs_e, sim_e, idx_e))
            rer_list.append(volume_error_event(obs_e, sim_e))

        REQ = np.median(req_list) if len(req_list)>0 else np.nan
        TP  = np.median(tp_list) if len(tp_list)>0 else np.nan
        RER = np.median(rer_list) if len(rer_list)>0 else np.nan

        # ================= SAVE =================

        all_metrics.append({
            "lead_time": LT,
            "model": name,
            "NSE": nse(obs_v, sim_v),
            "KGE": kge(obs_v, sim_v),
            "RMSE": rmse(obs_v, sim_v),
            "REQ_%": REQ,
            "TP_h": TP,
            "RER_%": RER,
            "MAPE_high_%": mape_high_flows(obs_v, sim_v, FLOOD_THR)
        })

# ============================================================
# SAVE
# ============================================================

metrics = pd.DataFrame(all_metrics)
metrics.to_csv(output_dir / "metrics_all_models.csv", index=False)

print(metrics)
print("finished")